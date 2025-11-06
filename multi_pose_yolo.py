#!/usr/bin/env python3
"""Multi-person 3D pose detection using YOLOv8 + MediaPipe Pose with OpenGL rendering.

Uses YOLOv8 for person detection and MediaPipe Pose for skeleton tracking.
"""

import threading
import numpy as np
import cv2
import mediapipe as mp

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Try to import ultralytics (YOLOv8)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    print("Running without YOLO - will only detect one person")
    YOLO_AVAILABLE = False


class MultiPoseDetectorYOLO:
    """Handles multi-person pose detection using YOLO + MediaPipe."""
    
    def __init__(self, source=0, max_people=5, yolo_model='yolov8n.pt', single_person=False):
        self.source = source
        self.is_video_file = isinstance(source, str)
        self.cap = None
        self.current_results = []  # List of (bbox, pose_landmarks) tuples
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.max_people = max_people
        self.frame_skip = 0  # Counter for frame skipping
        self.yolo_interval = 3  # Run YOLO every N frames
        self.cached_bboxes = []  # Cache YOLO detections
        self.single_person = single_person
        
        # YOLO setup (skip if single_person mode)
        if single_person:
            print("Single-person mode enabled (no YOLO, faster rendering)")
            self.yolo = None
        elif YOLO_AVAILABLE:
            print(f"Loading YOLO model: {yolo_model}")
            self.yolo = YOLO(yolo_model)
            print("YOLO model loaded!")
        else:
            self.yolo = None
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Person tracking (to maintain consistent IDs)
        self.tracked_people = {}  # {person_id: {'bbox': (x1,y1,x2,y2), 'last_seen': frame_num}}
        self.next_person_id = 0
        self.current_frame_num = 0
        self.iou_threshold = 0.3  # IoU threshold for matching bboxes
        
    def start(self):
        """Start the pose detection thread."""
        print("Loading MediaPipe Pose model...")
        
        self.cap = cv2.VideoCapture(self.source)
        
        # Optimize camera settings for webcam
        if not self.is_video_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        if self.is_video_file:
            print(f"Loading video file: {self.source}")
        
        if not self.cap.isOpened():
            error_msg = f"Cannot open video file: {self.source}" if self.is_video_file else f"Cannot open camera {self.source}"
            raise RuntimeError(error_msg)
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print("Multi-pose detection started!")
        
    def _detect_people_yolo(self, frame):
        """Detect people using YOLO and return bounding boxes."""
        # Single person mode - return full frame
        if self.single_person:
            h, w = frame.shape[:2]
            return [(0, 0, w, h)]
        
        if not YOLO_AVAILABLE or self.yolo is None:
            # Fallback: return full frame as single person
            h, w = frame.shape[:2]
            return [(0, 0, w, h)]
        
        # Only run YOLO every N frames to save processing
        self.frame_skip += 1
        if self.frame_skip < self.yolo_interval and len(self.cached_bboxes) > 0:
            # Use cached bounding boxes
            return self.cached_bboxes
        
        # Run YOLO detection with optimized settings
        results = self.yolo(frame, classes=[0], verbose=False, half=False, imgsz=640)  # class 0 is 'person'
        
        bboxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        # Limit to max_people and cache
        self.cached_bboxes = bboxes[:self.max_people]
        self.frame_skip = 0
        return self.cached_bboxes
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _track_people(self, bboxes):
        """Assign consistent IDs to detected people based on IoU matching."""
        self.current_frame_num += 1
        
        # Remove people not seen for 30 frames
        max_missing_frames = 30
        self.tracked_people = {
            pid: data for pid, data in self.tracked_people.items()
            if self.current_frame_num - data['last_seen'] < max_missing_frames
        }
        
        tracked_bboxes = []
        used_person_ids = set()
        
        for bbox in bboxes:
            best_match_id = None
            best_iou = 0
            
            # Find best matching tracked person
            for person_id, tracked_data in self.tracked_people.items():
                if person_id in used_person_ids:
                    continue
                
                iou = self._calculate_iou(bbox, tracked_data['bbox'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_id = person_id
            
            # Assign ID
            if best_match_id is not None:
                person_id = best_match_id
            else:
                person_id = self.next_person_id
                self.next_person_id += 1
            
            used_person_ids.add(person_id)
            
            # Update tracking
            self.tracked_people[person_id] = {
                'bbox': bbox,
                'last_seen': self.current_frame_num
            }
            
            tracked_bboxes.append({
                'bbox': bbox,
                'person_id': person_id
            })
        
        return tracked_bboxes
    
    def _detection_loop(self):
        """Continuously detect poses from webcam or video."""
        # Create pose detector instances with optimized settings
        num_detectors = 1 if self.single_person else self.max_people
        pose_detectors = [
            self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1 if self.single_person else 0,  # Higher quality for single person
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            for _ in range(num_detectors)
        ]
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    if self.is_video_file:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        continue
                
                # Flip for mirror effect (only for webcam)
                if not self.is_video_file:
                    frame = cv2.flip(frame, 1)
                
                h, w = frame.shape[:2]
                
                # Detect people with YOLO
                person_bboxes = self._detect_people_yolo(frame)
                
                # Track people to maintain consistent IDs
                tracked_bboxes = self._track_people(person_bboxes)
                
                # Process each detected person
                results_list = []
                
                for tracked in tracked_bboxes:
                    bbox = tracked['bbox']
                    person_id = tracked['person_id']
                    
                    if person_id >= self.max_people:
                        continue
                    
                    x1, y1, x2, y2 = bbox
                    
                    # Add padding to bbox
                    pad_x = int((x2 - x1) * 0.1)
                    pad_y = int((y2 - y1) * 0.1)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(w, x2 + pad_x)
                    y2 = min(h, y2 + pad_y)
                    
                    # Crop person from frame
                    person_crop = frame[y1:y2, x1:x2]
                    
                    if person_crop.size == 0:
                        continue
                    
                    # Process with MediaPipe Pose (don't resize - keep original crop dimensions)
                    person_crop.flags.writeable = False
                    crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    
                    # Use detector based on person_id (cycle through available detectors)
                    detector_idx = person_id % len(pose_detectors)
                    result = pose_detectors[detector_idx].process(crop_rgb)
                    
                    if result.pose_landmarks:
                        # Adjust landmarks to full frame coordinates
                        adjusted_landmarks = self._adjust_landmarks_to_frame(
                            result.pose_landmarks, 
                            x1, y1, x2, y2, w, h
                        )
                        results_list.append({
                            'landmarks': adjusted_landmarks,
                            'bbox': bbox,
                            'person_id': person_id
                        })
                
                # Update current results
                with self.lock:
                    self.current_results = results_list
        finally:
            for detector in pose_detectors:
                detector.close()
    
    def _adjust_landmarks_to_frame(self, pose_landmarks, x1, y1, x2, y2, frame_w, frame_h):
        """Adjust landmark coordinates from crop to full frame."""
        from mediapipe.framework.formats import landmark_pb2
        
        adjusted = landmark_pb2.NormalizedLandmarkList()
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        for lm in pose_landmarks.landmark:
            new_lm = adjusted.landmark.add()
            # Convert from crop coordinates to frame coordinates
            new_lm.x = (lm.x * crop_w + x1) / frame_w
            new_lm.y = (lm.y * crop_h + y1) / frame_h
            new_lm.z = lm.z
            new_lm.visibility = lm.visibility
        
        return adjusted
    
    def get_results(self):
        """Get the current pose results (thread-safe)."""
        with self.lock:
            return self.current_results.copy()
    
    def stop(self):
        """Stop pose detection."""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()


class Character3D:
    """3D character that animates based on pose detection."""
    
    def __init__(self, color=None, person_id=0):
        self.pose_landmarks = None
        self.mp_pose = mp.solutions.pose
        self.smoothing_factor = 0.3
        self.smoothed_pose = {}
        self.color = color if color else (1.0, 0.4, 0.4)
        self.person_id = person_id
        self.offset_x = 0  # X offset for spacing multiple people
        
    def update_landmarks(self, pose_landmarks):
        """Update character with pose results."""
        self.pose_landmarks = pose_landmarks
    
    def set_offset(self, offset_x):
        """Set horizontal offset for this character."""
        self.offset_x = offset_x
    
    def _smooth_landmark(self, landmark, cache, key):
        """Apply smoothing to a landmark."""
        if key not in cache:
            cache[key] = (landmark.x, landmark.y, landmark.z)
        else:
            prev = cache[key]
            cache[key] = (
                self.smoothing_factor * landmark.x + (1 - self.smoothing_factor) * prev[0],
                self.smoothing_factor * landmark.y + (1 - self.smoothing_factor) * prev[1],
                self.smoothing_factor * landmark.z + (1 - self.smoothing_factor) * prev[2]
            )
        return cache[key]
    
    def _get_3d_position(self, x, y, z, scale=2.5):
        """Convert normalized coordinates to 3D position."""
        x_3d = (x - 0.5) * scale + self.offset_x
        y_3d = (0.5 - y) * scale
        z_3d = -z * scale * 0.5
        return (x_3d, y_3d, z_3d)
    
    def draw(self, scale=2.5):
        """Draw the 3D pose character."""
        if self.pose_landmarks is None:
            return
        
        landmarks = self.pose_landmarks.landmark
        
        # Draw connections
        glLineWidth(6.0)
        glColor3f(*self.color)
        glBegin(GL_LINES)
        
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            if start_lm.visibility < 0.5 or end_lm.visibility < 0.5:
                continue
            
            start_smooth = self._smooth_landmark(start_lm, self.smoothed_pose, start_idx)
            end_smooth = self._smooth_landmark(end_lm, self.smoothed_pose, end_idx)
            
            start_pos = self._get_3d_position(*start_smooth, scale)
            end_pos = self._get_3d_position(*end_smooth, scale)
            
            glVertex3f(*start_pos)
            glVertex3f(*end_pos)
        
        glEnd()
        
        # Draw joints
        lighter_color = tuple(min(1.0, c + 0.2) for c in self.color)
        glColor3f(*lighter_color)
        for idx, lm in enumerate(landmarks):
            if lm.visibility < 0.5:
                continue
            
            smooth = self._smooth_landmark(lm, self.smoothed_pose, idx)
            pos = self._get_3d_position(*smooth, scale)
            
            glPushMatrix()
            glTranslatef(*pos)
            glutSolidSphere(0.05, 10, 10)
            glPopMatrix()


def init_opengl(width=1000, height=700):
    """Initialize OpenGL settings."""
    glClearColor(0.1, 0.1, 0.15, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    # Set up lighting
    glLightfv(GL_LIGHT0, GL_POSITION, [5, 5, 5, 1])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
    
    # Set up perspective
    setup_perspective(width, height)


def setup_perspective(width, height):
    """Set up or update the perspective projection."""
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glViewport(0, 0, width, height)


def draw_grid():
    """Draw a reference grid on the ground."""
    glDisable(GL_LIGHTING)
    glColor3f(0.3, 0.3, 0.3)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    
    grid_size = 5
    grid_y = -1.8
    for i in range(-grid_size, grid_size + 1):
        # Lines parallel to X axis
        glVertex3f(-grid_size, grid_y, i)
        glVertex3f(grid_size, grid_y, i)
        # Lines parallel to Z axis
        glVertex3f(i, grid_y, -grid_size)
        glVertex3f(i, grid_y, grid_size)
    
    glEnd()
    glEnable(GL_LIGHTING)


def main(source=0, max_people=5, yolo_model='yolov8n.pt', single_person=False):
    """Main application loop."""
    # Initialize Pygame and OpenGL
    pygame.init()
    display = (1200, 700)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
    
    mode = "Single-Person (Fast)" if single_person else "Multi-Person (YOLO+Pose)"
    title = f"{mode} - {source if isinstance(source, str) else 'Webcam'}"
    pygame.display.set_caption(title)
    
    # Initialize GLUT for drawing spheres
    glutInit()
    
    init_opengl(display[0], display[1])
    
    # Create detector
    detector = MultiPoseDetectorYOLO(source, max_people=max_people, yolo_model=yolo_model, single_person=single_person)
    
    # Create characters with different colors
    colors = [
        (1.0, 0.3, 0.3),  # Red
        (0.3, 1.0, 0.3),  # Green
        (0.3, 0.3, 1.0),  # Blue
        (1.0, 1.0, 0.3),  # Yellow
        (1.0, 0.3, 1.0),  # Magenta
        (0.3, 1.0, 1.0),  # Cyan
        (1.0, 0.6, 0.2),  # Orange
        (0.6, 0.3, 0.9),  # Purple
    ]
    
    characters = [Character3D(color=colors[i % len(colors)], person_id=i) for i in range(max_people)]
    
    # Start detection
    detector.start()
    
    # Camera settings
    camera_distance = 7.0
    camera_angle_x = 0
    camera_angle_y = 0
    camera_pan_x = 0.0
    camera_pan_y = 0.0
    
    # Mouse interaction
    mouse_dragging = False
    mouse_button = None
    last_mouse_pos = (0, 0)
    
    clock = pygame.time.Clock()
    running = True
    
    print(f"\n3D Pose Detection Started!")
    if single_person:
        print("Mode: Single-person (fast, no YOLO)")
    elif YOLO_AVAILABLE:
        print(f"Mode: Multi-person with YOLO model: {yolo_model}")
    else:
        print("Mode: Single-person (YOLO not available)")
    print("\nControls:")
    print("  ESC/Q - Quit")
    print("  Left Mouse + Drag - Rotate camera")
    print("  Right Mouse + Drag - Pan camera")
    print("  Mouse Wheel - Zoom in/out")
    print("  Arrow Keys - Rotate camera")
    print("  +/- - Zoom in/out")
    print("  R - Reset camera")
    print(f"\nDetecting up to {max_people} people\n")
    
    current_width, current_height = display
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                current_width, current_height = event.w, event.h
                pygame.display.set_mode((current_width, current_height), DOUBLEBUF | OPENGL | RESIZABLE)
                setup_perspective(current_width, current_height)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 or event.button == 3:
                    mouse_dragging = True
                    mouse_button = event.button
                    last_mouse_pos = event.pos
                elif event.button == 4:  # Mouse wheel up
                    camera_distance = max(2.0, camera_distance - 0.3)
                elif event.button == 5:  # Mouse wheel down
                    camera_distance = min(15.0, camera_distance + 0.3)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 or event.button == 3:
                    mouse_dragging = False
                    mouse_button = None
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging and mouse_button:
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    
                    if mouse_button == 1:  # Rotate
                        camera_angle_y += dx * 0.5
                        camera_angle_x += -dy * 0.5
                        camera_angle_x = max(-89, min(89, camera_angle_x))
                    elif mouse_button == 3:  # Pan
                        import math
                        angle_y_rad = math.radians(camera_angle_y)
                        pan_speed = 0.003 * camera_distance
                        
                        right_x = math.cos(angle_y_rad)
                        right_z = -math.sin(angle_y_rad)
                        
                        camera_pan_x += dx * pan_speed * right_x
                        camera_pan_y -= dy * pan_speed
                    
                    last_mouse_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    camera_distance = 7.0
                    camera_angle_x = 0
                    camera_angle_y = 0
                    camera_pan_x = 0.0
                    camera_pan_y = 0.0
        
        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            camera_angle_y -= 2
        if keys[pygame.K_RIGHT]:
            camera_angle_y += 2
        if keys[pygame.K_UP]:
            camera_angle_x -= 2
        if keys[pygame.K_DOWN]:
            camera_angle_x += 2
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
            camera_distance = max(2.0, camera_distance - 0.1)
        if keys[pygame.K_MINUS]:
            camera_distance = min(15.0, camera_distance + 0.1)
        
        # Update characters
        results_list = detector.get_results()
        
        # Map person_id to character with consistent color
        active_characters = {}
        for result in results_list:
            person_id = result.get('person_id', 0)
            
            # Get or create character for this person_id
            if person_id not in active_characters:
                if person_id < len(characters):
                    active_characters[person_id] = characters[person_id]
                else:
                    # Create new character if needed (shouldn't happen with proper max_people)
                    color = colors[person_id % len(colors)]
                    active_characters[person_id] = Character3D(color=color, person_id=person_id)
            
            # Update character
            active_characters[person_id].update_landmarks(result['landmarks'])
            active_characters[person_id].set_offset(0)  # No offset - use actual positions
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up camera
        glLoadIdentity()
        
        import math
        angle_x_rad = math.radians(camera_angle_x)
        angle_y_rad = math.radians(camera_angle_y)
        
        cam_x = camera_distance * math.cos(angle_x_rad) * math.sin(angle_y_rad)
        cam_y = camera_distance * math.sin(angle_x_rad)
        cam_z = camera_distance * math.cos(angle_x_rad) * math.cos(angle_y_rad)
        
        look_at_x = camera_pan_x
        look_at_y = camera_pan_y - 0.3
        look_at_z = 0
        
        gluLookAt(
            cam_x + look_at_x, cam_y + look_at_y, cam_z + look_at_z,
            look_at_x, look_at_y, look_at_z,
            0, 1, 0
        )
        
        # Draw scene
        draw_grid()
        
        # Draw all detected people with consistent IDs
        for person_id, character in active_characters.items():
            character.draw()
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    # Cleanup
    detector.stop()
    pygame.quit()
    print("Application closed.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-person 3D pose detection using YOLO + MediaPipe Pose'
    )
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera index (default: 0 if no video specified)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (alternative to camera)')
    parser.add_argument('--max-people', type=int, default=5,
                       help='Maximum number of people to detect (default: 5)')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt',
                       help='YOLO model to use (default: yolov8n.pt, options: yolov8s.pt, yolov8m.pt, etc.)')
    parser.add_argument('--single-person', action='store_true',
                       help='Fast single-person mode (skips YOLO, ~2-3x faster)')
    
    args = parser.parse_args()
    
    # Determine source (video file takes precedence over camera)
    if args.video:
        source = args.video
    else:
        source = args.camera if args.camera is not None else 0
    
    try:
        main(source=source, max_people=args.max_people, yolo_model=args.yolo_model, single_person=args.single_person)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
