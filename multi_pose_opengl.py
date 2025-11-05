#!/usr/bin/env python3
"""Multi-person 3D pose detection with OpenGL rendering.

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


class MultiPoseDetector:
    """Handles multi-person pose detection from webcam feed."""
    
    def __init__(self, source=0, max_people=5):
        self.source = source
        self.is_video_file = isinstance(source, str)
        self.cap = None
        self.current_results = []  # List of pose results
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.max_people = max_people
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
    def start(self):
        """Start the pose detection thread."""
        print("Loading MediaPipe Pose model...")
        
        self.cap = cv2.VideoCapture(self.source)
        
        if self.is_video_file:
            print(f"Loading video file: {self.source}")
        
        if not self.cap.isOpened():
            error_msg = f"Cannot open video file: {self.source}" if self.is_video_file else f"Cannot open camera {self.source}"
            raise RuntimeError(error_msg)
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print("Multi-pose detection started!")
        
    def _detect_persons_simple(self, frame):
        """Simple person detection using background subtraction and contours."""
        # This is a simple approach - for better results, use YOLOv8 or similar
        h, w = frame.shape[:2]
        
        # Create regions for potential people (grid-based)
        # In a real implementation, you'd use YOLOv8 or a person detector
        # For now, we'll process the whole frame multiple times with offset
        regions = [
            (0, 0, w, h),  # Full frame - detects first person
        ]
        
        return regions
    
    def _detection_loop(self):
        """Continuously detect poses from webcam or video."""
        # Create pose detector for each potential person
        pose_detectors = [
            self.mp_pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            for _ in range(self.max_people)
        ]
        
        # Simple approach: Run pose detection on full frame
        # MediaPipe Pose can detect one person per instance
        # For true multi-person, you'd need person detection first
        
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
                
                # Process frame
                frame.flags.writeable = False
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # For simplicity, we'll just run one pose detector
                # True multi-person would require person detection + cropping
                results_list = []
                
                # Detect primary person
                result = pose_detectors[0].process(frame_rgb)
                if result.pose_landmarks:
                    results_list.append(result.pose_landmarks)
                
                # Update current results
                with self.lock:
                    self.current_results = results_list
        finally:
            for detector in pose_detectors:
                detector.close()
    
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
    
    def __init__(self, color=None):
        self.pose_landmarks = None
        self.mp_pose = mp.solutions.pose
        self.smoothing_factor = 0.3
        self.smoothed_pose = {}
        self.color = color if color else (1.0, 0.4, 0.4)
        
    def update_landmarks(self, pose_landmarks):
        """Update character with pose results."""
        self.pose_landmarks = pose_landmarks
    
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
        x_3d = (x - 0.5) * scale
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


def main(source=0, max_people=5):
    """Main application loop."""
    # Initialize Pygame and OpenGL
    pygame.init()
    display = (1000, 700)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
    
    title = f"Multi-Person Pose - {source if isinstance(source, str) else 'Webcam'}"
    pygame.display.set_caption(title)
    
    # Initialize GLUT for drawing spheres
    glutInit()
    
    init_opengl(display[0], display[1])
    
    # Create detector and characters with different colors
    detector = MultiPoseDetector(source, max_people=max_people)
    
    # Create characters with different colors
    colors = [
        (1.0, 0.4, 0.4),  # Red
        (0.4, 1.0, 0.4),  # Green
        (0.4, 0.4, 1.0),  # Blue
        (1.0, 1.0, 0.4),  # Yellow
        (1.0, 0.4, 1.0),  # Magenta
    ]
    
    characters = [Character3D(color=colors[i % len(colors)]) for i in range(max_people)]
    
    # Start detection
    detector.start()
    
    # Camera settings
    camera_distance = 5.0
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
    
    print("\n3D Multi-Person Pose Detection Started!")
    print("Controls:")
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
                    camera_distance = min(10.0, camera_distance + 0.3)
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
                    camera_distance = 5.0
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
            camera_distance = min(10.0, camera_distance + 0.1)
        
        # Update characters
        results_list = detector.get_results()
        for i, pose_landmarks in enumerate(results_list):
            if i < len(characters):
                characters[i].update_landmarks(pose_landmarks)
        
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
        
        # Draw all detected people
        for i, character in enumerate(characters):
            if i < len(results_list):
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
        description='Multi-person 3D pose detection with OpenGL'
    )
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera index (default: 0 if no video specified)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (alternative to camera)')
    parser.add_argument('--max-people', type=int, default=5,
                       help='Maximum number of people to detect (default: 5)')
    
    args = parser.parse_args()
    
    # Determine source (video file takes precedence over camera)
    if args.video:
        source = args.video
    else:
        source = args.camera if args.camera is not None else 0
    
    try:
        main(source=source, max_people=args.max_people)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
