#!/usr/bin/env python3
"""3D holistic animation with OpenGL rendering.

Uses MediaPipe Holistic for pose, hands, and face detection with OpenGL 3D rendering.
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


class HolisticDetector:
    """Handles holistic detection from webcam or video file."""
    
    def __init__(self, source=0):
        self.source = source  # Can be camera index or video file path
        self.is_video_file = isinstance(source, str)
        self.cap = None
        self.holistic = None
        self.current_results = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.loop_video = True  # Whether to loop video files
        
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def start(self):
        """Start the holistic detection thread."""
        print("Loading MediaPipe Holistic model...")
        
        self.cap = cv2.VideoCapture(self.source)
        
        if self.is_video_file:
            print(f"Loading video file: {self.source}")
        if not self.cap.isOpened():
            error_msg = f"Cannot open video file: {self.source}" if self.is_video_file else f"Cannot open camera {self.source}"
            raise RuntimeError(error_msg)
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print("Holistic detection started!")
        
    def _detection_loop(self):
        """Continuously detect holistic from webcam or video."""
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    # If video file, loop it
                    if self.is_video_file and self.loop_video:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        continue
                
                # Flip for mirror effect (only for webcam, not video files)
                if not self.is_video_file:
                    frame = cv2.flip(frame, 1)
                
                # Process frame
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame)
                
                # Update current results
                with self.lock:
                    self.current_results = results
    
    def get_results(self):
        """Get the current holistic results (thread-safe)."""
        with self.lock:
            return self.current_results
    
    def stop(self):
        """Stop holistic detection."""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()


class Character3D:
    """3D character that animates based on holistic detection."""
    
    def __init__(self, show_hands=True, show_face=True):
        self.results = None
        self.mp_holistic = mp.solutions.holistic
        self.smoothing_factor = 0.3
        self.smoothed_pose = {}
        self.smoothed_left_hand = {}
        self.smoothed_right_hand = {}
        self.show_hands = show_hands
        self.show_face = show_face
        
    def update_results(self, results):
        """Update character with holistic results."""
        self.results = results
    
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
    
    def draw(self):
        """Draw the 3D holistic character."""
        if self.results is None:
            return
        
        scale = 2.5
        
        # Draw pose
        if self.results.pose_landmarks:
            self._draw_pose(scale)
        
        # Draw hands (if enabled)
        if self.show_hands:
            if self.results.left_hand_landmarks:
                self._draw_hand(self.results.left_hand_landmarks, (0.3, 0.6, 1.0), scale, 'left')
                # Draw connection from left elbow to left wrist
                self._draw_elbow_to_wrist('left', scale)
            
            if self.results.right_hand_landmarks:
                self._draw_hand(self.results.right_hand_landmarks, (0.3, 1.0, 0.6), scale, 'right')
                # Draw connection from right elbow to right wrist
                self._draw_elbow_to_wrist('right', scale)
        
        # Draw face key points (if enabled)
        if self.show_face and self.results.face_landmarks:
            self._draw_face(scale)
    
    def _draw_pose(self, scale):
        """Draw pose landmarks and connections."""
        landmarks = self.results.pose_landmarks.landmark
        
        # Face landmarks to exclude: 0-10 (nose, eyes, ears, mouth)
        # Hand landmarks: 15-22 (wrists and hand points)
        if self.show_hands:
            # When showing hands, exclude wrists and hand landmarks from pose
            excluded_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22}
        else:
            # When not showing hands, only exclude face, keep hand landmarks in pose
            excluded_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        
        # Draw connections
        glLineWidth(6.0)
        glColor3f(1.0, 0.4, 0.4)
        glBegin(GL_LINES)
        
        for connection in self.mp_holistic.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            # Skip connections that involve hand or face landmarks
            if start_idx in excluded_indices or end_idx in excluded_indices:
                continue
            
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
        
        # Draw joints (excluding hand and face nodes)
        glColor3f(1.0, 0.6, 0.6)
        for idx, lm in enumerate(landmarks):
            # Skip hand and face landmarks
            if idx in excluded_indices:
                continue
                
            if lm.visibility < 0.5:
                continue
            
            smooth = self._smooth_landmark(lm, self.smoothed_pose, idx)
            pos = self._get_3d_position(*smooth, scale)
            
            glPushMatrix()
            glTranslatef(*pos)
            glutSolidSphere(0.05, 10, 10)
            glPopMatrix()
    
    def _draw_hand(self, hand_landmarks, color, scale, hand_type):
        """Draw hand landmarks and connections."""
        landmarks = hand_landmarks.landmark
        cache = self.smoothed_left_hand if hand_type == 'left' else self.smoothed_right_hand
        
        # Draw connections with thinner lines
        glLineWidth(4)
        glColor3f(*color)
        glBegin(GL_LINES)
        
        for connection in self.mp_holistic.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            start_smooth = self._smooth_landmark(start_lm, cache, f"{hand_type}_{start_idx}")
            end_smooth = self._smooth_landmark(end_lm, cache, f"{hand_type}_{end_idx}")
            
            start_pos = self._get_3d_position(*start_smooth, scale)
            end_pos = self._get_3d_position(*end_smooth, scale)
            
            glVertex3f(*start_pos)
            glVertex3f(*end_pos)
        
        glEnd()
        
        # Draw joints
        lighter_color = tuple(min(1.0, c + 0.2) for c in color)
        glColor3f(*lighter_color)
        for idx, lm in enumerate(landmarks):
            smooth = self._smooth_landmark(lm, cache, f"{hand_type}_{idx}")
            pos = self._get_3d_position(*smooth, scale)
            
            glPushMatrix()
            glTranslatef(*pos)
            # Smaller spheres for finger joints - 0.025 instead of 0.04
            glutSolidSphere(0.025, 8, 8)
            glPopMatrix()
    
    def _draw_elbow_to_wrist(self, hand_type, scale):
        """Draw connection from elbow to wrist (hand base)."""
        if not self.results.pose_landmarks:
            return
        
        # Get elbow from pose landmarks
        # Left elbow: index 13, Right elbow: index 14
        elbow_idx = 13 if hand_type == 'left' else 14
        elbow_lm = self.results.pose_landmarks.landmark[elbow_idx]
        
        if elbow_lm.visibility < 0.5:
            return
        
        # Get wrist from hand landmarks
        hand_landmarks = (self.results.left_hand_landmarks if hand_type == 'left' 
                         else self.results.right_hand_landmarks)
        
        if not hand_landmarks:
            return
        
        # Wrist is always index 0 in hand landmarks
        wrist_lm = hand_landmarks.landmark[0]
        
        # Get smoothed positions
        cache = self.smoothed_left_hand if hand_type == 'left' else self.smoothed_right_hand
        elbow_smooth = self._smooth_landmark(elbow_lm, self.smoothed_pose, elbow_idx)
        wrist_smooth = self._smooth_landmark(wrist_lm, cache, f"{hand_type}_0")
        
        elbow_pos = self._get_3d_position(*elbow_smooth, scale)
        wrist_pos = self._get_3d_position(*wrist_smooth, scale)
        
        # Draw connection
        color = (0.3, 0.6, 1.0) if hand_type == 'left' else (0.3, 1.0, 0.6)
        glLineWidth(5.0)
        glColor3f(*color)
        glBegin(GL_LINES)
        glVertex3f(*elbow_pos)
        glVertex3f(*wrist_pos)
        glEnd()
    
    def _draw_face(self, scale):
        """Draw face as a simple head sphere."""
        if not self.results.pose_landmarks:
            return
            
        landmarks = self.results.face_landmarks.landmark
        pose_landmarks = self.results.pose_landmarks.landmark
        
        # Get face center from face landmarks
        # Use key face points to determine head center
        nose_tip = landmarks[1]
        forehead = landmarks[10]
        chin = landmarks[152]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        
        # Calculate face center
        face_x = np.mean([nose_tip.x, forehead.x, chin.x, left_cheek.x, right_cheek.x])
        face_y = np.mean([nose_tip.y, forehead.y, chin.y, left_cheek.y, right_cheek.y])
        face_z = np.mean([nose_tip.z, forehead.z, chin.z, left_cheek.z, right_cheek.z])
        
        head_center_pos = self._get_3d_position(face_x, face_y, face_z, scale)
        
        glColor3f(1.0, 0.85, 0.7)  # Skin tone
        glPushMatrix()
        glTranslatef(*head_center_pos)
        glutSolidSphere(0.18, 20, 20)  # Head sphere (reduced from 0.22 to 0.18)
        glPopMatrix()
        
        # Connect face to neck (to shoulders)
        self._draw_neck_connection(scale, head_center_pos, pose_landmarks)
    
    def _draw_neck_connection(self, scale, head_pos, pose_landmarks):
        """Draw neck connecting head to shoulders."""
        # Get neck/shoulder midpoint from pose
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        
        if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
            return
        
        # Calculate neck base (midpoint between shoulders, slightly raised)
        neck_x = (left_shoulder.x + right_shoulder.x) / 2
        neck_y = (left_shoulder.y + right_shoulder.y) / 2 + 0.03  # Raise slightly
        neck_z = (left_shoulder.z + right_shoulder.z) / 2
        
        neck_pos = self._get_3d_position(neck_x, neck_y, neck_z, scale)
        
        # Calculate bottom of head sphere (chin position)
        chin_pos = (head_pos[0], head_pos[1] + 0.18, head_pos[2])  # Bottom of sphere (updated to match new radius)
        
        # Draw neck as a thicker connection
        glLineWidth(10.0)
        glColor3f(1.0, 0.85, 0.7)  # Skin tone
        glBegin(GL_LINES)
        glVertex3f(*chin_pos)
        glVertex3f(*neck_pos)
        glEnd()
        
        # Draw neck joint spheres for smooth transition
        glColor3f(1.0, 0.82, 0.68)
        glPushMatrix()
        glTranslatef(*chin_pos)
        glutSolidSphere(0.055, 10, 10)
        glPopMatrix()
        
        glPushMatrix()
        glTranslatef(*neck_pos)
        glutSolidSphere(0.075, 10, 10)
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


def main(source=0, show_hands=True, show_face=True):
    """Main application loop."""
    # Initialize Pygame and OpenGL
    pygame.init()
    display = (1000, 700)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
    
    title = f"3D Holistic Animation - {source if isinstance(source, str) else 'Webcam'}"
    pygame.display.set_caption(title)
    
    # Initialize GLUT for drawing spheres
    glutInit()
    
    init_opengl(display[0], display[1])
    
    # Create detector and character
    detector = HolisticDetector(source)
    character = Character3D(show_hands=show_hands, show_face=show_face)
    
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
    
    print("\n3D Holistic Animation Started!")
    print("Controls:")
    print("  ESC/Q - Quit")
    print("  Left Mouse + Drag - Rotate camera")
    print("  Right Mouse + Drag - Pan camera")
    print("  Mouse Wheel - Zoom in/out")
    print("  Arrow Keys - Rotate camera")
    print("  +/- - Zoom in/out")
    print("  R - Reset camera")
    print("\nMove in front of the camera!\n")
    
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
        
        # Update character
        results = detector.get_results()
        character.update_results(results)
        
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
        description='3D holistic animation with OpenGL'
    )
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera index (default: 0 if no video specified)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (alternative to camera)')
    parser.add_argument('--no-loop', action='store_true',
                       help='Don\'t loop video file (only applies to video)')
    parser.add_argument('--no-hands', action='store_true',
                       help='Disable hand landmarks rendering')
    parser.add_argument('--no-face', action='store_true',
                       help='Disable face/head rendering')
    
    args = parser.parse_args()
    
    # Determine source (video file takes precedence over camera)
    if args.video:
        source = args.video
    else:
        source = args.camera if args.camera is not None else 0
    
    # Determine what to show
    show_hands = not args.no_hands
    show_face = not args.no_face
    
    try:
        main(source=source, show_hands=show_hands, show_face=show_face)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
