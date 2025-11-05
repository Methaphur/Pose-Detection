#!/usr/bin/env python3
"""Hand painting game using MediaPipe hand tracking.

Right hand index finger draws on the canvas.
Clench left hand to activate/deactivate the brush.
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
import math


class HandPaint:
    """Hand tracking paint application."""
    
    def __init__(self, camera_index=0):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )
        
        # Camera setup
        self.cap = cv2.VideoCapture(camera_index)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Pygame setup
        pygame.init()
        self.width = 1280
        self.height = 720
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Hand Paint - Right hand draws, Left hand peace sign to activate")
        
        # Canvas for drawing
        self.canvas = pygame.Surface((self.width, self.height))
        self.canvas.fill((255, 255, 255))  # White background
        
        # Drawing state
        self.brush_active = False
        self.prev_draw_pos = None
        self.prev_erase_pos = None
        self.brush_color = (0, 0, 255)  # Blue
        self.brush_size = 5
        self.eraser_size = 30
        
        # Font for UI
        self.font = pygame.font.Font(None, 36)
        
        self.clock = pygame.time.Clock()
        
    def is_two_fingers_up(self, hand_landmarks):
        """Check if index and middle fingers are up (peace sign)."""
        # Get landmarks
        # Fingertip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
        # Finger PIP joints: thumb=2, index=6, middle=10, ring=14, pinky=18
        
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        pinky_tip = hand_landmarks.landmark[20]
        pinky_pip = hand_landmarks.landmark[18]
        
        # Check if index and middle fingers are extended (tip higher than PIP joint)
        index_up = index_tip.y < index_pip.y
        middle_up = middle_tip.y < middle_pip.y
        
        # Check if ring and pinky are down (tip lower than PIP joint)
        ring_down = ring_tip.y > ring_pip.y
        pinky_down = pinky_tip.y > pinky_pip.y
        
        # Return true if index and middle are up, and ring and pinky are down
        return index_up and middle_up and ring_down and pinky_down
    
    def is_hand_open(self, hand_landmarks):
        """Check if all fingers are extended (open palm)."""
        # Check if all fingertips are above their PIP joints
        fingers = [
            (8, 6),   # Index
            (12, 10), # Middle
            (16, 14), # Ring
            (20, 18), # Pinky
        ]
        
        fingers_extended = 0
        for tip_idx, pip_idx in fingers:
            tip = hand_landmarks.landmark[tip_idx]
            pip = hand_landmarks.landmark[pip_idx]
            if tip.y < pip.y:  # Finger extended
                fingers_extended += 1
        
        # Consider hand open if at least 3 fingers are extended
        return fingers_extended >= 3
    
    def is_pointing(self, hand_landmarks):
        """Check if only index finger is extended (pointing)."""
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        
        # Index extended, others down
        index_up = index_tip.y < index_pip.y
        middle_down = middle_tip.y > middle_pip.y
        ring_down = ring_tip.y > ring_pip.y
        
        return index_up and middle_down and ring_down
    
    def process_hands(self, results):
        """Process hand tracking results."""
        if not results.multi_hand_landmarks:
            self.prev_draw_pos = None
            self.prev_erase_pos = None
            return
        
        right_hand = None
        left_hand = None
        
        # Identify left and right hands
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            
            if hand_label == "Right":
                right_hand = hand_landmarks
            elif hand_label == "Left":
                left_hand = hand_landmarks
        
        # Check left hand for brush activation (two fingers up = brush on)
        if left_hand:
            self.brush_active = self.is_two_fingers_up(left_hand)
        else:
            self.brush_active = False
        
        # Use right hand for drawing or erasing
        if right_hand and self.brush_active:
            # Check if hand is open (eraser mode) or pointing (draw mode)
            if self.is_hand_open(right_hand):
                # Eraser mode - use palm center
                palm_center = right_hand.landmark[9]  # Middle finger MCP joint
                
                x = int(palm_center.x * self.width)
                y = int(palm_center.y * self.height)
                erase_pos = (x, y)
                
                # Erase by drawing white circles
                if self.prev_erase_pos:
                    # Draw thick white line between positions
                    pygame.draw.line(self.canvas, (255, 255, 255), 
                                   self.prev_erase_pos, erase_pos, self.eraser_size)
                
                # Draw white circle at current position
                pygame.draw.circle(self.canvas, (255, 255, 255), erase_pos, self.eraser_size // 2)
                
                self.prev_erase_pos = erase_pos
                self.prev_draw_pos = None
                
            elif self.is_pointing(right_hand):
                # Draw mode - use index finger tip
                index_tip = right_hand.landmark[8]
                
                x = int(index_tip.x * self.width)
                y = int(index_tip.y * self.height)
                draw_pos = (x, y)
                
                # Draw line from previous position
                if self.prev_draw_pos:
                    pygame.draw.line(self.canvas, self.brush_color, 
                                   self.prev_draw_pos, draw_pos, self.brush_size)
                
                self.prev_draw_pos = draw_pos
                self.prev_erase_pos = None
            else:
                self.prev_draw_pos = None
                self.prev_erase_pos = None
        else:
            self.prev_draw_pos = None
            self.prev_erase_pos = None
    
    def draw_ui(self):
        """Draw UI elements."""
        # Status text
        status = "DRAWING" if self.brush_active else "PAUSED"
        color = (0, 255, 0) if self.brush_active else (255, 0, 0)
        status_text = self.font.render(f"Brush: {status}", True, color)
        self.screen.blit(status_text, (10, 10))
        
        # Instructions
        instructions = [
            "Right hand pointing - Draw",
            "Right hand open - Erase",
            "Left hand peace sign - Activate",
            "C - Clear canvas",
            "1-5 - Change color",
            "+/- - Brush size",
            "ESC - Quit"
        ]
        
        small_font = pygame.font.Font(None, 24)
        y_offset = 50
        for instruction in instructions:
            text = small_font.render(instruction, True, (100, 100, 100))
            self.screen.blit(text, (10, y_offset))
            y_offset += 25
        
        # Draw brush preview
        pygame.draw.circle(self.screen, self.brush_color, 
                         (self.width - 50, 50), self.brush_size)
        pygame.draw.circle(self.screen, (0, 0, 0), 
                         (self.width - 50, 50), self.brush_size, 1)
    
    def run(self):
        """Main application loop."""
        running = True
        
        print("\nHand Paint Started!")
        print("Controls:")
        print("  Right hand pointing (index finger) - Draw on canvas")
        print("  Right hand open (palm) - Erase")
        print("  Left hand peace sign (2 fingers up) - Activate brush/eraser")
        print("  C - Clear canvas")
        print("  1-5 - Change color (red, blue, green, yellow, black)")
        print("  +/- - Increase/decrease brush size")
        print("  ESC - Quit\n")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c:
                        # Clear canvas
                        self.canvas.fill((255, 255, 255))
                    elif event.key == pygame.K_1:
                        self.brush_color = (255, 0, 0)  # Red
                    elif event.key == pygame.K_2:
                        self.brush_color = (0, 0, 255)  # Blue
                    elif event.key == pygame.K_3:
                        self.brush_color = (0, 255, 0)  # Green
                    elif event.key == pygame.K_4:
                        self.brush_color = (255, 255, 0)  # Yellow
                    elif event.key == pygame.K_5:
                        self.brush_color = (0, 0, 0)  # Black
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.brush_size = min(20, self.brush_size + 1)
                    elif event.key == pygame.K_MINUS:
                        self.brush_size = max(1, self.brush_size - 1)
            
            # Read camera frame
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Process hand tracking
            self.process_hands(results)
            
            # Draw on screen
            self.screen.fill((255, 255, 255))
            self.screen.blit(self.canvas, (0, 0))
            
            # Draw hand landmarks overlay (optional - for debugging)
            if results.multi_hand_landmarks:
                # Convert camera frame to pygame surface for overlay
                frame_surface = pygame.surfarray.make_surface(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
                )
                # Scale to match screen
                frame_surface = pygame.transform.scale(frame_surface, (self.width, self.height))
                # Make semi-transparent
                frame_surface.set_alpha(50)
                self.screen.blit(frame_surface, (0, 0))
            
            # Draw UI
            self.draw_ui()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(30)
        
        # Cleanup
        self.cap.release()
        self.hands.close()
        pygame.quit()
        print("Application closed.")


def main():
    """Entry point."""
    try:
        app = HandPaint(camera_index=0)
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
