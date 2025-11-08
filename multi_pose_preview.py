#!/usr/bin/env python3
"""Multi-person pose detection with 2D visualization overlay.

Shows YOLO bounding boxes and MediaPipe pose landmarks on the webcam feed.
"""

import cv2
import numpy as np
import mediapipe as mp

# Try to import ultralytics (YOLOv8)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    print("Running without YOLO - will only detect one person")
    YOLO_AVAILABLE = False


class MultiPosePreview:
    """Real-time multi-person pose detection with visualization."""
    
    def __init__(self, source=0, max_people=5, yolo_model='yolov8n.pt', single_person=False):
        self.source = source
        self.is_video_file = isinstance(source, str)
        self.max_people = max_people
        self.frame_skip = 0
        self.yolo_interval = 3  # Run YOLO every N frames
        self.cached_bboxes = []
        self.single_person = single_person
        
        # YOLO setup (skip if single_person mode)
        if single_person:
            print("Single-person mode enabled (no YOLO)")
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
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create pose detectors
        num_detectors = 1 if single_person else max_people
        self.pose_detectors = [
            self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1 if single_person else 0,  # Higher quality for single person
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            for _ in range(num_detectors)
        ]
        
        # Colors for different people
        self.colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
        ]
        
        # Stats
        self.fps = 0
        self.frame_count = 0
        import time
        self.start_time = time.time()
        
        # Person tracking (to maintain consistent IDs)
        self.tracked_people = {}  # {person_id: {'bbox': (x1,y1,x2,y2), 'last_seen': frame_num}}
        self.next_person_id = 0
        self.current_frame_num = 0
        self.iou_threshold = 0.3  # IoU threshold for matching bboxes
    
    def _detect_people_yolo(self, frame):
        """Detect people using YOLO and return bounding boxes."""
        # Single person mode - return full frame
        if self.single_person:
            h, w = frame.shape[:2]
            return [{'bbox': (0, 0, w, h), 'confidence': 1.0}]
        
        if not YOLO_AVAILABLE or self.yolo is None:
            h, w = frame.shape[:2]
            return [{'bbox': (0, 0, w, h), 'confidence': 1.0}]
        
        # Frame skipping for performance
        self.frame_skip += 1
        if self.frame_skip < self.yolo_interval and len(self.cached_bboxes) > 0:
            return self.cached_bboxes
        
        # Run YOLO detection
        results = self.yolo(frame, classes=[0], verbose=False, half=False, imgsz=640)
        
        bboxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    bboxes.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf)
                    })
        
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
    
    def _track_people(self, detections):
        """Assign consistent IDs to detected people based on IoU matching."""
        self.current_frame_num += 1
        
        # Remove people not seen for 30 frames
        max_missing_frames = 30
        self.tracked_people = {
            pid: data for pid, data in self.tracked_people.items()
            if self.current_frame_num - data['last_seen'] < max_missing_frames
        }
        
        tracked_detections = []
        used_person_ids = set()
        
        for detection in detections:
            bbox = detection['bbox']
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
            
            tracked_detections.append({
                'detection': detection,
                'person_id': person_id
            })
        
        return tracked_detections
    
    def _draw_bbox(self, frame, bbox_data, color, person_id):
        """Draw bounding box with label."""
        x1, y1, x2, y2 = bbox_data['bbox']
        conf = bbox_data.get('confidence', 0)
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"Person {person_id+1}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 10, label_size[1])
        
        # Background for label
        cv2.rectangle(frame, 
                     (x1, label_y - label_size[1] - 4),
                     (x1 + label_size[0] + 4, label_y + 4),
                     color, -1)
        
        # Text
        cv2.putText(frame, label, (x1 + 2, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_pose_landmarks(self, frame, pose_landmarks, color):
        """Draw pose landmarks and connections."""
        # Draw connections
        self.mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    def process_frame(self, frame):
        """Process a single frame and return annotated frame."""
        h, w = frame.shape[:2]
        
        # Create output frame
        output_frame = frame.copy()
        
        # Single person mode - skip YOLO, process full frame
        if self.single_person:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            result = self.pose_detectors[0].process(frame_rgb)
            
            if result.pose_landmarks:
                # Draw pose directly on frame (no bbox)
                color = self.colors[0]
                self._draw_pose_on_frame(output_frame, result.pose_landmarks, color, w, h)
                
                # Draw simple label
                cv2.putText(output_frame, "Person Detected", (20, h - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw stats
            self._draw_stats(output_frame, 1 if result.pose_landmarks else 0)
            return output_frame
        
        # Multi-person mode with YOLO
        # Detect people with YOLO
        person_detections = self._detect_people_yolo(frame)
        
        # Track people to maintain consistent IDs
        tracked_detections = self._track_people(person_detections)
        
        # Process each detected person
        for tracked in tracked_detections:
            detection = tracked['detection']
            person_id = tracked['person_id']
            
            if person_id >= self.max_people:
                continue
            
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Get consistent color for this person ID
            color = self.colors[person_id % len(self.colors)]
            
            # Draw bounding box
            self._draw_bbox(output_frame, detection, color, person_id)
            
            # Add padding to bbox
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            # Crop person
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                continue
            
            # Process with MediaPipe
            person_crop.flags.writeable = False
            crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            
            # Use detector based on person_id (cycle through available detectors)
            detector_idx = person_id % len(self.pose_detectors)
            result = self.pose_detectors[detector_idx].process(crop_rgb)
            
            if result.pose_landmarks:
                # Adjust landmarks to full frame coordinates
                adjusted_landmarks = self._adjust_landmarks(
                    result.pose_landmarks,
                    x1, y1, x2, y2, w, h
                )
                
                # Draw pose on output frame
                self._draw_pose_on_frame(output_frame, adjusted_landmarks, color, w, h)
        
        # Draw stats
        self._draw_stats(output_frame, len(tracked_detections))
        
        return output_frame
    
    def _adjust_landmarks(self, pose_landmarks, x1, y1, x2, y2, frame_w, frame_h):
        """Adjust landmark coordinates from crop to full frame."""
        from mediapipe.framework.formats import landmark_pb2
        
        adjusted = landmark_pb2.NormalizedLandmarkList()
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        for lm in pose_landmarks.landmark:
            new_lm = adjusted.landmark.add()
            new_lm.x = (lm.x * crop_w + x1) / frame_w
            new_lm.y = (lm.y * crop_h + y1) / frame_h
            new_lm.z = lm.z
            new_lm.visibility = lm.visibility
        
        return adjusted
    
    def _draw_pose_on_frame(self, frame, landmarks, color, w, h):
        """Draw pose landmarks on frame."""
        # Check if landmarks are MediaPipe format or adjusted format
        if hasattr(landmarks, 'landmark'):
            # Direct MediaPipe landmarks
            landmark_list = landmarks.landmark
        else:
            # Already adjusted landmarks
            landmark_list = landmarks.landmark
        
        # Draw connections
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_lm = landmark_list[start_idx]
            end_lm = landmark_list[end_idx]
            
            if start_lm.visibility < 0.5 or end_lm.visibility < 0.5:
                continue
            
            start_x = int(start_lm.x * w)
            start_y = int(start_lm.y * h)
            end_x = int(end_lm.x * w)
            end_y = int(end_lm.y * h)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
        
        # Draw landmarks
        for idx, lm in enumerate(landmark_list):
            if lm.visibility < 0.5:
                continue
            
            x = int(lm.x * w)
            y = int(lm.y * h)
            
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
    
    def _draw_stats(self, frame, num_people):
        """Draw FPS and person count."""
        import time
        
        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        
        # Reset every 60 frames
        if self.frame_count >= 60:
            self.frame_count = 0
            self.start_time = time.time()
        
        # Draw stats background
        cv2.rectangle(frame, (10, 10), (250, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 80), (255, 255, 255), 2)
        
        # Draw text
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"People: {num_people}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def run(self):
        """Main loop."""
        cap = cv2.VideoCapture(self.source)
        
        # Optimize camera settings
        if not self.is_video_file:
            # Try multiple resolutions/FPS combos to find what works
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify actual FPS
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera: {actual_width:.0f}x{actual_height:.0f} @ {actual_fps:.0f} FPS")
        
        if self.is_video_file:
            print(f"Loading video file: {self.source}")
        
        if not cap.isOpened():
            error_msg = f"Cannot open video file: {self.source}" if self.is_video_file else f"Cannot open camera {self.source}"
            raise RuntimeError(error_msg)
        
        print("\n=== Multi-Person Pose Preview ===")
        if self.single_person:
            print("Mode: Single-person (fast)")
        elif YOLO_AVAILABLE:
            print(f"Mode: Multi-person (YOLO enabled)")
        else:
            print("Mode: Single-person (YOLO not available)")
        print(f"Max people: {self.max_people}")
        print("\nControls:")
        print("  Q/ESC - Quit")
        print("  SPACE - Pause/Resume")
        print("  S - Save screenshot")
        print("  F - Toggle fullscreen")
        print()
        
        paused = False
        fullscreen = False
        window_name = "Multi-Person Pose Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        if self.is_video_file:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            break
                    
                    # Flip for mirror effect (webcam only)
                    if not self.is_video_file:
                        frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    output_frame = self.process_frame(frame)
                else:
                    # Keep showing last frame when paused
                    pass
                
                # Show frame
                cv2.imshow(window_name, output_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord(' '):  # Space
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"{status}")
                elif key == ord('s'):  # Save screenshot
                    import time
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, output_frame)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('f'):  # Toggle fullscreen
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            for detector in self.pose_detectors:
                detector.close()
            print("Stopped")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-person pose detection with 2D visualization'
    )
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera index (default: 0 if no video specified)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file or stream URL')
    parser.add_argument('--max-people', type=int, default=5,
                       help='Maximum number of people to detect (default: 5)')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt',
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--single-person', action='store_true',
                       help='Fast single-person mode (skips YOLO, ~2-3x faster)')
    
    args = parser.parse_args()
    
    # Determine source
    if args.video:
        source = args.video
    else:
        source = args.camera if args.camera is not None else 0
    
    try:
        preview = MultiPosePreview(
            source=source,
            max_people=args.max_people,
            yolo_model=args.yolo_model,
            single_person=args.single_person
        )
        preview.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
