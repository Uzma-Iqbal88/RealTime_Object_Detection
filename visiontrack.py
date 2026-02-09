"""
VisionTrack AI - Real-Time Object Detection and Tracking System
================================================================

This is the main application file that integrates YOLOv8 object detection
with Deep SORT tracking to provide real-time multi-object tracking.

Author: [Your Name]
Project: CodeAlpha AI Internship
Date: February 2026
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Import utility functions
from utils import (
    draw_boxes,
    calculate_fps,
    get_color_for_class,
    save_video_writer,
    print_banner
)


class VisionTrackAI:
    """
    Main class for VisionTrack AI system.
    
    This class handles:
    - Video input (webcam or file)
    - YOLOv8 object detection
    - Deep SORT tracking
    - Visualization and output
    """
    
    def __init__(self, args):
        """
        Initialize VisionTrack AI system.
        
        Args:
            args: Command-line arguments containing configuration
        """
        self.args = args
        
        # Print welcome banner
        print_banner()
        
        # Initialize YOLOv8 detector
        print("🔄 Loading YOLOv8 model...")
        self.detector = YOLO('yolov8n.pt')  # Using YOLOv8 nano (fastest)
        print("✅ YOLOv8 model loaded successfully!")
        
        # Initialize Deep SORT tracker
        print("🔄 Initializing Deep SORT tracker...")
        self.tracker = DeepSort(
            max_age=30,              # Maximum frames to keep track alive without detection
            n_init=3,                # Number of frames to confirm a track
            max_iou_distance=0.7,    # Maximum IoU distance for matching
            max_cosine_distance=0.3, # Maximum cosine distance for appearance matching
            nn_budget=100            # Maximum size of appearance feature budget
        )
        print("✅ Deep SORT tracker initialized!")
        
        # Initialize video capture
        self.setup_video_capture()
        
        # Initialize video writer if saving output
        self.video_writer = None
        if args.save_output:
            self.setup_video_writer()
        
        # FPS calculation variables
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        
        # Statistics
        self.total_frames_processed = 0
        self.total_detections = 0
        
    def setup_video_capture(self):
        """
        Setup video capture from webcam or video file.
        """
        source = self.args.source
        
        # If source is a number (webcam index)
        if source.isdigit():
            source = int(source)
            print(f"📹 Opening webcam (index: {source})...")
        else:
            print(f"📹 Opening video file: {source}")
        
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"❌ Error: Could not open video source: {source}")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✅ Video source opened successfully!")
        print(f"   Resolution: {self.frame_width}x{self.frame_height}")
        print(f"   FPS: {self.video_fps}")
        
    def setup_video_writer(self):
        """
        Setup video writer to save output video.
        """
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"visiontrack_output_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.video_fps,
            (self.frame_width, self.frame_height)
        )
        
        print(f"💾 Output will be saved to: {output_path}")
        
    def detect_objects(self, frame):
        """
        Detect objects in frame using YOLOv8.
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            detections: List of detections in Deep SORT format
        """
        # Run YOLOv8 inference
        results = self.detector(frame, conf=self.args.conf, verbose=False)
        
        # Extract detections
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Get class ID and name
                class_id = int(box.cls[0])
                class_name = self.detector.names[class_id]
                
                # Convert to Deep SORT format: ([x, y, w, h], confidence, class_name)
                bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to xywh format
                
                detections.append((bbox, confidence, class_name))
                
                self.total_detections += 1
        
        return detections
    
    def track_objects(self, frame, detections):
        """
        Track detected objects using Deep SORT.
        
        Args:
            frame: Input video frame
            detections: List of detections from YOLO
            
        Returns:
            tracks: List of tracked objects with IDs
        """
        # Update tracker with new detections
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        return tracks
    
    def draw_results(self, frame, tracks):
        """
        Draw bounding boxes, labels, and tracking IDs on frame.
        
        Args:
            frame: Input video frame
            tracks: List of tracked objects
            
        Returns:
            frame: Annotated frame
        """
        active_tracks = 0
        
        for track in tracks:
            # Only draw confirmed tracks
            if not track.is_confirmed():
                continue
            
            active_tracks += 1
            
            # Get track ID
            track_id = track.track_id
            
            # Get bounding box
            bbox = track.to_ltrb()  # Convert to (left, top, right, bottom)
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get class name
            class_name = track.get_det_class()
            
            # Get color for this class
            color = get_color_for_class(class_name)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"ID:{track_id} {class_name}"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_w, label_h = label_size
            
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 10, y1),
                color,
                -1  # Filled rectangle
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                2
            )
        
        return frame, active_tracks
    
    def draw_info_panel(self, frame, active_tracks):
        """
        Draw information panel with FPS and statistics.
        
        Args:
            frame: Input video frame
            active_tracks: Number of currently tracked objects
            
        Returns:
            frame: Frame with info panel
        """
        # Calculate FPS
        self.fps_frame_count += 1
        elapsed_time = time.time() - self.fps_start_time
        
        if elapsed_time > 1.0:  # Update FPS every second
            self.current_fps = self.fps_frame_count / elapsed_time
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
        
        # Create semi-transparent overlay for info panel
        overlay = frame.copy()
        panel_height = 80
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw title
        cv2.putText(
            frame,
            "VisionTrack AI - Real-Time Detection & Tracking",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),  # Cyan
            2
        )
        
        # Draw statistics
        stats_text = f"FPS: {self.current_fps:.1f} | Objects: {active_tracks} | Frame: {self.total_frames_processed}"
        cv2.putText(
            frame,
            stats_text,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White
            2
        )
        
        # Draw instructions
        cv2.putText(
            frame,
            "Press 'q' to quit | 's' to screenshot",
            (frame.shape[1] - 350, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),  # Light gray
            1
        )
        
        return frame
    
    def run(self):
        """
        Main processing loop for VisionTrack AI.
        """
        print("\n🚀 Starting VisionTrack AI...")
        print("Press 'q' to quit, 's' to save screenshot\n")
        
        try:
            while True:
                # Read frame from video source
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️  End of video or cannot read frame")
                    break
                
                self.total_frames_processed += 1
                
                # Step 1: Detect objects using YOLOv8
                detections = self.detect_objects(frame)
                
                # Step 2: Track objects using Deep SORT
                tracks = self.track_objects(frame, detections)
                
                # Step 3: Draw results on frame
                frame, active_tracks = self.draw_results(frame, tracks)
                
                # Step 4: Draw info panel
                frame = self.draw_info_panel(frame, active_tracks)
                
                # Save frame to output video if enabled
                if self.video_writer is not None:
                    self.video_writer.write(frame)
                
                # Display frame
                if not self.args.no_display:
                    cv2.imshow('VisionTrack AI', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting VisionTrack AI...")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_dir = Path('screenshots')
                    screenshot_dir.mkdir(exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    screenshot_path = screenshot_dir / f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(str(screenshot_path), frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                elif key == ord('p'):
                    # Pause
                    print("⏸️  Paused. Press any key to continue...")
                    cv2.waitKey(0)
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Release resources and cleanup.
        """
        print("\n🧹 Cleaning up...")
        
        # Print final statistics
        print(f"\n📊 Final Statistics:")
        print(f"   Total frames processed: {self.total_frames_processed}")
        print(f"   Total detections: {self.total_detections}")
        print(f"   Average FPS: {self.current_fps:.1f}")
        
        # Release video capture
        self.cap.release()
        
        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()
            print("💾 Output video saved successfully!")
        
        # Close all windows
        cv2.destroyAllWindows()
        
        print("✅ VisionTrack AI terminated successfully!")


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='VisionTrack AI - Real-Time Object Detection and Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with webcam
  python visiontrack.py --source 0
  
  # Run with video file
  python visiontrack.py --source video.mp4
  
  # Save output and show FPS
  python visiontrack.py --source 0 --save-output --conf 0.6
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source: webcam index (0, 1, ...) or video file path'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold for detection (0.0 to 1.0, default: 0.5)'
    )
    
    parser.add_argument(
        '--save-output',
        action='store_true',
        help='Save processed video to output folder'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without displaying video (useful for headless systems)'
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for VisionTrack AI application.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create and run VisionTrack AI system
    visiontrack = VisionTrackAI(args)
    visiontrack.run()


if __name__ == '__main__':
    main()
