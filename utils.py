"""
VisionTrack AI - Utility Functions
===================================

This module contains helper functions for VisionTrack AI system.

Author: [Your Name]
Project: CodeAlpha AI Internship
"""

import cv2
import numpy as np
import time
from pathlib import Path


def print_banner():
    """
    Print VisionTrack AI welcome banner.
    """
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              🎯 VisionTrack AI v1.0                         ║
    ║                                                              ║
    ║        Real-Time Object Detection & Tracking System          ║
    ║                                                              ║
    ║        Powered by YOLOv8 + Deep SORT                        ║
    ║                                                              ║
    ║        CodeAlpha AI Internship Project                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def get_color_for_class(class_name):
    """
    Get consistent color for each object class.
    
    Args:
        class_name: Name of the object class
        
    Returns:
        color: BGR color tuple
    """
    # Predefined colors for common classes
    color_map = {
        'person': (0, 255, 0),      # Green
        'car': (255, 0, 0),          # Blue
        'truck': (0, 0, 255),        # Red
        'bus': (255, 255, 0),        # Cyan
        'motorcycle': (255, 0, 255), # Magenta
        'bicycle': (0, 255, 255),    # Yellow
        'dog': (128, 0, 128),        # Purple
        'cat': (255, 128, 0),        # Orange
        'bird': (0, 128, 255),       # Light Blue
        'horse': (128, 255, 0),      # Light Green
    }
    
    # Return predefined color or generate random color
    if class_name in color_map:
        return color_map[class_name]
    else:
        # Generate consistent random color based on class name hash
        np.random.seed(hash(class_name) % 2**32)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color


def draw_boxes(frame, detections, class_names):
    """
    Draw bounding boxes on frame (for YOLO detections only).
    
    Args:
        frame: Input frame
        detections: List of detections
        class_names: Dictionary mapping class IDs to names
        
    Returns:
        frame: Frame with bounding boxes
    """
    for detection in detections:
        bbox, confidence, class_name = detection
        x, y, w, h = map(int, bbox)
        
        # Get color for this class
        color = get_color_for_class(class_name)
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    return frame


def calculate_fps(start_time, frame_count):
    """
    Calculate frames per second.
    
    Args:
        start_time: Start time for FPS calculation
        frame_count: Number of frames processed
        
    Returns:
        fps: Frames per second
    """
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0
    return fps


def save_video_writer(output_path, fps, frame_size):
    """
    Create video writer for saving output.
    
    Args:
        output_path: Path to save video
        fps: Frames per second
        frame_size: Tuple of (width, height)
        
    Returns:
        writer: VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    return writer


def resize_frame(frame, max_width=1280, max_height=720):
    """
    Resize frame to fit within maximum dimensions while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        resized_frame: Resized frame
    """
    height, width = frame.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize frame
    if scale < 1.0:
        resized_frame = cv2.resize(frame, (new_width, new_height))
        return resized_frame
    else:
        return frame


def draw_tracking_trail(frame, track_history, track_id, max_trail_length=30):
    """
    Draw tracking trail for an object.
    
    Args:
        frame: Input frame
        track_history: Dictionary storing position history for each track
        track_id: ID of the track
        max_trail_length: Maximum number of points in trail
        
    Returns:
        frame: Frame with tracking trail
    """
    if track_id in track_history:
        points = track_history[track_id]
        
        # Draw lines connecting points
        for i in range(1, len(points)):
            # Calculate color gradient (fade out older points)
            alpha = i / len(points)
            color = (0, int(255 * alpha), 0)
            
            # Draw line
            cv2.line(frame, points[i - 1], points[i], color, 2)
    
    return frame


def create_heatmap(frame_shape, track_history, alpha=0.5):
    """
    Create heatmap showing areas with most activity.
    
    Args:
        frame_shape: Shape of the frame (height, width, channels)
        track_history: Dictionary storing position history for all tracks
        alpha: Transparency of heatmap overlay
        
    Returns:
        heatmap: Heatmap image
    """
    height, width = frame_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Add Gaussian blobs at each tracked position
    for track_id, points in track_history.items():
        for point in points:
            x, y = point
            # Create small Gaussian blob
            cv2.circle(heatmap, (x, y), 20, 1.0, -1)
    
    # Normalize and apply colormap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap_colored


def count_objects_by_class(tracks):
    """
    Count number of tracked objects by class.
    
    Args:
        tracks: List of tracked objects
        
    Returns:
        counts: Dictionary mapping class names to counts
    """
    counts = {}
    
    for track in tracks:
        if track.is_confirmed():
            class_name = track.get_det_class()
            counts[class_name] = counts.get(class_name, 0) + 1
    
    return counts


def draw_statistics_panel(frame, counts, position='bottom-right'):
    """
    Draw statistics panel showing object counts by class.
    
    Args:
        frame: Input frame
        counts: Dictionary of object counts by class
        position: Position of panel ('bottom-right', 'bottom-left', etc.)
        
    Returns:
        frame: Frame with statistics panel
    """
    if not counts:
        return frame
    
    # Calculate panel size
    panel_width = 250
    panel_height = 30 + len(counts) * 25
    
    # Determine position
    height, width = frame.shape[:2]
    if position == 'bottom-right':
        x = width - panel_width - 10
        y = height - panel_height - 10
    elif position == 'bottom-left':
        x = 10
        y = height - panel_height - 10
    else:  # top-right
        x = width - panel_width - 10
        y = 10
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Draw title
    cv2.putText(
        frame,
        "Object Counts",
        (x + 10, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Draw counts
    y_offset = y + 45
    for class_name, count in sorted(counts.items()):
        text = f"{class_name}: {count}"
        color = get_color_for_class(class_name)
        
        cv2.putText(
            frame,
            text,
            (x + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        y_offset += 25
    
    return frame


def log_detection_event(track_id, class_name, timestamp, log_file='detection_log.txt'):
    """
    Log detection event to file.
    
    Args:
        track_id: ID of tracked object
        class_name: Class of detected object
        timestamp: Timestamp of detection
        log_file: Path to log file
    """
    log_path = Path('logs')
    log_path.mkdir(exist_ok=True)
    
    with open(log_path / log_file, 'a') as f:
        f.write(f"{timestamp},{track_id},{class_name}\n")


def check_zone_intrusion(bbox, zone_coords):
    """
    Check if bounding box intersects with a defined zone.
    
    Args:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        zone_coords: Zone coordinates (x1, y1, x2, y2)
        
    Returns:
        intrusion: True if bbox intersects zone
    """
    x1, y1, x2, y2 = bbox
    zx1, zy1, zx2, zy2 = zone_coords
    
    # Check for intersection
    if x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2:
        return False
    else:
        return True


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        iou: IoU value (0.0 to 1.0)
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    if union > 0:
        iou = intersection / union
    else:
        iou = 0
    
    return iou


def format_time(seconds):
    """
    Format seconds into HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        formatted_time: Time string in HH:MM:SS format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_system_info():
    """
    Get system information for debugging.
    
    Returns:
        info: Dictionary containing system information
    """
    import platform
    import sys
    
    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'python_version': sys.version,
        'opencv_version': cv2.__version__,
    }
    
    return info


def print_system_info():
    """
    Print system information.
    """
    info = get_system_info()
    
    print("\n📊 System Information:")
    print(f"   Platform: {info['platform']}")
    print(f"   Python: {info['python_version'].split()[0]}")
    print(f"   OpenCV: {info['opencv_version']}")
    print()
