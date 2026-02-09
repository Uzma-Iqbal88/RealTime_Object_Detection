"""
VisionTrack AI - Model Downloader
==================================

This script downloads the YOLOv8 model for object detection.

Author: [Your Name]
Project: CodeAlpha AI Internship
"""

import os
from pathlib import Path
from ultralytics import YOLO


def download_yolov8_model(model_name='yolov8n.pt'):
    """
    Download YOLOv8 model.
    
    Args:
        model_name: Name of the model to download
                   Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    """
    print("=" * 70)
    print("VisionTrack AI - YOLOv8 Model Downloader")
    print("=" * 70)
    print()
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / model_name
    
    # Check if model already exists
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        
        response = input("\nDo you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    print(f"\n🔄 Downloading YOLOv8 model: {model_name}")
    print("This may take a few minutes depending on your internet speed...")
    print()
    
    try:
        # Download model (Ultralytics will handle the download)
        model = YOLO(model_name)
        
        print(f"\n✅ Model downloaded successfully!")
        print(f"   Location: {model_path.absolute()}")
        
        # Display model info
        print(f"\n📊 Model Information:")
        print(f"   Model: {model_name}")
        print(f"   Task: Object Detection")
        print(f"   Classes: 80 (COCO dataset)")
        print(f"   Input size: 640×640")
        
        # Test model
        print(f"\n🧪 Testing model...")
        results = model.predict(source='https://ultralytics.com/images/bus.jpg', verbose=False)
        print(f"✅ Model test successful!")
        print(f"   Detected {len(results[0].boxes)} objects in test image")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("Please check your internet connection and try again.")
        return
    
    print("\n" + "=" * 70)
    print("Download complete! You can now run VisionTrack AI.")
    print("=" * 70)


def download_all_models():
    """
    Download all YOLOv8 model variants.
    """
    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    
    print("Downloading all YOLOv8 models...")
    print("This will download approximately 250 MB of data.")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    for model in models:
        download_yolov8_model(model)
        print()


def show_model_info():
    """
    Display information about available YOLOv8 models.
    """
    print("=" * 70)
    print("YOLOv8 Model Variants")
    print("=" * 70)
    print()
    
    models_info = [
        {
            'name': 'YOLOv8n',
            'file': 'yolov8n.pt',
            'size': '6.2 MB',
            'params': '3.2M',
            'speed': '2-3 ms',
            'mAP': '37.3',
            'use_case': 'Real-time, embedded devices'
        },
        {
            'name': 'YOLOv8s',
            'file': 'yolov8s.pt',
            'size': '21.5 MB',
            'params': '11.2M',
            'speed': '3-4 ms',
            'mAP': '44.9',
            'use_case': 'Balanced speed and accuracy'
        },
        {
            'name': 'YOLOv8m',
            'file': 'yolov8m.pt',
            'size': '49.7 MB',
            'params': '25.9M',
            'speed': '5-7 ms',
            'mAP': '50.2',
            'use_case': 'High accuracy applications'
        },
        {
            'name': 'YOLOv8l',
            'file': 'yolov8l.pt',
            'size': '83.7 MB',
            'params': '43.7M',
            'speed': '8-10 ms',
            'mAP': '52.9',
            'use_case': 'Very high accuracy'
        },
        {
            'name': 'YOLOv8x',
            'file': 'yolov8x.pt',
            'size': '131.7 MB',
            'params': '68.2M',
            'speed': '12-15 ms',
            'mAP': '53.9',
            'use_case': 'Maximum accuracy'
        }
    ]
    
    for model in models_info:
        print(f"📦 {model['name']}")
        print(f"   File: {model['file']}")
        print(f"   Size: {model['size']}")
        print(f"   Parameters: {model['params']}")
        print(f"   Speed: {model['speed']} (GPU)")
        print(f"   mAP: {model['mAP']}")
        print(f"   Use Case: {model['use_case']}")
        print()
    
    print("Recommendation: Use YOLOv8n for real-time applications (default)")
    print("=" * 70)


def main():
    """
    Main function for model downloader.
    """
    print("\n")
    print("VisionTrack AI - Model Downloader")
    print()
    print("Options:")
    print("1. Download YOLOv8n (recommended, 6.2 MB)")
    print("2. Download YOLOv8s (balanced, 21.5 MB)")
    print("3. Download YOLOv8m (high accuracy, 49.7 MB)")
    print("4. Download all models (250 MB)")
    print("5. Show model information")
    print("6. Exit")
    print()
    
    choice = input("Enter your choice (1-6): ")
    
    if choice == '1':
        download_yolov8_model('yolov8n.pt')
    elif choice == '2':
        download_yolov8_model('yolov8s.pt')
    elif choice == '3':
        download_yolov8_model('yolov8m.pt')
    elif choice == '4':
        download_all_models()
    elif choice == '5':
        show_model_info()
    elif choice == '6':
        print("Goodbye!")
    else:
        print("Invalid choice. Please run again.")


if __name__ == '__main__':
    main()
