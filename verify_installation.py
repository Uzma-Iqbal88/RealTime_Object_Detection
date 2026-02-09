"""
VisionTrack AI - Installation Verification Script
==================================================

This script verifies that all dependencies are correctly installed
and the system is ready to run VisionTrack AI.

Author: [Your Name]
Project: CodeAlpha AI Internship
"""

import sys
import subprocess
from pathlib import Path


def print_header():
    """Print verification header."""
    print("\n" + "=" * 70)
    print("VisionTrack AI - Installation Verification")
    print("=" * 70)
    print()


def check_python_version():
    """Check Python version."""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version_str} (OK)")
        return True
    else:
        print(f"   ❌ Python {version_str} (Need 3.8+)")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"   ❌ {package_name}: Not installed")
        return False


def check_opencv():
    """Check OpenCV installation."""
    print("\n🔍 Checking OpenCV...")
    try:
        import cv2
        print(f"   ✅ OpenCV: {cv2.__version__}")
        
        # Test webcam access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(f"   ✅ Webcam: Accessible")
            cap.release()
        else:
            print(f"   ⚠️  Webcam: Not accessible (may need permissions)")
        
        return True
    except ImportError:
        print(f"   ❌ OpenCV: Not installed")
        return False


def check_pytorch():
    """Check PyTorch installation."""
    print("\n🔍 Checking PyTorch...")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"   ✅ CUDA: Available (GPU acceleration enabled)")
            print(f"      GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   ℹ️  CUDA: Not available (using CPU)")
        
        return True
    except ImportError:
        print(f"   ❌ PyTorch: Not installed")
        return False


def check_ultralytics():
    """Check Ultralytics (YOLOv8) installation."""
    print("\n🔍 Checking Ultralytics (YOLOv8)...")
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"   ✅ Ultralytics: {ultralytics.__version__}")
        
        # Check if model exists
        model_path = Path('models/yolov8n.pt')
        if model_path.exists():
            print(f"   ✅ YOLOv8 model: Downloaded")
        else:
            print(f"   ⚠️  YOLOv8 model: Not found (will auto-download on first run)")
        
        return True
    except ImportError:
        print(f"   ❌ Ultralytics: Not installed")
        return False


def check_deep_sort():
    """Check Deep SORT installation."""
    print("\n🔍 Checking Deep SORT...")
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        print(f"   ✅ Deep SORT: Installed")
        return True
    except ImportError:
        print(f"   ❌ Deep SORT: Not installed")
        return False


def check_other_packages():
    """Check other required packages."""
    print("\n🔍 Checking other packages...")
    
    packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('Pillow', 'PIL'),
        ('matplotlib', 'matplotlib'),
        ('PyYAML', 'yaml'),
        ('tqdm', 'tqdm'),
    ]
    
    all_ok = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_ok = False
    
    return all_ok


def check_project_files():
    """Check if all project files exist."""
    print("\n🔍 Checking project files...")
    
    required_files = [
        'visiontrack.py',
        'utils.py',
        'requirements.txt',
        'download_model.py',
        'README.md',
    ]
    
    all_ok = True
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}: Missing")
            all_ok = False
    
    return all_ok


def test_basic_functionality():
    """Test basic functionality."""
    print("\n🔍 Testing basic functionality...")
    
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        
        # Create test image
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        print(f"   ✅ Image creation: OK")
        
        # Test YOLO (will download model if needed)
        print(f"   🔄 Loading YOLOv8 model...")
        model = YOLO('yolov8n.pt')
        print(f"   ✅ Model loading: OK")
        
        # Test inference
        print(f"   🔄 Running test inference...")
        results = model(test_image, verbose=False)
        print(f"   ✅ Inference: OK")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def print_summary(checks):
    """Print verification summary."""
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    total = len(checks)
    passed = sum(checks.values())
    
    print(f"\nTotal checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if all(checks.values()):
        print("\n🎉 All checks passed! VisionTrack AI is ready to use.")
        print("\nNext steps:")
        print("1. Run: python visiontrack.py --source 0")
        print("2. Press 'q' to quit")
        print("3. Check README.md for more information")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nTo install missing packages:")
        print("pip install -r requirements.txt")
    
    print("\n" + "=" * 70)


def main():
    """Main verification function."""
    print_header()
    
    checks = {}
    
    # Run all checks
    checks['Python'] = check_python_version()
    checks['OpenCV'] = check_opencv()
    checks['PyTorch'] = check_pytorch()
    checks['Ultralytics'] = check_ultralytics()
    checks['Deep SORT'] = check_deep_sort()
    checks['Other Packages'] = check_other_packages()
    checks['Project Files'] = check_project_files()
    
    # Test basic functionality if all packages are installed
    if all([checks['OpenCV'], checks['PyTorch'], checks['Ultralytics']]):
        checks['Functionality Test'] = test_basic_functionality()
    
    # Print summary
    print_summary(checks)


if __name__ == '__main__':
    main()
