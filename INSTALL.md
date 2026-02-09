# VisionTrack AI - Installation Guide

## Quick Start (5 Minutes)

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Webcam or video file for testing
- 4GB+ RAM
- Internet connection (for downloading models)

---

## Step-by-Step Installation

### Windows

#### 1. Install Python
Download and install Python from [python.org](https://www.python.org/downloads/)

**Important**: Check "Add Python to PATH" during installation!

#### 2. Verify Installation
```bash
python --version
# Should show: Python 3.8.x or higher
```

#### 3. Create Project Directory
```bash
# Open Command Prompt or PowerShell
cd Desktop
mkdir VisionTrack-AI
cd VisionTrack-AI
```

#### 4. Create Virtual Environment
```bash
python -m venv venv
```

#### 5. Activate Virtual Environment
```bash
# Command Prompt
venv\Scripts\activate

# PowerShell
venv\Scripts\Activate.ps1
```

**Note**: If PowerShell gives an error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 6. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- OpenCV (computer vision)
- Ultralytics (YOLOv8)
- PyTorch (deep learning)
- Deep SORT (tracking)
- Other utilities

**Installation time**: 5-10 minutes (depending on internet speed)

#### 7. Download YOLOv8 Model
```bash
python download_model.py
```

Or the model will auto-download on first run.

#### 8. Test Installation
```bash
python visiontrack.py --source 0
```

Press 'q' to quit.

---

### Linux / macOS

#### 1. Install Python
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# macOS (using Homebrew)
brew install python3
```

#### 2. Create Virtual Environment
```bash
cd ~/Desktop
mkdir VisionTrack-AI
cd VisionTrack-AI
python3 -m venv venv
```

#### 3. Activate Virtual Environment
```bash
source venv/bin/activate
```

#### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Test Installation
```bash
python visiontrack.py --source 0
```

---

## GPU Acceleration (Optional)

For **NVIDIA GPU** users (10x faster performance):

### Check CUDA Availability
```bash
nvidia-smi
```

If this works, you have CUDA installed.

### Install PyTorch with CUDA
```bash
# Uninstall CPU version
pip uninstall torch torchvision

# Install GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU
```python
import torch
print(torch.cuda.is_available())  # Should print: True
```

---

## Troubleshooting

### Issue 1: "pip is not recognized"
**Solution**: Add Python to PATH
```bash
# Windows: Add to PATH manually
# Control Panel > System > Advanced > Environment Variables
# Add: C:\Users\YourName\AppData\Local\Programs\Python\Python3x\Scripts
```

### Issue 2: "No module named cv2"
**Solution**: Reinstall OpenCV
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.8.1.78
```

### Issue 3: "Could not open video source"
**Solution**: Check webcam permissions
```bash
# Windows: Settings > Privacy > Camera > Allow apps to access camera
# Linux: Check /dev/video0 permissions
ls -l /dev/video*
```

### Issue 4: "Out of memory"
**Solution**: Use smaller model or reduce resolution
```python
# In visiontrack.py, change:
self.detector = YOLO('yolov8n.pt')  # Smallest model
```

### Issue 5: Slow performance
**Solutions**:
1. Use GPU acceleration (see above)
2. Reduce confidence threshold
3. Process every 2nd frame
4. Close other applications

---

## Verify Installation

Run the verification script:

```bash
python verify_installation.py
```

Expected output:
```
✅ Python version: 3.x.x
✅ OpenCV installed: 4.8.1
✅ PyTorch installed: 2.1.1
✅ Ultralytics installed: 8.0.220
✅ Deep SORT installed: 1.3.2
✅ Webcam accessible: Yes
✅ YOLOv8 model downloaded: Yes

🎉 All checks passed! VisionTrack AI is ready to use.
```

---

## Running VisionTrack AI

### Basic Usage

```bash
# Run with webcam
python visiontrack.py --source 0

# Run with video file
python visiontrack.py --source path/to/video.mp4

# Save output
python visiontrack.py --source 0 --save-output

# Adjust confidence threshold
python visiontrack.py --source 0 --conf 0.6
```

### Keyboard Controls

- **'q'**: Quit application
- **'s'**: Save screenshot
- **'p'**: Pause/Resume

---

## Project Structure

After installation, your directory should look like:

```
VisionTrack-AI/
├── venv/                    # Virtual environment (don't commit)
├── visiontrack.py           # Main application
├── utils.py                 # Utility functions
├── requirements.txt         # Dependencies
├── download_model.py        # Model downloader
├── verify_installation.py   # Installation checker
├── README.md                # Project documentation
├── DOCUMENTATION.md         # Technical documentation
├── INSTALL.md               # This file
├── models/                  # Downloaded models
│   └── yolov8n.pt
├── output/                  # Saved videos (created on first run)
├── screenshots/             # Saved screenshots (created on first run)
└── logs/                    # Detection logs (created on first run)
```

---

## Next Steps

1. **Test with webcam**: `python visiontrack.py --source 0`
2. **Test with video**: Download sample video and test
3. **Read documentation**: Check `DOCUMENTATION.md` for advanced features
4. **Customize**: Modify code for your specific use case

---

## Getting Help

### Common Commands

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Update all packages
pip install --upgrade -r requirements.txt

# Deactivate virtual environment
deactivate

# Reactivate virtual environment
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

### Resources

- **OpenCV Documentation**: https://docs.opencv.org/
- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Deep SORT Paper**: https://arxiv.org/abs/1703.07402

### Support

If you encounter issues:
1. Check this installation guide
2. Read `DOCUMENTATION.md` troubleshooting section
3. Check GitHub issues
4. Contact: [your.email@example.com]

---

## Uninstallation

To remove VisionTrack AI:

```bash
# Deactivate virtual environment
deactivate

# Delete project directory
cd ..
rm -rf VisionTrack-AI  # Linux/Mac
# or
rmdir /s VisionTrack-AI  # Windows
```

---

**Installation Guide Version**: 1.0  
**Last Updated**: February 2026  
**Tested On**: Windows 10/11, Ubuntu 20.04/22.04, macOS 12+
