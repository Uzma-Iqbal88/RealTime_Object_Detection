# VisionTrack AI - Quick Start Guide

## 🚀 Get Started in 5 Minutes!

This guide will help you get VisionTrack AI running quickly.

---

## Step 1: Install Python (if not already installed)

### Windows:
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ **CHECK "Add Python to PATH"**
4. Click "Install Now"

### Verify Installation:
```bash
python --version
```
Should show: `Python 3.8.x` or higher

---

## Step 2: Install Dependencies

Open terminal/command prompt in project directory:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

⏱️ **This will take 5-10 minutes**

---

## Step 3: Run VisionTrack AI

### With Webcam:
```bash
python visiontrack.py --source 0
```

### With Video File:
```bash
python visiontrack.py --source path/to/video.mp4
```

### Save Output:
```bash
python visiontrack.py --source 0 --save-output
```

---

## Keyboard Controls

- **'q'**: Quit
- **'s'**: Save screenshot
- **'p'**: Pause/Resume

---

## Troubleshooting

### "No module named cv2"
```bash
pip install opencv-python
```

### "Could not open video source"
- Check webcam permissions
- Try different camera index: `--source 1` or `--source 2`

### Slow performance
- Use GPU if available
- Lower confidence threshold: `--conf 0.3`
- Close other applications

---

## Next Steps

1. ✅ Test with webcam
2. ✅ Test with video file
3. ✅ Read `README.md` for detailed information
4. ✅ Check `DOCUMENTATION.md` for advanced features
5. ✅ Customize for your use case

---

## Command Reference

```bash
# Basic usage
python visiontrack.py --source 0

# With options
python visiontrack.py --source 0 --conf 0.6 --save-output

# Help
python visiontrack.py --help
```

### Options:
- `--source`: Video source (0 for webcam, or file path)
- `--conf`: Confidence threshold (0.0-1.0, default: 0.5)
- `--save-output`: Save processed video
- `--no-display`: Run without GUI

---

## Example Commands

```bash
# Webcam with default settings
python visiontrack.py --source 0

# Webcam with higher confidence
python visiontrack.py --source 0 --conf 0.7

# Video file
python visiontrack.py --source demo.mp4

# Save output video
python visiontrack.py --source 0 --save-output

# Process video without display (headless)
python visiontrack.py --source video.mp4 --save-output --no-display
```

---

## Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║              🎯 VisionTrack AI v1.0                         ║
║        Real-Time Object Detection & Tracking System          ║
║        Powered by YOLOv8 + Deep SORT                        ║
╚══════════════════════════════════════════════════════════════╝

🔄 Loading YOLOv8 model...
✅ YOLOv8 model loaded successfully!
🔄 Initializing Deep SORT tracker...
✅ Deep SORT tracker initialized!
📹 Opening webcam (index: 0)...
✅ Video source opened successfully!
   Resolution: 1280x720
   FPS: 30

🚀 Starting VisionTrack AI...
Press 'q' to quit, 's' to save screenshot
```

---

## Getting Help

- **Installation Issues**: Check `INSTALL.md`
- **Technical Details**: Check `DOCUMENTATION.md`
- **Project Overview**: Check `README.md`
- **Full Explanation**: Check `PROJECT_EXPLANATION.md`

---

## Support

If you encounter issues:
1. Run verification script: `python verify_installation.py`
2. Check error messages carefully
3. Consult documentation files
4. Contact: [your.email@example.com]

---

**Happy Tracking! 🎯**
