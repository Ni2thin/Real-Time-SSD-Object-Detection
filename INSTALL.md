# Installation Guide - SSD Object Detection System

This guide will help you install and set up the SSD Object Detection System on MacOS.

## Prerequisites

### System Requirements
- **Operating System**: macOS 10.15 (Catalina) or later
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 2GB free space for models and dependencies
- **Camera**: Built-in webcam or external camera for real-time detection

### Python Environment
We recommend using a virtual environment to avoid conflicts with other Python packages.

## Installation Methods

### Method 1: Automatic Setup (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ssd-object-detection
   ```

2. **Run the automatic setup script**:
   ```bash
   python setup.py
   ```

   This script will:
   - Check Python version compatibility
   - Install all required dependencies
   - Create necessary directories
   - Download pre-trained models
   - Test the installation
   - Create launcher scripts

3. **Verify installation**:
   ```bash
   python test_system.py
   ```

### Method 2: Manual Installation

If you prefer to install manually or the automatic setup fails:

1. **Create a virtual environment**:
   ```bash
   python3 -m venv ssd_env
   source ssd_env/bin/activate
   ```

2. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create directories**:
   ```bash
   mkdir -p models data output uploads logs trained_models
   ```

5. **Download models**:
   ```bash
   python download_model.py
   ```

6. **Test the installation**:
   ```bash
   python test_system.py
   ```

## Troubleshooting

### Common Issues

#### 1. PyTorch Installation Issues
If you encounter issues installing PyTorch:

```bash
# For CPU-only installation (recommended for MacOS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For MPS (Metal Performance Shaders) support on Apple Silicon
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 2. OpenCV Installation Issues
If OpenCV fails to install:

```bash
# Try installing with conda
conda install opencv

# Or install specific version
pip install opencv-python==4.8.0.76
```

#### 3. Permission Issues
If you encounter permission errors:

```bash
# Make scripts executable
chmod +x *.py
chmod +x *.sh

# Or run with sudo (not recommended)
sudo python setup.py
```

#### 4. Model Download Issues
If model download fails:

```bash
# Check internet connection
ping google.com

# Try downloading manually
python download_model.py --verbose

# Or download from alternative source
# (Check README.md for alternative download links)
```

#### 5. Webcam Access Issues
If webcam doesn't work:

1. **Check camera permissions**:
   - Go to System Preferences > Security & Privacy > Privacy > Camera
   - Ensure your terminal/IDE has camera access

2. **Test webcam manually**:
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam available:', cap.isOpened()); cap.release()"
   ```

3. **Try different camera index**:
   ```bash
   python detect_video.py --source 1  # Try camera index 1
   ```

### Performance Optimization

#### For Better Performance on MacOS:

1. **Use quantized models** (default):
   ```python
   # In config.py
   MODEL_CONFIG['model_type'] = 'quantized'
   ```

2. **Optimize thread usage**:
   ```python
   # In config.py
   PERFORMANCE_CONFIG['num_threads'] = 4  # Adjust based on your CPU
   ```

3. **Enable MPS acceleration** (Apple Silicon Macs):
   ```python
   # In config.py
   PERFORMANCE_CONFIG['use_mps'] = True
   ```

4. **Reduce frame resolution**:
   ```python
   # In config.py
   VIDEO_CONFIG['frame_width'] = 480
   VIDEO_CONFIG['frame_height'] = 360
   ```

## Verification

After installation, verify everything works:

1. **Run system test**:
   ```bash
   python test_system.py
   ```

2. **Test image detection**:
   ```bash
   python detect_image.py --image demo_image.jpg
   ```

3. **Test webcam detection**:
   ```bash
   python detect_video.py --source 0
   ```

4. **Start web application**:
   ```bash
   python app.py
   ```
   Then open http://localhost:5000 in your browser

5. **Run performance benchmark**:
   ```bash
   python benchmark.py
   ```

## Expected Performance

On a typical MacOS system, you should expect:

- **Inference Speed**: 20-50 FPS (depending on hardware)
- **Memory Usage**: ~200-500MB
- **CPU Usage**: 30-70% (depending on model type)
- **Accuracy**: mAP@0.5: 0.74 on COCO dataset

## Uninstallation

To remove the system:

1. **Deactivate virtual environment** (if used):
   ```bash
   deactivate
   ```

2. **Remove the directory**:
   ```bash
   rm -rf ssd-object-detection
   ```

3. **Remove virtual environment** (if created):
   ```bash
   rm -rf ssd_env
   ```

## Support

If you encounter issues:

1. **Check the troubleshooting section above**
2. **Run the test script**: `python test_system.py`
3. **Check system requirements**
4. **Review error logs in the `logs/` directory**
5. **Create an issue on the project repository**

## Next Steps

After successful installation:

1. **Read the README.md** for usage instructions
2. **Try the demo**: `python demo.py`
3. **Explore the web interface**: `python app.py`
4. **Customize configuration**: Edit `config.py`
5. **Train on custom data**: Use `train_model.py`

## System Information

To get detailed system information for troubleshooting:

```bash
python -c "
import platform
import torch
import cv2
print(f'Platform: {platform.platform()}')
print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available()}')
"
``` 