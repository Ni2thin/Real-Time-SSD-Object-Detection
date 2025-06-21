#!/usr/bin/env python3
"""
Setup script for SSD Object Detection System
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    # Upgrade pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        'models',
        'data',
        'output',
        'uploads',
        'logs',
        'trained_models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def download_models():
    """Download pre-trained models"""
    print("\nDownloading pre-trained models...")
    
    if not run_command(f"{sys.executable} download_model.py", "Downloading models"):
        print("Warning: Model download failed. You can run 'python download_model.py' manually later.")
        return False
    
    return True

def test_installation():
    """Test the installation"""
    print("\nTesting installation...")
    
    # Test imports
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        print("✓ All core dependencies imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test model creation
    try:
        from models.ssd_model import create_ssd_model
        model = create_ssd_model(model_type='quantized')
        print("✓ Model creation test passed")
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False
    
    return True

def create_launcher_scripts():
    """Create launcher scripts"""
    print("\nCreating launcher scripts...")
    
    # Create web app launcher
    web_launcher = """#!/bin/bash
cd "$(dirname "$0")"
python app.py
"""
    
    with open('start_web_app.sh', 'w') as f:
        f.write(web_launcher)
    
    os.chmod('start_web_app.sh', 0o755)
    print("✓ Created web app launcher: start_web_app.sh")
    
    # Create demo launcher
    demo_launcher = """#!/bin/bash
cd "$(dirname "$0")"
python demo.py
"""
    
    with open('run_demo.sh', 'w') as f:
        f.write(demo_launcher)
    
    os.chmod('run_demo.sh', 0o755)
    print("✓ Created demo launcher: run_demo.sh")
    
    return True

def main():
    """Main setup function"""
    print("="*60)
    print("SSD Object Detection System Setup")
    print("Optimized for MacOS")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("\n✗ Setup failed during directory creation")
        sys.exit(1)
    
    # Download models
    download_models()
    
    # Test installation
    if not test_installation():
        print("\n✗ Setup failed during testing")
        sys.exit(1)
    
    # Create launcher scripts
    if not create_launcher_scripts():
        print("\n✗ Setup failed during launcher creation")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ Setup completed successfully!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Start the web application:")
    print("   ./start_web_app.sh")
    print("   or")
    print("   python app.py")
    print()
    print("2. Run the demo:")
    print("   ./run_demo.sh")
    print("   or")
    print("   python demo.py")
    print()
    print("3. Test video detection:")
    print("   python detect_video.py --source 0")
    print()
    print("4. Test image detection:")
    print("   python detect_image.py --image your_image.jpg")
    print()
    print("5. Run performance benchmark:")
    print("   python benchmark.py")
    print()
    print("For more information, see README.md")

if __name__ == '__main__':
    main() 