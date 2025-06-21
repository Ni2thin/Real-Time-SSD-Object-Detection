#!/usr/bin/env python3
"""
Test script for SSD Object Detection System
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from models.ssd_model import create_ssd_model, COCO_CLASSES
        print("✓ SSD Model")
    except ImportError as e:
        print(f"✗ SSD Model import failed: {e}")
        return False
    
    try:
        from detection_engine import DetectionEngine
        print("✓ Detection Engine")
    except ImportError as e:
        print(f"✗ Detection Engine import failed: {e}")
        return False
    
    try:
        from config import MODEL_CONFIG, VIDEO_CONFIG
        print("✓ Configuration")
    except ImportError as e:
        print(f"✗ Configuration import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        from models.ssd_model import create_ssd_model
        
        # Test quantized model
        model = create_ssd_model(model_type='quantized')
        print("✓ Quantized model created")
        
        # Test lite model
        model = create_ssd_model(model_type='lite')
        print("✓ Lite model created")
        
        # Test standard model
        model = create_ssd_model(model_type='standard')
        print("✓ Standard model created")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_detection_engine():
    """Test detection engine"""
    print("\nTesting detection engine...")
    
    try:
        from detection_engine import DetectionEngine
        
        # Create engine
        engine = DetectionEngine(model_type='quantized', device='cpu')
        print("✓ Detection engine created")
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection
        start_time = time.time()
        result_image, predictions = engine.detect_image(test_image)
        detection_time = time.time() - start_time
        
        print(f"✓ Detection completed in {detection_time*1000:.2f} ms")
        print(f"✓ FPS: {1.0/detection_time:.1f}")
        
        total_detections = sum(len(pred['boxes']) for pred in predictions)
        print(f"✓ Detections: {total_detections}")
        
        return True
    except Exception as e:
        print(f"✗ Detection engine test failed: {e}")
        return False

def test_webcam_access():
    """Test webcam access"""
    print("\nTesting webcam access...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return False
        
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            print("✗ Could not read frame from webcam")
            cap.release()
            return False
        
        print(f"✓ Webcam working - frame size: {frame.shape}")
        cap.release()
        return True
    except Exception as e:
        print(f"✗ Webcam test failed: {e}")
        return False

def test_file_operations():
    """Test file operations"""
    print("\nTesting file operations...")
    
    try:
        # Test directory creation
        test_dir = Path("test_output")
        test_dir.mkdir(exist_ok=True)
        print("✓ Directory creation")
        
        # Test image saving
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_path = test_dir / "test_image.jpg"
        cv2.imwrite(str(test_path), test_image)
        print("✓ Image saving")
        
        # Test image loading
        loaded_image = cv2.imread(str(test_path))
        if loaded_image is not None:
            print("✓ Image loading")
        else:
            print("✗ Image loading failed")
            return False
        
        # Cleanup
        test_path.unlink()
        test_dir.rmdir()
        print("✓ File cleanup")
        
        return True
    except Exception as e:
        print(f"✗ File operations test failed: {e}")
        return False

def test_performance():
    """Test basic performance"""
    print("\nTesting performance...")
    
    try:
        from detection_engine import DetectionEngine
        
        engine = DetectionEngine(model_type='quantized', device='cpu')
        
        # Create test images
        test_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
        
        # Warmup
        for _ in range(3):
            engine.detect_image(test_images[0])
        
        # Performance test
        inference_times = []
        for i, image in enumerate(test_images):
            start_time = time.time()
            result_image, predictions = engine.detect_image(image)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            print(f"  Image {i+1}: {inference_time*1000:.2f} ms")
        
        avg_time = np.mean(inference_times)
        avg_fps = 1.0 / avg_time
        
        print(f"✓ Average inference time: {avg_time*1000:.2f} ms")
        print(f"✓ Average FPS: {avg_fps:.1f}")
        
        if avg_fps > 5:  # Should be able to achieve at least 5 FPS
            print("✓ Performance acceptable")
            return True
        else:
            print("✗ Performance below expected threshold")
            return False
            
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config import MODEL_CONFIG, VIDEO_CONFIG, DETECTION_CONFIG
        
        # Check required config keys
        required_keys = {
            'MODEL_CONFIG': ['model_type', 'num_classes', 'confidence_threshold'],
            'VIDEO_CONFIG': ['fps', 'frame_width', 'frame_height'],
            'DETECTION_CONFIG': ['draw_boxes', 'draw_labels', 'draw_scores']
        }
        
        config_vars = {
            'MODEL_CONFIG': MODEL_CONFIG,
            'VIDEO_CONFIG': VIDEO_CONFIG,
            'DETECTION_CONFIG': DETECTION_CONFIG
        }
        for config_name, keys in required_keys.items():
            config = config_vars[config_name]
            for key in keys:
                if key not in config:
                    print(f"✗ Missing config key: {config_name}.{key}")
                    return False
        
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("SSD Object Detection System - System Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Model Creation", test_model_creation),
        ("Detection Engine", test_detection_engine),
        ("File Operations", test_file_operations),
        ("Performance", test_performance),
        ("Webcam Access", test_webcam_access),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} test failed")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nYou can now:")
        print("1. Run the demo: python demo.py")
        print("2. Start the web app: python app.py")
        print("3. Test video detection: python detect_video.py --source 0")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("You may need to:")
        print("1. Install missing dependencies")
        print("2. Check your Python environment")
        print("3. Ensure all files are in the correct locations")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 