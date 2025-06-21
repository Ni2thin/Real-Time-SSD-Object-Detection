#!/usr/bin/env python3
"""
Demo script for SSD Object Detection System
"""

import cv2
import numpy as np
import time
from pathlib import Path

from detection_engine import DetectionEngine
from config import MODEL_CONFIG

def create_demo_image():
    """Create a demo image with simple shapes"""
    # Create a 640x480 image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate objects
    cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(image, (300, 150), (400, 250), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(image, (500, 200), (600, 300), (0, 0, 255), -1)  # Red rectangle
    
    # Add some text
    cv2.putText(image, "Demo Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image

def demo_image_detection():
    """Demo image detection"""
    print("="*50)
    print("SSD Object Detection Demo - Image Processing")
    print("="*50)
    
    # Initialize detection engine
    print("Initializing detection engine...")
    engine = DetectionEngine(model_type='quantized', device='cpu')
    
    # Create demo image
    print("Creating demo image...")
    demo_image = create_demo_image()
    
    # Save demo image
    cv2.imwrite('demo_image.jpg', demo_image)
    print("Demo image saved as 'demo_image.jpg'")
    
    # Perform detection
    print("Performing object detection...")
    start_time = time.time()
    result_image, predictions = engine.detect_image(demo_image)
    detection_time = time.time() - start_time
    
    # Save result
    cv2.imwrite('demo_result.jpg', result_image)
    print("Detection result saved as 'demo_result.jpg'")
    
    # Print results
    print(f"\nDetection completed in {detection_time*1000:.2f} ms")
    print(f"FPS: {1.0/detection_time:.1f}")
    
    total_detections = sum(len(pred['boxes']) for pred in predictions)
    print(f"Total objects detected: {total_detections}")
    
    # Print detection details
    for i, pred in enumerate(predictions):
        for j, (box, score, label) in enumerate(zip(pred['boxes'], pred['scores'], pred['labels'])):
            class_name = engine.model.COCO_CLASSES[label] if label < len(engine.model.COCO_CLASSES) else f'class_{label}'
            print(f"  Detection {j+1}: {class_name} (confidence: {score:.3f})")
    
    print("\nDemo completed successfully!")

def demo_webcam():
    """Demo webcam detection"""
    print("="*50)
    print("SSD Object Detection Demo - Webcam")
    print("="*50)
    
    # Initialize detection engine
    print("Initializing detection engine...")
    engine = DetectionEngine(model_type='quantized', device='cpu')
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Webcam initialized successfully")
    print("Press 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Perform detection
            result_frame, predictions = engine.detect_image(frame)
            
            # Add FPS counter
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.1f}")
            
            # Add text overlay
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            total_detections = sum(len(pred['boxes']) for pred in predictions)
            cv2.putText(result_frame, f"Objects: {total_detections}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('SSD Object Detection Demo', result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, result_frame)
                print(f"Screenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nDemo completed:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.1f}")

def demo_performance():
    """Demo performance testing"""
    print("="*50)
    print("SSD Object Detection Demo - Performance Test")
    print("="*50)
    
    # Initialize detection engine
    print("Initializing detection engine...")
    engine = DetectionEngine(model_type='quantized', device='cpu')
    
    # Create test images
    print("Creating test images...")
    test_images = []
    for i in range(10):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_images.append(image)
    
    # Warmup
    print("Running warmup...")
    for _ in range(5):
        engine.detect_image(test_images[0])
    
    # Performance test
    print("Running performance test...")
    inference_times = []
    
    for i, image in enumerate(test_images):
        start_time = time.time()
        result_image, predictions = engine.detect_image(image)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        print(f"  Image {i+1}: {inference_time*1000:.2f} ms")
    
    # Calculate statistics
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    print(f"\nPerformance Results:")
    print(f"  Average time: {avg_time*1000:.2f} ms")
    print(f"  Standard deviation: {std_time*1000:.2f} ms")
    print(f"  Min/Max time: {min_time*1000:.2f} / {max_time*1000:.2f} ms")
    print(f"  Average FPS: {1.0/avg_time:.1f}")
    print(f"  Max FPS: {1.0/min_time:.1f}")

def main():
    """Main demo function"""
    print("SSD Object Detection System Demo")
    print("Optimized for MacOS with Quantized Models")
    print()
    
    while True:
        print("Choose a demo:")
        print("1. Image Detection Demo")
        print("2. Webcam Detection Demo")
        print("3. Performance Test Demo")
        print("4. Run All Demos")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            demo_image_detection()
        elif choice == '2':
            demo_webcam()
        elif choice == '3':
            demo_performance()
        elif choice == '4':
            demo_image_detection()
            print("\n" + "="*50 + "\n")
            demo_performance()
            print("\n" + "="*50 + "\n")
            demo_webcam()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
        
        print("\n" + "="*50 + "\n")

if __name__ == '__main__':
    main() 