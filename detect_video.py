#!/usr/bin/env python3
"""
Real-time video object detection using SSD
Supports webcam, video files, and IP cameras
"""

import cv2
import argparse
import time
import sys
from pathlib import Path
import logging

from detection_engine import DetectionEngine
from config import MODEL_CONFIG, VIDEO_CONFIG, DETECTION_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Real-time video object detection with SSD')
    
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, path to video file, or IP camera URL)')
    parser.add_argument('--model', type=str, default='quantized',
                       choices=['quantized', 'lite', 'standard'],
                       help='Model type to use')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path (optional)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS for processing')
    parser.add_argument('--width', type=int, default=640,
                       help='Frame width')
    parser.add_argument('--height', type=int, default=480,
                       help='Frame height')
    parser.add_argument('--show-fps', action='store_true',
                       help='Show FPS counter on video')
    parser.add_argument('--record', action='store_true',
                       help='Record output video')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without displaying video (headless mode)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize detection engine
    logger.info(f"Initializing {args.model} SSD model...")
    try:
        engine = DetectionEngine(model_type=args.model, device='cpu')
    except Exception as e:
        logger.error(f"Failed to initialize detection engine: {e}")
        sys.exit(1)
    
    # Determine video source
    source = args.source
    if source.isdigit():
        source = int(source)
        logger.info(f"Using webcam {source}")
    else:
        logger.info(f"Using video source: {source}")
    
    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {source}")
        sys.exit(1)
    
    # Set video properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Get video properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Video properties: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
    
    # Initialize video writer if recording
    writer = None
    if args.record or args.output:
        output_path = args.output or f"output_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, args.fps, (actual_width, actual_height))
        logger.info(f"Recording to: {output_path}")
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    logger.info("Starting video detection...")
    logger.info("Press 'q' to quit, 's' to save screenshot")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            # Perform detection
            detection_start = time.time()
            result_frame, predictions = engine.detect_image(frame)
            detection_time = time.time() - detection_time
            
            # Add performance overlay
            if args.show_fps:
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Draw FPS and detection info
                cv2.putText(result_frame, f"FPS: {current_fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Detection: {detection_time*1000:.1f}ms", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw detection count
                total_detections = sum(len(pred['boxes']) for pred in predictions)
                cv2.putText(result_frame, f"Objects: {total_detections}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame if recording
            if writer:
                writer.write(result_frame)
            
            # Display frame
            if not args.no_display:
                cv2.imshow('SSD Object Detection', result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested by user")
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, result_frame)
                logger.info(f"Screenshot saved: {screenshot_path}")
            elif key == ord('h'):
                # Show help
                print("\nControls:")
                print("  q - Quit")
                print("  s - Save screenshot")
                print("  h - Show this help")
                print("  r - Toggle recording")
                print("  f - Toggle FPS display")
            
            frame_count += 1
            
            # Limit FPS if needed
            if args.fps > 0:
                time.sleep(max(0, 1.0/args.fps - detection_time))
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # Print statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        logger.info(f"Processing completed:")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average FPS: {avg_fps:.1f}")
        
        if args.record or args.output:
            logger.info(f"Video saved to: {output_path}")


if __name__ == '__main__':
    main() 