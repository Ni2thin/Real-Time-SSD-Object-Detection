#!/usr/bin/env python3
"""
Static image object detection using SSD
"""

import cv2
import argparse
import sys
import json
from pathlib import Path
import logging

from detection_engine import DetectionEngine
from config import MODEL_CONFIG, DETECTION_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Static image object detection with SSD')
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image path (optional)')
    parser.add_argument('--model', type=str, default='quantized',
                       choices=['quantized', 'lite', 'standard'],
                       help='Model type to use')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--json', type=str, default=None,
                       help='Save detection results to JSON file')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without displaying image')
    parser.add_argument('--batch', type=str, default=None,
                       help='Process all images in directory')
    
    return parser.parse_args()


def process_single_image(image_path, engine, args):
    """Process a single image"""
    logger.info(f"Processing image: {image_path}")
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return None
    
    # Perform detection
    result_image, predictions = engine.detect_image(image)
    
    # Save output image
    if args.output:
        output_path = Path(args.output)
        if args.batch:
            # For batch processing, create output in same directory structure
            output_path = Path(args.output) / image_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), result_image)
        logger.info(f"Result saved to: {output_path}")
    
    # Save JSON results
    if args.json:
        json_results = {
            'image_path': str(image_path),
            'image_size': [image.shape[1], image.shape[0]],
            'detections': []
        }
        
        for pred in predictions:
            for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                class_name = engine.model.COCO_CLASSES[label] if label < len(engine.model.COCO_CLASSES) else f'class_{label}'
                json_results['detections'].append({
                    'class': class_name,
                    'confidence': float(score),
                    'bbox': box.tolist()
                })
        
        json_path = Path(args.json)
        if args.batch:
            json_path = Path(args.json) / f"{image_path.stem}_detections.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"JSON results saved to: {json_path}")
    
    # Display image
    if not args.no_display:
        cv2.imshow('SSD Object Detection', result_image)
        cv2.waitKey(0)
    
    return predictions


def process_batch_images(input_dir, engine, args):
    """Process all images in a directory"""
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"No image files found in: {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Create output directories
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.json:
        Path(args.json).mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
        
        try:
            predictions = process_single_image(image_file, engine, args)
            if predictions:
                total_detections = sum(len(pred['boxes']) for pred in predictions)
                logger.info(f"  Detected {total_detections} objects")
        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
    
    logger.info("Batch processing completed")


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
    
    # Process images
    if args.batch:
        process_batch_images(args.batch, engine, args)
    else:
        # Process single image
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"Image file does not exist: {args.image}")
            sys.exit(1)
        
        predictions = process_single_image(image_path, engine, args)
        
        if predictions:
            total_detections = sum(len(pred['boxes']) for pred in predictions)
            logger.info(f"Detection completed: {total_detections} objects found")
            
            # Print detection summary
            if not args.no_display:
                print("\nDetection Summary:")
                for pred in predictions:
                    for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                        class_name = engine.model.COCO_CLASSES[label] if label < len(engine.model.COCO_CLASSES) else f'class_{label}'
                        print(f"  {class_name}: {score:.3f} at {box}")
    
    # Cleanup
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 