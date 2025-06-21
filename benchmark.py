#!/usr/bin/env python3
"""
Performance benchmarking script for SSD object detection on MacOS
"""

import torch
import cv2
import numpy as np
import time
import argparse
import logging
import json
from pathlib import Path
from tqdm import tqdm
import psutil
import os

from detection_engine import DetectionEngine
from config import MODEL_CONFIG, PERFORMANCE_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_system_info():
    """Get system information"""
    info = {
        'platform': os.uname().sysname,
        'architecture': os.uname().machine,
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'torch_version': torch.__version__,
        'torchvision_version': torchvision.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    }
    return info


def create_test_images(num_images=100, width=640, height=480):
    """Create synthetic test images"""
    images = []
    for i in range(num_images):
        # Create random image
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        images.append(image)
    return images


def benchmark_inference(engine, images, warmup_runs=10):
    """Benchmark inference performance"""
    logger.info("Starting inference benchmark...")
    
    # Warmup
    logger.info(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        engine.detect_image(images[0])
    
    # Benchmark
    logger.info("Running benchmark...")
    inference_times = []
    memory_usage = []
    
    for image in tqdm(images, desc="Benchmarking"):
        # Record memory before
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Time inference
        start_time = time.time()
        result_image, predictions = engine.detect_image(image)
        inference_time = time.time() - start_time
        
        # Record memory after
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        inference_times.append(inference_time)
        memory_usage.append(memory_after - memory_before)
    
    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    
    avg_fps = 1.0 / avg_inference_time
    max_fps = 1.0 / min_inference_time
    
    avg_memory = np.mean(memory_usage)
    
    results = {
        'avg_inference_time_ms': avg_inference_time * 1000,
        'std_inference_time_ms': std_inference_time * 1000,
        'min_inference_time_ms': min_inference_time * 1000,
        'max_inference_time_ms': max_inference_time * 1000,
        'avg_fps': avg_fps,
        'max_fps': max_fps,
        'avg_memory_mb': avg_memory,
        'total_images': len(images)
    }
    
    return results


def benchmark_video(engine, video_path, max_frames=1000):
    """Benchmark video processing performance"""
    logger.info(f"Starting video benchmark: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return None
    
    frame_count = 0
    inference_times = []
    fps_counter = 0
    start_time = time.time()
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Time inference
        inference_start = time.time()
        result_frame, predictions = engine.detect_image(frame)
        inference_time = time.time() - inference_start
        
        inference_times.append(inference_time)
        frame_count += 1
        fps_counter += 1
        
        # Calculate current FPS
        if time.time() - start_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            start_time = time.time()
            logger.info(f"Frame {frame_count}: FPS = {current_fps}")
    
    cap.release()
    
    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    avg_fps = 1.0 / avg_inference_time
    
    results = {
        'total_frames': frame_count,
        'avg_inference_time_ms': avg_inference_time * 1000,
        'avg_fps': avg_fps,
        'video_path': video_path
    }
    
    return results


def benchmark_memory_usage(engine, image_size=(640, 480)):
    """Benchmark memory usage"""
    logger.info("Benchmarking memory usage...")
    
    # Create test image
    image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
    
    # Measure memory before
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Run multiple inferences
    for _ in range(10):
        result_image, predictions = engine.detect_image(image)
    
    # Measure memory after
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    memory_usage = memory_after - memory_before
    
    return {
        'memory_before_mb': memory_before,
        'memory_after_mb': memory_after,
        'memory_usage_mb': memory_usage
    }


def run_comprehensive_benchmark(model_type='quantized', num_images=100, video_path=None):
    """Run comprehensive benchmark"""
    logger.info(f"Starting comprehensive benchmark for {model_type} model...")
    
    # Get system info
    system_info = get_system_info()
    logger.info(f"System: {system_info}")
    
    # Initialize engine
    engine = DetectionEngine(model_type=model_type, device='cpu')
    
    # Create test images
    test_images = create_test_images(num_images)
    
    # Run benchmarks
    results = {
        'system_info': system_info,
        'model_type': model_type,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Inference benchmark
    inference_results = benchmark_inference(engine, test_images)
    results['inference'] = inference_results
    
    # Memory benchmark
    memory_results = benchmark_memory_usage(engine)
    results['memory'] = memory_results
    
    # Video benchmark (if video provided)
    if video_path and Path(video_path).exists():
        video_results = benchmark_video(engine, video_path)
        results['video'] = video_results
    
    return results


def save_results(results, output_path):
    """Save benchmark results"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


def print_results(results):
    """Print benchmark results in a formatted way"""
    print("\n" + "="*60)
    print("SSD OBJECT DETECTION BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\nSystem Information:")
    print(f"  Platform: {results['system_info']['platform']}")
    print(f"  Architecture: {results['system_info']['architecture']}")
    print(f"  CPU Cores: {results['system_info']['cpu_count']}")
    print(f"  Memory: {results['system_info']['memory_gb']:.1f} GB")
    print(f"  PyTorch: {results['system_info']['torch_version']}")
    print(f"  CUDA Available: {results['system_info']['cuda_available']}")
    print(f"  MPS Available: {results['system_info']['mps_available']}")
    
    print(f"\nModel: {results['model_type']}")
    print(f"Timestamp: {results['timestamp']}")
    
    print(f"\nInference Performance:")
    inference = results['inference']
    print(f"  Average Inference Time: {inference['avg_inference_time_ms']:.2f} ms")
    print(f"  Standard Deviation: {inference['std_inference_time_ms']:.2f} ms")
    print(f"  Min/Max Time: {inference['min_inference_time_ms']:.2f} / {inference['max_inference_time_ms']:.2f} ms")
    print(f"  Average FPS: {inference['avg_fps']:.1f}")
    print(f"  Max FPS: {inference['max_fps']:.1f}")
    print(f"  Total Images: {inference['total_images']}")
    
    print(f"\nMemory Usage:")
    memory = results['memory']
    print(f"  Memory Before: {memory['memory_before_mb']:.1f} MB")
    print(f"  Memory After: {memory['memory_after_mb']:.1f} MB")
    print(f"  Memory Usage: {memory['memory_usage_mb']:.1f} MB")
    
    if 'video' in results:
        print(f"\nVideo Performance:")
        video = results['video']
        print(f"  Total Frames: {video['total_frames']}")
        print(f"  Average FPS: {video['avg_fps']:.1f}")
        print(f"  Average Inference Time: {video['avg_inference_time_ms']:.2f} ms")
    
    print("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Benchmark SSD object detection performance')
    parser.add_argument('--model', type=str, default='quantized',
                       choices=['quantized', 'lite', 'standard'],
                       help='Model type to benchmark')
    parser.add_argument('--num-images', type=int, default=100,
                       help='Number of test images')
    parser.add_argument('--video', type=str, default=None,
                       help='Video file for video benchmark')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_comprehensive_benchmark(
        model_type=args.model,
        num_images=args.num_images,
        video_path=args.video
    )
    
    # Print results
    print_results(results)
    
    # Save results
    if not args.no_save:
        save_results(results, args.output)


if __name__ == '__main__':
    main() 