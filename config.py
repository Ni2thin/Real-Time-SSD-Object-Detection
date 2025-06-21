import os
from pathlib import Path

# Model Configuration
MODEL_CONFIG = {
    'model_type': 'quantized',  # 'quantized', 'lite', or 'standard'
    'num_classes': 91,  # COCO dataset classes (including background)
    'pretrained': True,
    'confidence_threshold': 0.3,
    'nms_threshold': 0.45,
    'max_detections': 100
}

# Video Configuration
VIDEO_CONFIG = {
    'fps': 30,
    'frame_width': 640,
    'frame_height': 480,
    'webcam_id': 0,
    'ip_camera_url': None,
    'record_output': False,
    'output_path': 'output/'
}

# Web Interface Configuration
WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 5001,
    'debug': False,
    'upload_folder': 'uploads/',
    'max_file_size': 16 * 1024 * 1024  # 16MB
}

# Detection Configuration
DETECTION_CONFIG = {
    'min_box_size': 10,
    'max_box_size': 1000,
    'class_filter': None,  # List of class names to detect, None for all
    'draw_boxes': True,
    'draw_labels': True,
    'draw_scores': True,
    'box_thickness': 2,
    'font_scale': 0.6,
    'font_thickness': 2
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'use_gpu': False,  # Set to True if CUDA is available
    'num_threads': 4,
    'batch_size': 1,
    'enable_optimization': True,
    'memory_efficient': True
}

# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
UPLOAD_DIR = BASE_DIR / 'uploads'

# Create directories if they don't exist
for directory in [MODEL_DIR, DATA_DIR, OUTPUT_DIR, UPLOAD_DIR]:
    directory.mkdir(exist_ok=True)

# Model paths
MODEL_PATHS = {
    'quantized': MODEL_DIR / 'ssd_quantized.pth',
    'lite': MODEL_DIR / 'ssd_lite.pth',
    'standard': MODEL_DIR / 'ssd_standard.pth'
}

# Colors for visualization (BGR format for OpenCV)
COLORS = {
    'person': (0, 255, 0),      # Green
    'car': (255, 0, 0),         # Blue
    'bicycle': (0, 255, 255),   # Yellow
    'motorcycle': (255, 0, 255), # Magenta
    'bus': (128, 0, 128),       # Purple
    'truck': (0, 128, 255),     # Orange
    'default': (0, 255, 255)    # Yellow for unknown classes
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/app.log'
}

# API Configuration
API_CONFIG = {
    'enable_cors': True,
    'rate_limit': 100,  # requests per minute
    'timeout': 30,      # seconds
    'max_workers': 4
} 