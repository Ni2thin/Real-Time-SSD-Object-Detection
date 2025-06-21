from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import time
import threading
import logging
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

from detection_engine import DetectionEngine
from config import WEB_CONFIG, MODEL_CONFIG, VIDEO_CONFIG, API_CONFIG
from models.ssd_model import COCO_CLASSES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) if API_CONFIG['enable_cors'] else None

# Global variables
detection_engine = None
camera_thread = None
current_frame = None
frame_lock = threading.Lock()

def initialize_detection_engine():
    """Initialize the detection engine"""
    global detection_engine
    try:
        detection_engine = DetectionEngine(
            model_type=MODEL_CONFIG['model_type'],
            device='cpu'  # Use CPU for MacOS optimization
        )
        logger.info("Detection engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize detection engine: {e}")
        return False

def camera_worker():
    """Worker thread for camera processing"""
    global current_frame, detection_engine
    
    if not detection_engine:
        logger.error("Detection engine not initialized")
        return
    
    try:
        detection_engine.start_video_capture(VIDEO_CONFIG['webcam_id'])
        
        while detection_engine.is_running:
            frame, predictions = detection_engine.process_video_frame()
            if frame is not None:
                with frame_lock:
                    current_frame = frame.copy()
            time.sleep(1.0 / VIDEO_CONFIG['fps'])
            
    except Exception as e:
        logger.error(f"Camera worker error: {e}")
    finally:
        detection_engine.stop_video_capture()

def generate_frames():
    """Generate video frames for streaming"""
    global current_frame
    
    while True:
        with frame_lock:
            if current_frame is not None:
                # Encode frame for streaming
                ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_data = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        
        time.sleep(1.0 / VIDEO_CONFIG['fps'])

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera detection"""
    global camera_thread, detection_engine
    
    try:
        if not detection_engine:
            if not initialize_detection_engine():
                return jsonify({'success': False, 'error': 'Failed to initialize detection engine'})
        
        if camera_thread is None or not camera_thread.is_alive():
            camera_thread = threading.Thread(target=camera_worker, daemon=True)
            camera_thread.start()
            logger.info("Camera detection started")
        
        return jsonify({'success': True, 'message': 'Camera started successfully'})
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera detection"""
    global detection_engine
    
    try:
        if detection_engine:
            detection_engine.stop_video_capture()
            logger.info("Camera detection stopped")
        
        return jsonify({'success': True, 'message': 'Camera stopped successfully'})
    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/detect', methods=['POST'])
def detect_image():
    """Detect objects in uploaded image"""
    global detection_engine
    
    try:
        if not detection_engine:
            if not initialize_detection_engine():
                return jsonify({'success': False, 'error': 'Failed to initialize detection engine'})
        
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})
        
        # Read and process image
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image format'})
        
        # Perform detection
        result_image, predictions = detection_engine.detect_image(image)
        
        # Convert result to base64 for web display
        _, buffer = cv2.imencode('.jpg', result_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Format predictions for JSON response
        formatted_predictions = []
        for pred in predictions:
            for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f'class_{label}'
                formatted_predictions.append({
                    'class': class_name,
                    'confidence': float(score),
                    'bbox': box.tolist()
                })
        
        return jsonify({
            'success': True,
            'image': result_base64,
            'predictions': formatted_predictions,
            'total_detections': len(formatted_predictions)
        })
        
    except Exception as e:
        logger.error(f"Error in image detection: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stats')
def get_stats():
    """Get system statistics"""
    global detection_engine
    
    stats = {
        'engine_initialized': detection_engine is not None,
        'camera_running': detection_engine.is_running if detection_engine else False,
        'model_type': MODEL_CONFIG['model_type'],
        'device': 'cpu'
    }
    
    if detection_engine:
        stats.update(detection_engine.get_performance_stats())
    
    return jsonify(stats)

@app.route('/config', methods=['GET', 'POST'])
def config():
    """Get or update configuration"""
    if request.method == 'GET':
        return jsonify({
            'model_config': MODEL_CONFIG,
            'video_config': VIDEO_CONFIG,
            'detection_config': DETECTION_CONFIG
        })
    else:
        # Update configuration (implement as needed)
        return jsonify({'success': True, 'message': 'Configuration updated'})

if __name__ == '__main__':
    # Initialize detection engine on startup
    if initialize_detection_engine():
        logger.info("Application started successfully")
    else:
        logger.error("Failed to initialize detection engine")
    
    app.run(
        host=WEB_CONFIG['host'],
        port=WEB_CONFIG['port'],
        debug=WEB_CONFIG['debug'],
        threaded=True
    ) 