import cv2
import torch
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from models.ssd_model import create_ssd_model, COCO_CLASSES
from config import MODEL_CONFIG, DETECTION_CONFIG, PERFORMANCE_CONFIG, COLORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionEngine:
    """
    Real-time object detection engine optimized for MacOS
    """
    
    def __init__(self, model_type='quantized', device='cpu'):
        self.device = device
        self.model_type = model_type
        
        # Initialize model
        logger.info(f"Loading {model_type} SSD model...")
        self.model = create_ssd_model(model_type=model_type)
        self.model.to(device)
        self.model.eval()
        
        # Performance optimization for MacOS
        if PERFORMANCE_CONFIG['enable_optimization']:
            torch.set_num_threads(PERFORMANCE_CONFIG['num_threads'])
            if PERFORMANCE_CONFIG['memory_efficient']:
                torch.backends.cudnn.benchmark = False
        
        # Initialize video capture
        self.cap = None
        self.is_running = False
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0
        
        logger.info("Detection engine initialized successfully")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference using torchvision.transforms
        """
        # Store original dimensions for scaling predictions back
        self.original_shape = image.shape[:2]

        # Convert numpy array (BGR) to PIL Image (RGB)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Define the standard transformation pipeline for SSD
        transform = T.Compose([
            T.Resize((300, 300)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Apply transformation and add batch dimension
        image_tensor = transform(image_pil).unsqueeze(0)
        
        return image_tensor
    
    def postprocess_predictions(self, predictions: List[Dict],
                              original_shape: Tuple[int, int]) -> List[Dict]:
        """
        Post-process model predictions to scale boxes to original image size.
        The model already performs filtering and NMS.
        """
        processed_predictions = []
        original_height, original_width = original_shape
        
        # Scale factors to convert from 300x300 back to original size
        scale_x = original_width / 300.0
        scale_y = original_height / 300.0

        for pred in predictions:
            # The model output is already filtered, just scale the boxes
            boxes = pred['boxes'].cpu().numpy()
            
            if len(boxes) > 0:
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
            
            processed_predictions.append({
                'boxes': boxes,
                'scores': pred['scores'].cpu().numpy(),
                'labels': pred['labels'].cpu().numpy()
            })
            
        return processed_predictions
    
    def draw_detections(self, image: np.ndarray, predictions: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes and labels on image
        """
        if not DETECTION_CONFIG['draw_boxes']:
            return image
        
        for pred in predictions:
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.astype(int)
                
                # Get class name and color
                class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f'class_{label}'
                color = COLORS.get(class_name, COLORS['default'])
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, DETECTION_CONFIG['box_thickness'])
                
                # Draw label and score
                if DETECTION_CONFIG['draw_labels'] or DETECTION_CONFIG['draw_scores']:
                    label_text = ""
                    if DETECTION_CONFIG['draw_labels']:
                        label_text += class_name
                    if DETECTION_CONFIG['draw_scores']:
                        label_text += f" {score:.2f}"
                    
                    # Calculate text position
                    (text_width, text_height), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 
                        DETECTION_CONFIG['font_scale'], DETECTION_CONFIG['font_thickness']
                    )
                    
                    # Draw text background
                    cv2.rectangle(image, (x1, y1 - text_height - 10), 
                                (x1 + text_width, y1), color, -1)
                    
                    # Draw text
                    cv2.putText(image, label_text, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, DETECTION_CONFIG['font_scale'],
                              (255, 255, 255), DETECTION_CONFIG['font_thickness'])
        
        return image
    
    def detect_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Perform object detection on a single image
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Perform inference (the model now handles filtering and NMS)
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Post-process predictions to scale boxes
        scaled_predictions = self.postprocess_predictions(predictions, self.original_shape)
        
        # Draw detections
        result_image = self.draw_detections(image.copy(), scaled_predictions)
        
        return result_image, scaled_predictions
    
    def start_video_capture(self, source=0):
        """
        Start video capture from webcam or video file
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        self.is_running = True
        logger.info(f"Video capture started from source: {source}")
    
    def stop_video_capture(self):
        """
        Stop video capture
        """
        self.is_running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        logger.info("Video capture stopped")
    
    def process_video_frame(self) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        Process a single video frame
        """
        if not hasattr(self, 'cap') or not self.cap or not self.is_running:
            return None, []
        
        ret, frame = self.cap.read()
        if not ret:
            return None, []
        
        # Perform detection
        result_frame, predictions = self.detect_image(frame)
        
        # Update FPS counter
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.avg_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        # Add FPS text to frame
        cv2.putText(result_frame, f"FPS: {self.avg_fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result_frame, predictions
    
    def get_performance_stats(self) -> Dict:
        """
        Get current performance statistics
        """
        return {
            'fps': self.avg_fps,
            'model_type': self.model_type,
            'device': self.device,
            'is_running': self.is_running
        }
    
    def __del__(self):
        """
        Cleanup resources
        """
        try:
            self.stop_video_capture()
        except:
            pass


class BatchDetectionEngine(DetectionEngine):
    """
    Batch processing detection engine for multiple images
    """
    
    def __init__(self, model_type='quantized', device='cpu', batch_size=4):
        super().__init__(model_type, device)
        self.batch_size = batch_size
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, List[Dict]]]:
        """
        Perform batch detection on multiple images
        """
        results = []
        
        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                tensor = self.preprocess_image(img)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.cat(batch_tensors, dim=0)
            batch_tensor = batch_tensor.to(self.device)
            
            # Perform inference
            with torch.no_grad():
                predictions = self.model(batch_tensor)
            
            # Post-process and draw detections
            for j, (img, pred) in enumerate(zip(batch_images, predictions)):
                pred = self.postprocess_predictions([pred], img.shape[:2])[0]
                result_img = self.draw_detections(img.copy(), [pred])
                results.append((result_img, pred))
        
        return results 