import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDHead
import numpy as np

from config import MODEL_CONFIG

class QuantizedSSD(nn.Module):
    """
    Quantized Single Shot Multibox Detector optimized for MacOS
    """
    def __init__(self, num_classes=91, pretrained=True, quantized=True):
        super(QuantizedSSD, self).__init__()
        
        # Load pre-trained SSD model with built-in post-processing
        self.model = ssd300_vgg16(
            pretrained=pretrained, 
            num_classes=num_classes,
            score_thresh=MODEL_CONFIG['confidence_threshold'],
            nms_thresh=MODEL_CONFIG['nms_threshold']
        )
        
        # Quantization for MacOS optimization
        if quantized:
            self.model = torch.quantization.quantize_dynamic(
                self.model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8
            )
        
        self.num_classes = num_classes
        self.quantized = quantized
        
    def forward(self, images, targets=None):
        # The forward pass now handles prediction
        return self.model(images, targets)

class SSDLite(nn.Module):
    """
    Lightweight SSD implementation for faster inference
    """
    def __init__(self, num_classes=91, pretrained=True):
        super(SSDLite, self).__init__()
        
        # Use MobileNetV3 as backbone for better performance on MacOS
        self.backbone = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        
        # Remove the classifier
        self.backbone.classifier = nn.Identity()
        
        # SSD head
        self.ssd_head = SSDHead(
            in_channels=[16, 24, 40, 80, 160, 160],
            num_anchors=[4, 6, 6, 6, 4, 4],
            num_classes=num_classes
        )
        
        self.num_classes = num_classes
        
    def forward(self, x):
        features = self.backbone.features(x)
        return self.ssd_head(features)


def create_ssd_model(model_type='quantized', num_classes=91, pretrained=True):
    """
    Factory function to create SSD models
    """
    if model_type == 'quantized':
        return QuantizedSSD(num_classes=num_classes, pretrained=pretrained, quantized=True)
    elif model_type == 'lite':
        return SSDLite(num_classes=num_classes, pretrained=pretrained)
    else:
        # For the standard model, also use the built-in processing
        return ssd300_vgg16(
            pretrained=pretrained,
            num_classes=num_classes,
            score_thresh=MODEL_CONFIG['confidence_threshold'],
            nms_thresh=MODEL_CONFIG['nms_threshold']
        )


def load_pretrained_model(model_path=None, model_type='quantized'):
    """
    Load a pre-trained SSD model
    """
    if model_path and torch.cuda.is_available():
        model = torch.load(model_path, map_location='cpu')
    else:
        model = create_ssd_model(model_type=model_type)
    
    return model


# COCO class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
] 