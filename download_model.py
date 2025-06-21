#!/usr/bin/env python3
"""
Download and prepare pre-trained SSD models for MacOS optimization
"""

import torch
import torchvision
import os
import logging
from pathlib import Path
import requests
from tqdm import tqdm

from models.ssd_model import create_ssd_model, QuantizedSSD
from config import MODEL_PATHS, MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_file(url, filepath, chunk_size=8192):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)


def create_quantized_model():
    """Create and save quantized SSD model"""
    logger.info("Creating quantized SSD model...")
    
    try:
        # Create quantized model
        model = QuantizedSSD(
            num_classes=MODEL_CONFIG['num_classes'],
            pretrained=True,
            quantized=True
        )
        
        # Save model
        torch.save(model.state_dict(), MODEL_PATHS['quantized'])
        logger.info(f"Quantized model saved to: {MODEL_PATHS['quantized']}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to create quantized model: {e}")
        return False


def create_lite_model():
    """Create and save lightweight SSD model"""
    logger.info("Creating lightweight SSD model...")
    
    try:
        # Create lite model
        model = create_ssd_model(model_type='lite')
        
        # Save model
        torch.save(model.state_dict(), MODEL_PATHS['lite'])
        logger.info(f"Lite model saved to: {MODEL_PATHS['lite']}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to create lite model: {e}")
        return False


def create_standard_model():
    """Create and save standard SSD model"""
    logger.info("Creating standard SSD model...")
    
    try:
        # Create standard model
        model = create_ssd_model(model_type='standard')
        
        # Save model
        torch.save(model.state_dict(), MODEL_PATHS['standard'])
        logger.info(f"Standard model saved to: {MODEL_PATHS['standard']}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to create standard model: {e}")
        return False


def test_model(model_path, model_type):
    """Test a saved model"""
    logger.info(f"Testing {model_type} model...")
    
    try:
        # Load model
        model = create_ssd_model(model_type=model_type)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 300, 300)
        
        # Test inference
        with torch.no_grad():
            predictions = model.predict(dummy_input)
        
        logger.info(f"{model_type} model test successful")
        return True
    except Exception as e:
        logger.error(f"Failed to test {model_type} model: {e}")
        return False


def main():
    """Main function"""
    logger.info("Starting model download and preparation...")
    
    # Create model directory
    MODEL_PATHS['quantized'].parent.mkdir(parents=True, exist_ok=True)
    
    # Check if models already exist
    models_to_create = []
    for model_type, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            models_to_create.append(model_type)
        else:
            logger.info(f"{model_type} model already exists: {model_path}")
    
    if not models_to_create:
        logger.info("All models already exist. Testing models...")
        for model_type, model_path in MODEL_PATHS.items():
            test_model(model_path, model_type)
        return
    
    # Create models
    success_count = 0
    
    if 'quantized' in models_to_create:
        if create_quantized_model():
            success_count += 1
    
    if 'lite' in models_to_create:
        if create_lite_model():
            success_count += 1
    
    if 'standard' in models_to_create:
        if create_standard_model():
            success_count += 1
    
    # Test created models
    logger.info("Testing created models...")
    for model_type, model_path in MODEL_PATHS.items():
        if model_path.exists():
            test_model(model_path, model_type)
    
    logger.info(f"Model preparation completed: {success_count}/{len(models_to_create)} models created successfully")
    
    # Print model information
    logger.info("\nModel Information:")
    for model_type, model_path in MODEL_PATHS.items():
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"  {model_type}: {model_path} ({size_mb:.1f} MB)")
    
    logger.info("\nModels are ready for use!")


if __name__ == '__main__':
    main() 