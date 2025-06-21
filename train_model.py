#!/usr/bin/env python3
"""
Training script for fine-tuning SSD model on custom datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDHead
import numpy as np
import argparse
import logging
import json
from pathlib import Path
from tqdm import tqdm
import time

from models.ssd_model import create_ssd_model, COCO_CLASSES
from config import MODEL_CONFIG, PERFORMANCE_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset for object detection"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.annotations = []
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from directory"""
        # Implementation depends on dataset format
        # This is a placeholder for COCO or PASCAL VOC format
        pass
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Implementation for loading image and annotations
        # This is a placeholder
        pass


def create_dataloaders(train_dir, val_dir, batch_size=4, num_workers=4):
    """Create data loaders for training and validation"""
    # Define transforms
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CustomDataset(train_dir, transform=train_transform)
    val_dataset = CustomDataset(val_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate function for object detection"""
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{losses.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    return total_loss / num_batches


def validate(model, data_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Validation'):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logger.info(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
    return epoch, loss


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train SSD model on custom dataset')
    parser.add_argument('--train-dir', type=str, required=True,
                       help='Training data directory')
    parser.add_argument('--val-dir', type=str, required=True,
                       help='Validation data directory')
    parser.add_argument('--output-dir', type=str, default='trained_models',
                       help='Output directory for trained models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--model-type', type=str, default='quantized',
                       choices=['quantized', 'lite', 'standard'],
                       help='Model type to train')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and PERFORMANCE_CONFIG['use_gpu'] else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = create_ssd_model(model_type=args.model_type)
    model.to(device)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        args.train_dir,
        args.val_dir,
        batch_size=args.batch_size,
        num_workers=PERFORMANCE_CONFIG['num_threads']
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Log results
        logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / f'best_{args.model_type}_model.pth'
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved: {best_model_path}")
        
        # Save training history
        history = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }
        
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {output_dir}")


if __name__ == '__main__':
    main() 