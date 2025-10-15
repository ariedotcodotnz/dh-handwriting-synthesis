"""
Utility helper functions
"""
import torch
import numpy as np
import random
import os
import json
from typing import Dict, List, Tuple


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: (Optional) Optimizer to load state into
        device: Device to load model on
    
    Returns:
        epoch: Epoch number from checkpoint
        loss: Loss from checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_gpu=True):
    """
    Get the best available device
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
    
    Returns:
        device: torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def normalize_strokes(strokes):
    """
    Normalize stroke coordinates to zero mean and unit variance
    
    Args:
        strokes: [N, 3] array of (dx, dy, pen_state)
    
    Returns:
        normalized: Normalized strokes
        stats: Dictionary of normalization statistics (mean, std)
    """
    strokes = np.array(strokes)
    
    # Separate coordinates and pen state
    coords = strokes[:, :2]
    pen = strokes[:, 2:]
    
    # Compute statistics
    mean = np.mean(coords, axis=0)
    std = np.std(coords, axis=0) + 1e-8  # Avoid division by zero
    
    # Normalize
    normalized_coords = (coords - mean) / std
    
    # Recombine
    normalized = np.concatenate([normalized_coords, pen], axis=1)
    
    stats = {'mean': mean, 'std': std}
    return normalized, stats


def denormalize_strokes(normalized_strokes, stats):
    """
    Denormalize strokes back to original scale
    
    Args:
        normalized_strokes: Normalized strokes
        stats: Dictionary with 'mean' and 'std'
    
    Returns:
        strokes: Denormalized strokes
    """
    normalized_strokes = np.array(normalized_strokes)
    
    # Separate
    coords = normalized_strokes[:, :2]
    pen = normalized_strokes[:, 2:]
    
    # Denormalize
    coords = coords * stats['std'] + stats['mean']
    
    # Recombine
    strokes = np.concatenate([coords, pen], axis=1)
    return strokes


def create_style_image(strokes, img_size=(64, 64)):
    """
    Create a rasterized image from strokes for style encoding
    
    Args:
        strokes: [N, 3] array of strokes
        img_size: Target image size (H, W)
    
    Returns:
        image: [1, H, W] numpy array (grayscale image)
    """
    import cv2
    
    H, W = img_size
    image = np.ones((H, W), dtype=np.uint8) * 255  # White background
    
    # Convert cumulative strokes to absolute positions
    positions = np.cumsum(strokes[:, :2], axis=0)
    
    # Normalize to fit image
    min_x, min_y = np.min(positions, axis=0)
    max_x, max_y = np.max(positions, axis=0)
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    
    if range_x == 0:
        range_x = 1
    if range_y == 0:
        range_y = 1
    
    # Scale to fit with margin
    margin = 5
    scale_x = (W - 2 * margin) / range_x
    scale_y = (H - 2 * margin) / range_y
    scale = min(scale_x, scale_y)
    
    # Transform positions
    positions = (positions - [min_x, min_y]) * scale + margin
    positions = positions.astype(np.int32)
    
    # Draw strokes
    for i in range(len(positions) - 1):
        if strokes[i, 2] < 0.5:  # Pen down
            cv2.line(image, tuple(positions[i]), tuple(positions[i+1]), 0, 2)
    
    # Normalize to [0, 1] and invert (black on white -> white on black for network)
    image = 1.0 - (image / 255.0)
    image = image[np.newaxis, ...]  # Add channel dimension
    
    return image.astype(np.float32)


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config, config_path):
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize_strokes_matplotlib(strokes, title="Handwriting", save_path=None):
    """
    Visualize strokes using matplotlib
    
    Args:
        strokes: [N, 3] array of strokes
        title: Plot title
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    # Convert to absolute positions
    positions = np.cumsum(strokes[:, :2], axis=0)
    pen_states = strokes[:, 2]
    
    plt.figure(figsize=(12, 4))
    plt.title(title)
    
    # Plot strokes
    current_stroke = []
    for i, (x, y) in enumerate(positions):
        current_stroke.append([x, y])
        
        if pen_states[i] > 0.5 or i == len(positions) - 1:  # Pen up or end
            if len(current_stroke) > 1:
                stroke_array = np.array(current_stroke)
                plt.plot(stroke_array[:, 0], -stroke_array[:, 1], 'k-', linewidth=2)
            current_stroke = []
    
    plt.axis('equal')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
