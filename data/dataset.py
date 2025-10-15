"""
Data loading utilities for handwriting datasets
Supports IAM-OnDB format and custom datasets
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from typing import List, Tuple, Dict


class HandwritingDataset(Dataset):
    """
    Dataset for handwriting synthesis
    
    Expected data format:
    - Stroke data: [N, 3] arrays of (dx, dy, pen_state)
    - Text labels: strings
    - Writer IDs: integers for grouping by writer
    
    Args:
        data_path: Path to dataset directory or file
        char_to_idx: Character to index mapping
        max_seq_len: Maximum sequence length for strokes
        max_text_len: Maximum text length
        style_samples_per_writer: Number of style samples to use per writer
    """
    def __init__(self, data_path, char_to_idx, max_seq_len=1000, 
                 max_text_len=100, style_samples_per_writer=5):
        self.char_to_idx = char_to_idx
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.style_samples_per_writer = style_samples_per_writer
        
        # Load data
        self.samples = self._load_data(data_path)
        
        # Group by writer for style sampling
        self.writer_samples = self._group_by_writer()
    
    def _load_data(self, data_path):
        """
        Load handwriting data from path
        
        Expected format: JSON file with list of samples
        Each sample: {
            'strokes': [[dx, dy, pen], ...],
            'text': 'transcription',
            'writer_id': 123
        }
        """
        if os.path.isfile(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            raise ValueError(f"Data path {data_path} not found")
    
    def _group_by_writer(self):
        """Group samples by writer ID"""
        writer_dict = {}
        for i, sample in enumerate(self.samples):
            writer_id = sample.get('writer_id', 0)
            if writer_id not in writer_dict:
                writer_dict[writer_id] = []
            writer_dict[writer_id].append(i)
        return writer_dict
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            char_indices: [max_text_len] - character indices
            char_mask: [max_text_len] - valid character mask
            strokes: [max_seq_len, 3] - stroke sequence
            stroke_mask: [max_seq_len] - valid stroke mask
            style_images: [style_samples_per_writer, 1, 64, 64] - style references
        """
        sample = self.samples[idx]
        
        # Get text and convert to indices
        text = sample['text']
        char_indices = [self.char_to_idx.get(c, self.char_to_idx.get('<UNK>', 0)) 
                       for c in text]
        
        # Pad or truncate text
        if len(char_indices) < self.max_text_len:
            char_mask = [1] * len(char_indices) + [0] * (self.max_text_len - len(char_indices))
            char_indices = char_indices + [self.char_to_idx.get('<PAD>', 0)] * (self.max_text_len - len(char_indices))
        else:
            char_indices = char_indices[:self.max_text_len]
            char_mask = [1] * self.max_text_len
        
        # Get strokes
        strokes = np.array(sample['strokes'], dtype=np.float32)
        
        # Pad or truncate strokes
        if len(strokes) < self.max_seq_len:
            stroke_mask = [1] * len(strokes) + [0] * (self.max_seq_len - len(strokes))
            padding = np.zeros((self.max_seq_len - len(strokes), 3), dtype=np.float32)
            strokes = np.vstack([strokes, padding])
        else:
            strokes = strokes[:self.max_seq_len]
            stroke_mask = [1] * self.max_seq_len
        
        # Get style samples from same writer
        writer_id = sample.get('writer_id', 0)
        style_indices = self._get_style_samples(writer_id, exclude_idx=idx)
        
        # Create style images
        style_images = []
        for style_idx in style_indices:
            style_strokes = np.array(self.samples[style_idx]['strokes'])
            style_img = self._create_style_image(style_strokes)
            style_images.append(style_img)
        
        style_images = np.stack(style_images, axis=0)  # [N, 1, H, W]
        
        return {
            'char_indices': torch.tensor(char_indices, dtype=torch.long),
            'char_mask': torch.tensor(char_mask, dtype=torch.float),
            'strokes': torch.tensor(strokes, dtype=torch.float),
            'stroke_mask': torch.tensor(stroke_mask, dtype=torch.float),
            'style_images': torch.tensor(style_images, dtype=torch.float)
        }
    
    def _get_style_samples(self, writer_id, exclude_idx=None):
        """
        Get random style samples from the same writer
        
        Args:
            writer_id: Writer ID
            exclude_idx: Index to exclude (the current sample)
        
        Returns:
            style_indices: List of sample indices for style
        """
        writer_samples = self.writer_samples.get(writer_id, [])
        
        # Exclude current sample
        if exclude_idx is not None and exclude_idx in writer_samples:
            writer_samples = [i for i in writer_samples if i != exclude_idx]
        
        # Sample randomly
        if len(writer_samples) >= self.style_samples_per_writer:
            style_indices = np.random.choice(writer_samples, 
                                            self.style_samples_per_writer, 
                                            replace=False)
        else:
            # If not enough samples, sample with replacement
            style_indices = np.random.choice(writer_samples, 
                                            self.style_samples_per_writer, 
                                            replace=True)
        
        return style_indices.tolist()
    
    def _create_style_image(self, strokes, img_size=(64, 64)):
        """
        Create a rasterized image from strokes
        
        Args:
            strokes: [N, 3] array
            img_size: (H, W) tuple
        
        Returns:
            image: [1, H, W] array
        """
        from utils.helpers import create_style_image
        return create_style_image(strokes, img_size)


def create_dataloader(data_path, char_to_idx, batch_size=32, 
                     shuffle=True, num_workers=4, **dataset_kwargs):
    """
    Create a DataLoader for handwriting synthesis
    
    Args:
        data_path: Path to dataset
        char_to_idx: Character to index mapping
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for HandwritingDataset
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = HandwritingDataset(data_path, char_to_idx, **dataset_kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def prepare_iam_ondb_data(iam_path, output_path):
    """
    Prepare IAM-OnDB dataset for handwriting synthesis
    
    This is a placeholder - you'll need to implement based on your IAM-OnDB format
    
    Args:
        iam_path: Path to IAM-OnDB dataset
        output_path: Path to save processed data
    """
    print("Preparing IAM-OnDB dataset...")
    print("Note: This is a placeholder. Implement based on your IAM-OnDB format.")
    
    # Example structure for processed data
    processed_data = []
    
    # TODO: Load and process IAM-OnDB files
    # Each sample should be converted to:
    # {
    #     'strokes': [[dx, dy, pen], ...],
    #     'text': 'transcription',
    #     'writer_id': 123
    # }
    
    # Save processed data
    with open(output_path, 'w') as f:
        json.dump(processed_data, f)
    
    print(f"Processed data saved to {output_path}")


def create_synthetic_dataset(num_samples=1000, num_writers=10, output_path='synthetic_data.json'):
    """
    Create a synthetic dataset for testing
    
    Args:
        num_samples: Number of samples to generate
        num_writers: Number of synthetic writers
        output_path: Path to save dataset
    """
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    data = []
    texts = [
        "Hello world",
        "The quick brown fox",
        "jumps over the lazy dog",
        "Machine learning is fascinating",
        "Handwriting synthesis with AI",
        "Deep neural networks",
        "Transformer architecture",
        "Style transfer model"
    ]
    
    for i in range(num_samples):
        # Random text
        text = np.random.choice(texts)
        
        # Random writer
        writer_id = np.random.randint(0, num_writers)
        
        # Generate random stroke data (simple simulation)
        num_strokes = len(text) * 8 + np.random.randint(-10, 10)
        strokes = []
        
        for j in range(num_strokes):
            dx = np.random.randn() * 2
            dy = np.random.randn() * 2
            pen = 1 if j % 10 == 0 else 0  # Pen up occasionally
            strokes.append([float(dx), float(dy), float(pen)])
        
        data.append({
            'strokes': strokes,
            'text': text,
            'writer_id': writer_id
        })
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Synthetic dataset saved to {output_path}")
    return output_path
