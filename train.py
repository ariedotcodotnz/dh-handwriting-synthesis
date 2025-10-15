"""
Training script for handwriting synthesis model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

from models.full_model import HandwritingSynthesisModel
from models.content_encoder import create_vocabulary
from data.dataset import create_dataloader, create_synthetic_dataset
from utils.helpers import (set_seed, save_checkpoint, load_checkpoint, 
                           count_parameters, get_device, AverageMeter)


def train_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: Handwriting synthesis model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        # Move to device
        char_indices = batch['char_indices'].to(device)
        char_mask = batch['char_mask'].to(device)
        strokes = batch['strokes'].to(device)
        style_images = batch['style_images'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        loss, loss_dict = model.compute_loss(
            char_indices, style_images, strokes, char_mask
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), char_indices.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'coord': f"{loss_dict.get('coord_loss', 0):.4f}",
            'pen': f"{loss_dict.get('pen_loss', 0):.4f}"
        })
    
    return loss_meter.avg


def validate(model, dataloader, device):
    """
    Validate model
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        device: Device
    
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            char_indices = batch['char_indices'].to(device)
            char_mask = batch['char_mask'].to(device)
            strokes = batch['strokes'].to(device)
            style_images = batch['style_images'].to(device)
            
            loss, _ = model.compute_loss(
                char_indices, style_images, strokes, char_mask
            )
            
            loss_meter.update(loss.item(), char_indices.size(0))
    
    return loss_meter.avg


def main(args):
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(prefer_gpu=args.use_gpu)
    
    # Create vocabulary
    print("Creating vocabulary...")
    char_to_idx, idx_to_char = create_vocabulary()
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create or load dataset
    if not os.path.exists(args.data_path):
        print(f"Dataset not found at {args.data_path}")
        print("Creating synthetic dataset for demonstration...")
        args.data_path = create_synthetic_dataset(
            num_samples=args.num_synthetic_samples,
            num_writers=10,
            output_path='synthetic_data.json'
        )
    
    # Create data loaders
    print("Loading dataset...")
    train_loader = create_dataloader(
        args.data_path,
        char_to_idx,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        max_text_len=args.max_text_len,
        style_samples_per_writer=args.style_samples
    )
    
    # For validation, we'd typically have a separate dataset
    # For this demo, we'll use the same dataset
    val_loader = create_dataloader(
        args.data_path,
        char_to_idx,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        max_text_len=args.max_text_len,
        style_samples_per_writer=args.style_samples
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = HandwritingSynthesisModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        num_mixtures=args.num_mixtures,
        writer_style_dim=args.writer_style_dim,
        glyph_style_dim=args.glyph_style_dim
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Load checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        if os.path.exists(args.resume):
            start_epoch, _ = load_checkpoint(args.resume, model, optimizer, device)
            start_epoch += 1
        else:
            print(f"Checkpoint {args.resume} not found, starting from scratch")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.6f}")

        # Learning rate scheduling
        old_lr = current_lr
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != old_lr:
            print(f"  → Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            print(f"New best model saved! Loss: {best_loss:.4f}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Models saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train handwriting synthesis model')

    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/handwriting_data.json',
                       help='Path to training data')
    parser.add_argument('--num_synthetic_samples', type=int, default=1000,
                       help='Number of synthetic samples if creating demo dataset')

    # Model arguments
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                       help='Feedforward dimension')
    parser.add_argument('--num_mixtures', type=int, default=20,
                       help='Number of GMM components')
    parser.add_argument('--writer_style_dim', type=int, default=128,
                       help='Writer style dimension')
    parser.add_argument('--glyph_style_dim', type=int, default=128,
                       help='Glyph style dimension')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--max_seq_len', type=int, default=1000,
                       help='Maximum stroke sequence length')
    parser.add_argument('--max_text_len', type=int, default=100,
                       help='Maximum text length')
    parser.add_argument('--style_samples', type=int, default=5,
                       help='Number of style samples per writer')

    # System arguments
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers (0 for Windows compatibility)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Checkpoint arguments
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    main(args)