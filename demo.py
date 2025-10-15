"""
Demo script for handwriting synthesis
Tests the model with synthetic data
"""
import torch
import numpy as np
import os

from models.full_model import HandwritingSynthesisModel
from models.content_encoder import create_vocabulary, text_to_indices
from utils.svg_generator import create_svg
from utils.helpers import set_seed, create_style_image

def demo_generation():
    """
    Quick demo of handwriting generation
    """
    print("="*60)
    print("Handwriting Synthesis Demo")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create vocabulary
    print("\n1. Creating vocabulary...")
    char_to_idx, idx_to_char = create_vocabulary()
    vocab_size = len(char_to_idx)
    print(f"   Vocabulary size: {vocab_size}")
    
    # Create model (smaller for demo)
    print("\n2. Creating model...")
    model = HandwritingSynthesisModel(
        vocab_size=vocab_size,
        d_model=256,      # Smaller for demo
        nhead=4,
        num_decoder_layers=3,
        dim_feedforward=1024,
        num_mixtures=10,
        writer_style_dim=64,
        glyph_style_dim=64
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    print("   Note: Using untrained model - outputs will be random!")
    
    # Create synthetic style images
    print("\n3. Creating style references...")
    num_style_samples = 3
    style_images_list = []
    
    for i in range(num_style_samples):
        # Generate random strokes for style
        num_points = 50
        strokes = np.random.randn(num_points, 3) * 1.5
        strokes[:, 2] = (np.random.rand(num_points) > 0.85).astype(float)
        
        # Create image
        style_img = create_style_image(strokes, img_size=(64, 64))
        style_images_list.append(style_img)
    
    style_images = np.stack(style_images_list, axis=0)
    style_images = torch.tensor(style_images, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"   Style images shape: {style_images.shape}")
    
    # Demo texts
    demo_texts = [
        "Hello World!",
        "The quick brown fox jumps over the lazy dog.",
        "Handwriting synthesis with AI is amazing!",
        "Testing different temperatures and styles.",
    ]
    
    # Generate for each text
    print("\n4. Generating handwriting samples...")
    os.makedirs('demo_outputs', exist_ok=True)
    
    model.eval()
    for i, text in enumerate(demo_texts):
        print(f"\n   Generating: '{text}'")
        
        # Convert to indices
        indices, mask = text_to_indices(text, char_to_idx)
        char_indices = torch.tensor([indices], dtype=torch.long, device=device)
        char_mask = torch.tensor([mask], dtype=torch.float, device=device)
        
        # Generate strokes
        with torch.no_grad():
            strokes = model.generate(
                char_indices,
                style_images,
                char_mask,
                points_per_char=8,
                temperature=0.7
            )
        
        # Convert to numpy
        strokes_np = strokes[0].cpu().numpy()
        print(f"   Generated {len(strokes_np)} stroke points")
        
        # Save as SVG
        output_path = f'demo_outputs/demo_{i+1}.svg'
        create_svg(
            [strokes_np],
            output_path,
            width=1200,
            height=200,
            stroke_width=2,
            smooth=True
        )
        print(f"   Saved to {output_path}")
    
    # Generate a multi-line letter
    print("\n5. Generating full letter...")
    letter_lines = [
        "Dear Friend,",
        "",
        "I hope this letter finds you well.",
        "This is a demonstration of AI-generated handwriting.",
        "The model can create full pages of text in various styles.",
        "",
        "Best regards,",
        "Your AI Assistant"
    ]
    
    all_strokes = []
    for line in letter_lines:
        if not line.strip():
            # Empty line
            all_strokes.append(np.array([[0, 0, 1]]))
            continue
        
        indices, mask = text_to_indices(line, char_to_idx)
        char_indices = torch.tensor([indices], dtype=torch.long, device=device)
        char_mask = torch.tensor([mask], dtype=torch.float, device=device)
        
        with torch.no_grad():
            strokes = model.generate(char_indices, style_images, char_mask,
                                   points_per_char=8, temperature=0.7)
        
        all_strokes.append(strokes[0].cpu().numpy())
    
    letter_output = 'demo_outputs/demo_letter.svg'
    create_svg(
        all_strokes,
        letter_output,
        width=800,
        height=1000,
        stroke_width=2,
        smooth=True
    )
    print(f"   Letter saved to {letter_output}")
    
    # Summary
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print(f"\nGenerated {len(demo_texts)} samples + 1 letter")
    print(f"Outputs saved to: demo_outputs/")
    print("\nNotes:")
    print("- This demo uses an UNTRAINED model, so outputs are random")
    print("- To get realistic handwriting, train the model first:")
    print("  python train.py --num_epochs 50")
    print("- Then generate with the trained model:")
    print("  python generate.py --checkpoint checkpoints/best_model.pt \\")
    print("                     --text 'Your text here' --output output.svg")
    print("\nFor more information, see README.md")
    print("="*60)


if __name__ == '__main__':
    try:
        demo_generation()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
