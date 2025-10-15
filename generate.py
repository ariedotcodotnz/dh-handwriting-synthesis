"""
Generate handwriting from text using trained model
"""
import torch
import argparse
import os
import numpy as np

from models.full_model import HandwritingSynthesisModel
from models.content_encoder import create_vocabulary, text_to_indices
from utils.svg_generator import create_full_page_svg, batch_generate_pages
from utils.helpers import load_checkpoint, get_device, create_style_image, set_seed


def load_style_images_from_text(style_texts, model, char_to_idx, device):
    """
    Generate style images from text samples
    For demonstration - in practice, you'd use actual handwriting samples
    
    Args:
        style_texts: List of text strings to use as style references
        model: Trained model
        char_to_idx: Character mapping
        device: Device
    
    Returns:
        style_images: [1, N, 1, H, W] tensor
    """
    print("Note: Using synthetic style images for demonstration.")
    print("For best results, use actual handwriting images as style references.")
    
    # Create simple synthetic style images
    style_images_list = []
    for text in style_texts:
        # Generate random strokes (placeholder)
        num_points = len(text) * 8
        strokes = np.random.randn(num_points, 3) * 2
        strokes[:, 2] = (np.random.rand(num_points) > 0.9).astype(float)
        
        # Create image
        style_img = create_style_image(strokes, img_size=(64, 64))
        style_images_list.append(style_img)
    
    style_images = np.stack(style_images_list, axis=0)  # [N, 1, H, W]
    style_images = torch.tensor(style_images, dtype=torch.float32).unsqueeze(0)  # [1, N, 1, H, W]
    
    return style_images.to(device)


def generate_from_text(text, model, style_images, char_to_idx, device, 
                      output_path='output.svg', temperature=0.7, **svg_kwargs):
    """
    Generate handwriting for a single text string
    
    Args:
        text: Input text string
        model: Trained model
        style_images: Style reference images [1, N, 1, H, W]
        char_to_idx: Character to index mapping
        device: Device
        output_path: Path to save SVG
        temperature: Sampling temperature
        **svg_kwargs: Additional SVG generation arguments
    """
    model.eval()
    
    # Convert text to indices
    indices, mask = text_to_indices(text, char_to_idx)
    
    # Prepare tensors
    char_indices = torch.tensor([indices], dtype=torch.long, device=device)
    char_mask = torch.tensor([mask], dtype=torch.float, device=device)
    
    # Generate strokes
    with torch.no_grad():
        strokes = model.generate(
            char_indices, 
            style_images, 
            char_mask,
            points_per_char=10,
            temperature=temperature
        )
    
    # Convert to numpy
    strokes_np = strokes[0].cpu().numpy()
    
    # Save as SVG
    from utils.svg_generator import create_svg
    create_svg([strokes_np], output_path, **svg_kwargs)
    
    print(f"Generated handwriting saved to {output_path}")


def generate_letter(text_lines, model, style_images, char_to_idx, device,
                   output_path='letter.svg', temperature=0.7, **svg_kwargs):
    """
    Generate a full letter/document from multiple lines of text
    
    Args:
        text_lines: List of text strings (one per line)
        model: Trained model
        style_images: Style references
        char_to_idx: Character mapping
        device: Device
        output_path: Output path
        temperature: Sampling temperature
        **svg_kwargs: SVG arguments
    """
    print(f"Generating handwriting for {len(text_lines)} lines...")
    
    create_full_page_svg(
        text_lines,
        model,
        style_images,
        char_to_idx,
        output_path,
        device=device,
        temperature=temperature,
        **svg_kwargs
    )
    
    print(f"Letter saved to {output_path}")


def interactive_mode(model, style_images, char_to_idx, device):
    """
    Interactive mode for generating handwriting
    
    Args:
        model: Trained model
        style_images: Style references
        char_to_idx: Character mapping
        device: Device
    """
    print("\n" + "="*50)
    print("Interactive Handwriting Generation Mode")
    print("="*50)
    print("Commands:")
    print("  - Type text to generate handwriting")
    print("  - 'temp X' to set temperature (e.g., 'temp 0.8')")
    print("  - 'quit' to exit")
    print("="*50 + "\n")
    
    temperature = 0.7
    output_counter = 0
    
    while True:
        try:
            user_input = input("Enter text (or command): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower().startswith('temp '):
                try:
                    temperature = float(user_input.split()[1])
                    temperature = max(0.1, min(2.0, temperature))
                    print(f"Temperature set to {temperature}")
                except:
                    print("Invalid temperature. Use: temp 0.8")
                continue
            
            # Generate handwriting
            output_path = f'output_{output_counter:03d}.svg'
            generate_from_text(
                user_input,
                model,
                style_images,
                char_to_idx,
                device,
                output_path=output_path,
                temperature=temperature,
                width=1200,
                height=200
            )
            output_counter += 1
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main(args):
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(prefer_gpu=args.use_gpu)
    
    # Create vocabulary
    char_to_idx, idx_to_char = create_vocabulary()
    vocab_size = len(char_to_idx)
    
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
    
    # Load checkpoint
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            load_checkpoint(args.checkpoint, model, device=device)
        else:
            print(f"Checkpoint {args.checkpoint} not found!")
            print("Proceeding with untrained model (outputs will be random)")
    else:
        print("No checkpoint specified. Using untrained model (outputs will be random)")
    
    # Load or create style images
    if args.style_images_dir:
        # Load actual style images
        print(f"Loading style images from {args.style_images_dir}...")
        # TODO: Implement loading actual style images
        print("Note: Style image loading not yet implemented in this demo.")
        print("Using synthetic style images instead.")
        style_images = load_style_images_from_text(
            ['Sample text for style'] * args.num_style_samples,
            model, char_to_idx, device
        )
    else:
        # Use synthetic style images
        print("Using synthetic style images (for demonstration)")
        style_images = load_style_images_from_text(
            ['Sample text'] * args.num_style_samples,
            model, char_to_idx, device
        )
    
    # Generation mode
    if args.mode == 'text':
        # Generate from single text
        if not args.text:
            print("Error: --text required for text mode")
            return
        
        generate_from_text(
            args.text,
            model,
            style_images,
            char_to_idx,
            device,
            output_path=args.output,
            temperature=args.temperature,
            width=args.width,
            height=args.height,
            stroke_width=args.stroke_width
        )
    
    elif args.mode == 'file':
        # Generate from text file
        if not args.text_file:
            print("Error: --text_file required for file mode")
            return
        
        if not os.path.exists(args.text_file):
            print(f"Error: File {args.text_file} not found")
            return
        
        with open(args.text_file, 'r') as f:
            text_lines = [line.rstrip() for line in f.readlines()]
        
        generate_letter(
            text_lines,
            model,
            style_images,
            char_to_idx,
            device,
            output_path=args.output,
            temperature=args.temperature,
            width=args.width,
            height=args.height,
            stroke_width=args.stroke_width
        )
    
    elif args.mode == 'interactive':
        # Interactive mode
        interactive_mode(model, style_images, char_to_idx, device)
    
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate handwriting from text')
    
    # Mode
    parser.add_argument('--mode', type=str, default='text',
                       choices=['text', 'file', 'interactive'],
                       help='Generation mode')
    
    # Input
    parser.add_argument('--text', type=str, default='',
                       help='Text to generate (for text mode)')
    parser.add_argument('--text_file', type=str, default='',
                       help='Text file to generate (for file mode)')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default='',
                       help='Path to model checkpoint')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--num_mixtures', type=int, default=20)
    parser.add_argument('--writer_style_dim', type=int, default=128)
    parser.add_argument('--glyph_style_dim', type=int, default=128)
    
    # Style arguments
    parser.add_argument('--style_images_dir', type=str, default='',
                       help='Directory containing style reference images')
    parser.add_argument('--num_style_samples', type=int, default=5,
                       help='Number of style samples to use')
    
    # Generation arguments
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (0.1-2.0)')
    parser.add_argument('--output', type=str, default='output.svg',
                       help='Output file path')
    
    # SVG arguments
    parser.add_argument('--width', type=int, default=800,
                       help='SVG width')
    parser.add_argument('--height', type=int, default=1000,
                       help='SVG height')
    parser.add_argument('--stroke_width', type=int, default=2,
                       help='Stroke width')
    
    # System
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    main(args)
