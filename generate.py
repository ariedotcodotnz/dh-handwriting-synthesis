"""
Generate handwriting from text using trained model
ENHANCED VERSION with real IAM style references and improved long-text handling
"""
import torch
import argparse
import os
import numpy as np
import json
from pathlib import Path

from models.full_model import HandwritingSynthesisModel
from models.content_encoder import create_vocabulary, text_to_indices
from utils.svg_generator import create_full_page_svg, batch_generate_pages, create_svg
from utils.helpers import load_checkpoint, get_device, create_style_image, set_seed


class StyleReferenceManager:
    """
    Manages loading and caching of style references from IAM dataset
    """
    def __init__(self, data_path, char_to_idx, device='cpu'):
        """
        Args:
            data_path: Path to processed IAM JSON data
            char_to_idx: Character to index mapping
            device: Device to load tensors on
        """
        self.data_path = data_path
        self.char_to_idx = char_to_idx
        self.device = device
        self.data = None
        self.writer_samples = {}
        self.style_cache = {}

        if os.path.exists(data_path):
            self._load_data()
        else:
            print(f"Warning: Data path {data_path} not found")
            print("Please provide a valid path to processed IAM data")

    def _load_data(self):
        """Load and index the dataset by writer"""
        print(f"Loading style reference data from {self.data_path}...")
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)

        # Index by writer
        for idx, sample in enumerate(self.data):
            writer_id = sample.get('writer_id', 0)
            if writer_id not in self.writer_samples:
                self.writer_samples[writer_id] = []
            self.writer_samples[writer_id].append(idx)

        print(f"Loaded {len(self.data)} samples from {len(self.writer_samples)} writers")

        # Print some writer statistics
        writer_counts = {wid: len(samples) for wid, samples in self.writer_samples.items()}
        top_writers = sorted(writer_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 writers by sample count:")
        for wid, count in top_writers:
            print(f"  Writer {wid}: {count} samples")

    def get_writer_ids(self):
        """Get list of all available writer IDs"""
        return list(self.writer_samples.keys())

    def get_writer_info(self, writer_id):
        """Get information about a specific writer"""
        if writer_id not in self.writer_samples:
            return None

        sample_indices = self.writer_samples[writer_id]
        samples = [self.data[idx] for idx in sample_indices]

        avg_strokes = np.mean([len(s['strokes']) for s in samples])
        texts = [s['text'][:50] for s in samples[:3]]  # First 3 sample texts

        return {
            'writer_id': writer_id,
            'num_samples': len(sample_indices),
            'avg_strokes': avg_strokes,
            'sample_texts': texts
        }

    def load_style_references(self, writer_id=None, num_samples=5,
                             img_size=(64, 64), cache=True):
        """
        Load style reference images from real handwriting samples

        Args:
            writer_id: Specific writer ID (None = random writer)
            num_samples: Number of style samples to load
            img_size: Size of style images (H, W)
            cache: Whether to cache the style images

        Returns:
            style_images: [1, N, 1, H, W] tensor of style images
            writer_id: The writer ID used
        """
        if self.data is None:
            print("No data loaded. Creating synthetic style images...")
            return self._create_synthetic_styles(num_samples, img_size), 0

        # Check cache
        cache_key = (writer_id, num_samples, img_size)
        if cache and cache_key in self.style_cache:
            return self.style_cache[cache_key], writer_id

        # Select writer
        if writer_id is None:
            # Choose writer with most samples
            writer_id = max(self.writer_samples.keys(),
                          key=lambda k: len(self.writer_samples[k]))
            print(f"Auto-selected writer {writer_id} with {len(self.writer_samples[writer_id])} samples")

        if writer_id not in self.writer_samples:
            print(f"Writer {writer_id} not found. Using random writer.")
            writer_id = list(self.writer_samples.keys())[0]

        # Get samples from this writer
        sample_indices = self.writer_samples[writer_id]

        # Select samples (prefer diverse samples)
        if len(sample_indices) >= num_samples:
            # Spread out sample selection
            step = len(sample_indices) // num_samples
            selected_indices = [sample_indices[i * step] for i in range(num_samples)]
        else:
            # Not enough samples, repeat some
            selected_indices = sample_indices * (num_samples // len(sample_indices) + 1)
            selected_indices = selected_indices[:num_samples]

        # Create style images from strokes
        style_images_list = []
        for idx in selected_indices:
            sample = self.data[idx]
            strokes = np.array(sample['strokes'], dtype=np.float32)

            # Create style image
            style_img = create_style_image(strokes, img_size)
            style_images_list.append(style_img)

        # Stack and convert to tensor
        style_images = np.stack(style_images_list, axis=0)  # [N, 1, H, W]
        style_images = torch.tensor(style_images, dtype=torch.float32).unsqueeze(0)  # [1, N, 1, H, W]
        style_images = style_images.to(self.device)

        # Cache if requested
        if cache:
            self.style_cache[cache_key] = style_images

        print(f"Loaded {num_samples} style references from writer {writer_id}")
        return style_images, writer_id

    def _create_synthetic_styles(self, num_samples, img_size):
        """Fallback: create synthetic style images"""
        print("Creating synthetic style images (not recommended for production)")
        style_images_list = []

        for _ in range(num_samples):
            # Generate random strokes
            num_points = 50
            strokes = np.random.randn(num_points, 3) * 2
            strokes[:, 2] = (np.random.rand(num_points) > 0.9).astype(float)

            # Create image
            style_img = create_style_image(strokes, img_size)
            style_images_list.append(style_img)

        style_images = np.stack(style_images_list, axis=0)
        style_images = torch.tensor(style_images, dtype=torch.float32).unsqueeze(0)
        return style_images.to(self.device)


class ChunkedTextGenerator:
    """
    Generate long text in chunks to handle long-range dependencies better
    Implements sliding window approach with context overlap
    """
    def __init__(self, model, style_images, char_to_idx, device):
        self.model = model
        self.style_images = style_images
        self.char_to_idx = char_to_idx
        self.device = device
        self.model.eval()

    def generate_long_text(self, text, chunk_size=50, overlap=10,
                          points_per_char=10, temperature=0.7):
        """
        Generate handwriting for long text using chunked approach

        Args:
            text: Input text (can be very long)
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters between chunks
            points_per_char: Stroke points per character
            temperature: Sampling temperature

        Returns:
            strokes: [seq_len, 3] array of generated strokes
        """
        if len(text) <= chunk_size:
            # Short text, generate directly
            return self._generate_chunk(text, points_per_char, temperature)

        # Split into overlapping chunks
        chunks = []
        i = 0
        while i < len(text):
            end = min(i + chunk_size, len(text))
            chunk = text[i:end]
            chunks.append((i, chunk))
            i += chunk_size - overlap

        print(f"Generating {len(text)} characters in {len(chunks)} chunks...")

        all_strokes = []

        for idx, (start_pos, chunk) in enumerate(chunks):
            print(f"  Chunk {idx+1}/{len(chunks)}: '{chunk[:30]}...'")

            # Generate chunk
            chunk_strokes = self._generate_chunk(chunk, points_per_char, temperature)

            # Handle overlap blending
            if idx > 0 and overlap > 0:
                # Smooth transition by removing some overlap points
                blend_points = min(overlap * points_per_char, len(chunk_strokes) // 4)
                chunk_strokes = chunk_strokes[blend_points:]

            all_strokes.append(chunk_strokes)

        # Concatenate all chunks
        final_strokes = np.concatenate(all_strokes, axis=0)
        print(f"Generated {len(final_strokes)} stroke points total")

        return final_strokes

    def _generate_chunk(self, text, points_per_char=10, temperature=0.7):
        """Generate strokes for a single chunk of text"""
        with torch.no_grad():
            # Convert text to indices
            indices, mask = text_to_indices(text, self.char_to_idx)

            # Prepare tensors
            char_indices = torch.tensor([indices], dtype=torch.long, device=self.device)
            char_mask = torch.tensor([mask], dtype=torch.float, device=self.device)

            # Generate strokes
            strokes = self.model.generate(
                char_indices,
                self.style_images,
                char_mask,
                points_per_char=points_per_char,
                temperature=temperature
            )

            # Convert to numpy
            strokes_np = strokes[0].cpu().numpy()

        return strokes_np


def load_best_model(checkpoint_dir, model, optimizer=None, device='cpu'):
    """
    Load the best trained model from checkpoint directory

    Args:
        checkpoint_dir: Directory containing checkpoints
        model: Model instance to load weights into
        optimizer: Optional optimizer to load state
        device: Device to load on

    Returns:
        epoch: Epoch number of best model
        loss: Loss value of best model
    """
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')

    if not os.path.exists(best_model_path):
        # Try to find latest checkpoint
        print(f"Best model not found at {best_model_path}")
        print("Looking for latest checkpoint...")

        checkpoint_files = []
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir)
                              if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]

        if checkpoint_files:
            # Get the latest checkpoint
            epochs = []
            for f in checkpoint_files:
                try:
                    epoch_num = int(f.split('_')[-1].replace('.pt', ''))
                    epochs.append((epoch_num, f))
                except:
                    continue

            if epochs:
                latest_epoch, latest_file = max(epochs)
                latest_path = os.path.join(checkpoint_dir, latest_file)
                print(f"Using latest checkpoint: {latest_file}")
                return load_checkpoint(latest_path, model, optimizer, device)

        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    print(f"Loading best model from {best_model_path}")
    return load_checkpoint(best_model_path, model, optimizer, device)


def generate_from_text(text, model, style_images, char_to_idx, device,
                      output_path='output.svg', temperature=0.7,
                      use_chunking=True, **svg_kwargs):
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
        use_chunking: Whether to use chunked generation for long text
        **svg_kwargs: Additional SVG generation arguments
    """
    model.eval()

    # Decide whether to use chunking
    if use_chunking and len(text) > 50:
        print(f"Using chunked generation for long text ({len(text)} chars)")
        generator = ChunkedTextGenerator(model, style_images, char_to_idx, device)
        strokes_np = generator.generate_long_text(
            text,
            chunk_size=50,
            overlap=10,
            points_per_char=10,
            temperature=temperature
        )
    else:
        # Direct generation
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
    create_svg([strokes_np], output_path, **svg_kwargs)

    print(f"✓ Generated handwriting saved to {output_path}")


def generate_letter(text_lines, model, style_images, char_to_idx, device,
                   output_path='letter.svg', temperature=0.7,
                   use_chunking=True, **svg_kwargs):
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
        use_chunking: Use chunked generation for long lines
        **svg_kwargs: SVG arguments
    """
    print(f"Generating handwriting for {len(text_lines)} lines...")

    model.eval()
    all_strokes = []

    generator = ChunkedTextGenerator(model, style_images, char_to_idx, device) if use_chunking else None

    for i, line_text in enumerate(text_lines):
        if not line_text.strip():
            # Empty line - add spacing
            all_strokes.append(np.array([[0, 0, 1]]))
            continue

        print(f"Line {i+1}/{len(text_lines)}: '{line_text[:50]}...'")

        # Generate based on line length
        if use_chunking and len(line_text) > 50 and generator:
            strokes_np = generator.generate_long_text(
                line_text,
                chunk_size=50,
                overlap=10,
                points_per_char=10,
                temperature=temperature
            )
        else:
            # Direct generation
            indices, mask = text_to_indices(line_text, char_to_idx)
            char_indices = torch.tensor([indices], dtype=torch.long, device=device)
            char_mask = torch.tensor([mask], dtype=torch.float, device=device)

            with torch.no_grad():
                strokes = model.generate(
                    char_indices, style_images, char_mask,
                    points_per_char=10, temperature=temperature
                )
            strokes_np = strokes[0].cpu().numpy()

        all_strokes.append(strokes_np)

    # Create SVG
    create_svg(all_strokes, output_path, **svg_kwargs)
    print(f"✓ Letter saved to {output_path}")


def interactive_mode(model, style_manager, char_to_idx, device):
    """
    Interactive mode for generating handwriting

    Args:
        model: Trained model
        style_manager: StyleReferenceManager instance
        char_to_idx: Character mapping
        device: Device
    """
    print("\n" + "="*70)
    print("Interactive Handwriting Generation Mode")
    print("="*70)
    print("Commands:")
    print("  - Type text to generate handwriting")
    print("  - 'temp X' to set temperature (e.g., 'temp 0.8')")
    print("  - 'writer X' to change writer ID (e.g., 'writer 123')")
    print("  - 'list' to list available writers")
    print("  - 'info X' to get info about writer X")
    print("  - 'quit' to exit")
    print("="*70 + "\n")

    # Load initial style
    temperature = 0.7
    output_counter = 0
    style_images, current_writer = style_manager.load_style_references(num_samples=5)

    print(f"Current writer: {current_writer}")
    print(f"Current temperature: {temperature}\n")

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

            if user_input.lower().startswith('writer '):
                try:
                    writer_id = int(user_input.split()[1])
                    style_images, current_writer = style_manager.load_style_references(
                        writer_id=writer_id, num_samples=5
                    )
                    print(f"Switched to writer {current_writer}")
                except Exception as e:
                    print(f"Error switching writer: {e}")
                continue

            if user_input.lower() == 'list':
                writer_ids = style_manager.get_writer_ids()
                print(f"\nAvailable writers: {len(writer_ids)}")
                print("Top 10 writers:")
                for wid in writer_ids[:10]:
                    info = style_manager.get_writer_info(wid)
                    if info:
                        print(f"  Writer {wid}: {info['num_samples']} samples")
                continue

            if user_input.lower().startswith('info '):
                try:
                    writer_id = int(user_input.split()[1])
                    info = style_manager.get_writer_info(writer_id)
                    if info:
                        print(f"\nWriter {writer_id}:")
                        print(f"  Samples: {info['num_samples']}")
                        print(f"  Avg strokes: {info['avg_strokes']:.1f}")
                        print(f"  Sample texts:")
                        for text in info['sample_texts']:
                            print(f"    - {text}")
                    else:
                        print(f"Writer {writer_id} not found")
                except:
                    print("Usage: info <writer_id>")
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
                height=200,
                use_chunking=True
            )
            output_counter += 1

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main(args):
    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device(prefer_gpu=args.use_gpu)

    # Create vocabulary
    char_to_idx, idx_to_char = create_vocabulary()
    vocab_size = len(char_to_idx)

    # Create model
    print("\n" + "="*70)
    print("Creating handwriting synthesis model...")
    print("="*70)
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

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint (best model)
    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        try:
            epoch, loss = load_best_model(args.checkpoint_dir, model, device=device)
            print(f"✓ Loaded model from epoch {epoch} (loss: {loss:.4f})")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Proceeding with untrained model (outputs will be random)")
    elif args.checkpoint:
        # Single checkpoint file specified
        if os.path.exists(args.checkpoint):
            epoch, loss = load_checkpoint(args.checkpoint, model, device=device)
            print(f"✓ Loaded checkpoint from epoch {epoch} (loss: {loss:.4f})")
        else:
            print(f"Checkpoint {args.checkpoint} not found!")
            print("Proceeding with untrained model (outputs will be random)")
    else:
        print("⚠ No checkpoint specified. Using untrained model (outputs will be random)")
        print("Use --checkpoint_dir to specify checkpoint directory")

    # Initialize style reference manager
    print("\n" + "="*70)
    print("Loading style references...")
    print("="*70)
    style_manager = StyleReferenceManager(args.data_path, char_to_idx, device)

    # Load style images
    if args.writer_id is not None:
        style_images, writer_id = style_manager.load_style_references(
            writer_id=args.writer_id,
            num_samples=args.num_style_samples
        )
    else:
        # Auto-select best writer
        style_images, writer_id = style_manager.load_style_references(
            num_samples=args.num_style_samples
        )

    print(f"Using writer {writer_id} for style reference")

    # Generation mode
    print("\n" + "="*70)
    print(f"Generation mode: {args.mode}")
    print("="*70 + "\n")

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
            use_chunking=args.use_chunking,
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

        with open(args.text_file, 'r', encoding='utf-8') as f:
            text_lines = [line.rstrip() for line in f.readlines()]

        generate_letter(
            text_lines,
            model,
            style_images,
            char_to_idx,
            device,
            output_path=args.output,
            temperature=args.temperature,
            use_chunking=args.use_chunking,
            width=args.width,
            height=args.height,
            stroke_width=args.stroke_width
        )

    elif args.mode == 'interactive':
        # Interactive mode
        interactive_mode(model, style_manager, char_to_idx, device)

    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate handwriting from text using trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single line (uses best model automatically)
  python generate.py --mode text --text "Hello World" --checkpoint_dir checkpoints

  # Generate from file with specific writer
  python generate.py --mode file --text_file letter.txt --writer_id 123 --checkpoint_dir checkpoints
  
  # Interactive mode
  python generate.py --mode interactive --checkpoint_dir checkpoints --data_path iam_processed.json
  
  # Use specific checkpoint file
  python generate.py --mode text --text "Test" --checkpoint checkpoints/best_model.pt
        """
    )

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
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing checkpoints (will use best_model.pt)')
    parser.add_argument('--checkpoint', type=str, default='',
                       help='Specific checkpoint file (overrides checkpoint_dir)')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--num_mixtures', type=int, default=20)
    parser.add_argument('--writer_style_dim', type=int, default=128)
    parser.add_argument('--glyph_style_dim', type=int, default=128)

    # Data arguments
    parser.add_argument('--data_path', type=str, default='iam_processed.json',
                       help='Path to processed IAM data (for loading style references)')
    parser.add_argument('--writer_id', type=int, default=None,
                       help='Specific writer ID to use for style (None = auto-select)')
    parser.add_argument('--num_style_samples', type=int, default=5,
                       help='Number of style samples to use')

    # Generation arguments
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (0.1-2.0, lower=neater, higher=messier)')
    parser.add_argument('--use_chunking', action='store_true', default=True,
                       help='Use chunked generation for long text (better quality)')
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