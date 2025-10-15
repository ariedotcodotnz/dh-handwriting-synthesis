# Modern Handwriting Synthesis System

A state-of-the-art handwriting synthesis model that generates realistic handwritten text in SVG vector format. The model uses a **Style-Disentangled Transformer** architecture that separates writer-wise and character-wise styles for high-quality, controllable handwriting generation.

## Features

- **Full-Page Handwriting Generation**: Generate complete letters, documents, or notes
- **SVG Vector Output**: High-quality vector paths that scale perfectly
- **Style Adaptation**: Learn new handwriting styles from just a few samples
- **Dual-Style Encoding**: Separate writer-wise (global) and character-wise (local) styles
- **Transformer Architecture**: Modern attention-based architecture for better quality
- **Mixture Density Networks**: Gaussian Mixture Models for realistic stroke variation

## Architecture

The model combines several state-of-the-art techniques from recent research:

### Key Components

1. **Dual-Head Style Encoder**
   - **Writer Head**: Captures global writing characteristics (slant, spacing, overall style)
   - **Glyph Head**: Captures fine-grained character-specific details
   - Based on CNN feature extraction with attention mechanisms

2. **Content Encoder**
   - Processes input text using character embeddings
   - Transformer encoder for contextual understanding
   - Supports full character set (letters, numbers, punctuation)

3. **Transformer Decoder**
   - Multi-head self-attention and cross-attention
   - Fuses content and style information
   - Progressive generation of stroke sequences

4. **GMM Decoder**
   - Mixture Density Network with 20 Gaussian components
   - Generates (dx, dy, pen_state) stroke triplets
   - Produces smooth, natural-looking handwriting

### Research Foundation

This implementation is based on several influential papers:

1. **"Generating Sequences with Recurrent Neural Networks"** (Graves, 2013)
   - Original LSTM-based approach with Mixture Density Networks
   - ArXiv: [1308.0850](https://arxiv.org/abs/1308.0850)

2. **"Disentangling Writer and Character Styles for Handwriting Generation"** (SDT, CVPR 2023)
   - Style-disentangled Transformer architecture
   - Dual-head style encoding (writer-wise + character-wise)
   - ArXiv: [2303.14736](https://arxiv.org/abs/2303.14736)

3. **"DeepWriteSYN: On-line Handwriting Synthesis via Deep Short-Term Representations"** (2020)
   - VAE-based short-term synthesis
   - ArXiv: [2009.06308](https://arxiv.org/abs/2009.06308)

4. **"Making DeepWriting Erased"** (2018)
   - Conditional handwriting synthesis with style/content disentanglement
   - ArXiv: [1801.08379](https://arxiv.org/abs/1801.08379)

## Installation

```bash
# Clone the repository
git clone <your-repo>
cd handwriting_synthesis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For PyTorch, install the appropriate version for your system:
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### 1. Training

```bash
# Train with synthetic data (for testing)
python train.py --num_epochs 50 --batch_size 32 --output_dir checkpoints

# Train with your own data
python train.py \
    --data_path path/to/your/data.json \
    --num_epochs 100 \
    --batch_size 32 \
    --d_model 512 \
    --num_layers 6 \
    --output_dir checkpoints
```

### 2. Generation

```bash
# Generate from a single text string
python generate.py \
    --mode text \
    --text "Hello, this is handwritten text!" \
    --checkpoint checkpoints/best_model.pt \
    --output output.svg

# Generate from a text file (full letter/document)
python generate.py \
    --mode file \
    --text_file letter.txt \
    --checkpoint checkpoints/best_model.pt \
    --output letter.svg \
    --width 800 \
    --height 1000

# Interactive mode
python generate.py \
    --mode interactive \
    --checkpoint checkpoints/best_model.pt
```

### 3. Style Adaptation

To generate handwriting in a specific person's style:

1. **Collect style samples**: Get 5-10 handwriting samples from the target person
2. **Process samples**: Convert to the required format (strokes or images)
3. **Generate with style**:

```python
from models.full_model import HandwritingSynthesisModel
from models.content_encoder import create_vocabulary, text_to_indices
import torch

# Load model
model = HandwritingSynthesisModel(...)
model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])

# Load style images (shape: [1, N, 1, 64, 64])
style_images = load_your_style_images()  # Implement based on your data

# Generate
char_to_idx, _ = create_vocabulary()
text = "Your text here"
indices, mask = text_to_indices(text, char_to_idx)

strokes = model.generate(
    torch.tensor([indices]), 
    style_images,
    torch.tensor([mask]),
    temperature=0.7
)

# Save as SVG
from utils.svg_generator import create_svg
create_svg([strokes[0].numpy()], 'output.svg')
```

## Data Format

### Training Data

The model expects data in JSON format:

```json
[
    {
        "strokes": [
            [dx1, dy1, pen_state1],
            [dx2, dy2, pen_state2],
            ...
        ],
        "text": "transcription of the handwriting",
        "writer_id": 123
    },
    ...
]
```

Where:
- `dx, dy`: Pen displacement (change in x and y coordinates)
- `pen_state`: 0 = pen down (drawing), 1 = pen up (end of stroke/character)
- `writer_id`: Unique identifier for grouping samples by writer

### IAM-OnDB Dataset

To use the IAM On-Line Handwriting Database:

1. Download from: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
2. Process the data:

```python
from data.dataset import prepare_iam_ondb_data

prepare_iam_ondb_data(
    iam_path='path/to/iam-ondb',
    output_path='processed_data.json'
)
```

## Customization

### Model Parameters

Key hyperparameters you can adjust:

```bash
python train.py \
    --d_model 512              # Model dimension (256, 512, 1024)
    --nhead 8                  # Attention heads (4, 8, 16)
    --num_layers 6             # Decoder layers (4, 6, 8, 12)
    --num_mixtures 20          # GMM components (10, 20, 30)
    --writer_style_dim 128     # Writer style dimension
    --glyph_style_dim 128      # Character style dimension
```

### Generation Parameters

Control output quality and style:

```bash
python generate.py \
    --temperature 0.7          # Randomness (0.5=neat, 1.5=messy)
    --stroke_width 2           # SVG stroke thickness
    --width 800                # Canvas width
    --height 1000              # Canvas height
```

## Project Structure

```
handwriting_synthesis/
├── models/
│   ├── transformer.py         # Transformer encoder/decoder
│   ├── style_encoder.py       # Dual-head style encoder
│   ├── content_encoder.py     # Content encoder
│   ├── gmm_decoder.py         # Gaussian Mixture Model decoder
│   └── full_model.py          # Complete model
├── data/
│   └── dataset.py            # Data loading utilities
├── utils/
│   ├── svg_generator.py      # SVG generation
│   └── helpers.py            # Utility functions
├── train.py                  # Training script
├── generate.py               # Generation/inference script
└── requirements.txt          # Dependencies
```

## Example Outputs

### Temperature Effects

- **Temperature 0.5**: Clean, consistent handwriting (good for formal documents)
- **Temperature 0.7**: Natural variation (recommended for most cases)
- **Temperature 1.0**: More variation, realistic imperfections
- **Temperature 1.5**: High variation, artistic/messy style

### Use Cases

1. **Personal Letters**: Generate handwritten letters in your own style
2. **Educational Materials**: Create handwriting practice sheets
3. **Typography**: Design custom handwritten fonts
4. **Assistive Technology**: Help people unable to write by hand
5. **Data Augmentation**: Generate training data for handwriting recognition

## Technical Details

### Stroke Representation

Each stroke point is represented as `(dx, dy, pen_state)`:
- `dx, dy`: Relative displacement from previous point
- `pen_state`: Binary indicator (0=drawing, 1=pen up)

Cumulative sum of displacements gives absolute positions for rendering.

### Training Process

1. **Style Encoding**: CNN extracts features from reference handwriting images
2. **Content Encoding**: Transformer processes input text
3. **Fusion**: Decoder combines style and content with attention
4. **Stroke Generation**: GMM samples realistic stroke sequences
5. **Loss**: Negative log-likelihood of generating ground truth strokes

### Generation Process

1. Load trained model and style references
2. Encode target text and style
3. Sample strokes autoregressively from GMM
4. Convert strokes to SVG paths
5. Optionally smooth and post-process

## Advanced Usage

### Custom Style Encoder

To train on your own style representation:

```python
from models.style_encoder import DualHeadStyleEncoder

# Modify architecture
style_encoder = DualHeadStyleEncoder(
    input_channels=1,
    feature_dim=512,  # Increase capacity
    writer_style_dim=256,  # Larger style vectors
    glyph_style_dim=256
)
```

### Multi-Language Support

The model supports any character set. To add new characters:

```python
from models.content_encoder import create_vocabulary

# Extend vocabulary
additional_chars = ['你', '好', '世', '界']  # Chinese characters
# Modify create_vocabulary() to include these
```

## Performance Tips

1. **GPU Usage**: Use CUDA for 10-20x speedup during training
2. **Batch Size**: Larger batches (32-64) for better gradient estimates
3. **Style Samples**: More style references (5-10) improve adaptation
4. **Sequence Length**: Longer sequences need more memory; adjust `max_seq_len`
5. **Learning Rate**: Start with 1e-4, reduce if training is unstable

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size: `--batch_size 16`
   - Reduce model size: `--d_model 256 --num_layers 4`
   - Shorten sequences: `--max_seq_len 500`

2. **Poor Quality Outputs**
   - Train longer: `--num_epochs 200`
   - Adjust temperature: `--temperature 0.6`
   - Check data quality and preprocessing

3. **Style Transfer Not Working**
   - Ensure style samples are from same writer
   - Increase style dimension: `--writer_style_dim 256`
   - Use more style samples: `--style_samples 10`

## Citation

If you use this code in your research, please cite the relevant papers:

```bibtex
@article{graves2013generating,
  title={Generating sequences with recurrent neural networks},
  author={Graves, Alex},
  journal={arXiv preprint arXiv:1308.0850},
  year={2013}
}

@inproceedings{lyu2023disentangling,
  title={Disentangling writer and character styles for handwriting generation},
  author={Lyu, Guangyao and others},
  booktitle={CVPR},
  year={2023}
}
```

## Contributing

Contributions are welcome! Areas for improvement:

- Support for more datasets (CASIA, etc.)
- Real-time generation optimization
- Style interpolation between writers
- Mobile/web deployment
- Better style extraction from images

## License

This project is for research and educational purposes. Please check individual paper licenses for commercial use.

## Acknowledgments

This implementation builds upon research from:
- Alex Graves (DeepMind) - Original RNN handwriting synthesis
- ETH Zurich - DeepWriting project
- CVPR 2023 - Style-Disentangled Transformer

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].

---

**Note**: This model generates synthetic handwriting. While outputs are realistic, they should not be used to forge signatures or documents. Use responsibly and ethically.
