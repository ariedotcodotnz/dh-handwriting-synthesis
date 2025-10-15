# Quick Start Guide

## Installation

```bash
# 1. Install dependencies
pip install torch torchvision numpy pillow svgwrite tqdm opencv-python

# For full requirements:
pip install -r requirements.txt
```

## Testing the System (Without Training)

```bash
# Run the demo to test the system
python demo.py
```

This will:
- Create an untrained model
- Generate several handwriting samples
- Save outputs to `demo_outputs/`
- Note: Outputs will be random since model is untrained

## Training Your Model

### Option 1: Quick Training with Synthetic Data

```bash
# Train on synthetic data (for testing)
python train.py --num_epochs 20 --batch_size 16

# This creates:
# - checkpoints/checkpoint_epoch_X.pt (every 5 epochs)
# - checkpoints/best_model.pt (best validation loss)
```

### Option 2: Training with Real Data

```bash
# 1. Prepare your data in JSON format:
# [
#   {
#     "strokes": [[dx, dy, pen], ...],
#     "text": "transcription",
#     "writer_id": 123
#   },
#   ...
# ]

# 2. Train the model
python train.py \
    --data_path your_data.json \
    --num_epochs 100 \
    --batch_size 32 \
    --d_model 512 \
    --num_layers 6 \
    --output_dir checkpoints
```

## Generating Handwriting

### From Command Line

```bash
# Simple text generation
python generate.py \
    --mode text \
    --text "Hello, World!" \
    --checkpoint checkpoints/best_model.pt \
    --output hello.svg

# Generate from text file
python generate.py \
    --mode file \
    --text_file example_letter.txt \
    --checkpoint checkpoints/best_model.pt \
    --output letter.svg \
    --temperature 0.7

# Interactive mode
python generate.py \
    --mode interactive \
    --checkpoint checkpoints/best_model.pt
```

### From Python

```python
import torch
from models.full_model import HandwritingSynthesisModel
from models.content_encoder import create_vocabulary, text_to_indices
from utils.svg_generator import create_svg

# Load model
model = HandwritingSynthesisModel(vocab_size=100)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input
char_to_idx, _ = create_vocabulary()
text = "Your handwritten text"
indices, mask = text_to_indices(text, char_to_idx)

# Create style references (simplified for demo)
style_images = torch.randn(1, 5, 1, 64, 64)  # Replace with real style images

# Generate
with torch.no_grad():
    strokes = model.generate(
        torch.tensor([indices]),
        style_images,
        torch.tensor([mask]),
        temperature=0.7
    )

# Save as SVG
create_svg([strokes[0].numpy()], 'output.svg')
```

## Adjusting Style and Quality

### Temperature Settings

- **0.5**: Very neat and consistent (formal documents)
- **0.7**: Natural variation (recommended default)
- **1.0**: More random variation (casual writing)
- **1.5**: High variation (artistic/messy style)

```bash
python generate.py \
    --text "Compare different styles" \
    --temperature 0.5 \
    --output neat.svg

python generate.py \
    --text "Compare different styles" \
    --temperature 1.5 \
    --output messy.svg
```

### SVG Customization

```bash
python generate.py \
    --text "Custom styling" \
    --output custom.svg \
    --width 1200 \
    --height 400 \
    --stroke_width 3
```

## Using Your Own Handwriting Style

1. **Collect samples**: Write several words/sentences on paper
2. **Digitize**: Scan or photograph your samples
3. **Preprocess**: Convert to required format (images or strokes)
4. **Extract style**:

```python
from models.style_encoder import DualHeadStyleEncoder
from utils.helpers import create_style_image

# Load your handwriting images
# Process them to extract style
style_encoder = DualHeadStyleEncoder()
# ... (implement based on your data format)
```

5. **Generate with your style**:

```python
# Use the extracted style images when generating
strokes = model.generate(text_indices, your_style_images, ...)
```

## Troubleshooting

### GPU Out of Memory

```bash
# Reduce batch size
python train.py --batch_size 8

# Reduce model size
python train.py --d_model 256 --num_layers 4

# Use CPU (slower but works)
python train.py --use_gpu False
```

### Poor Quality Output

1. Train longer (100+ epochs)
2. Adjust temperature (try 0.6-0.8)
3. Use more training data
4. Increase model capacity

### Model Not Learning

1. Check data format is correct
2. Reduce learning rate: `--lr 0.00005`
3. Ensure data has sufficient variation
4. Check for NaN losses (reduce learning rate if occurring)

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Check the research papers** listed in README
3. **Experiment with different architectures** by modifying model parameters
4. **Contribute improvements** via pull requests

## Common Workflows

### Training Pipeline

```bash
# 1. Prepare data
python -c "from data.dataset import create_synthetic_dataset; create_synthetic_dataset()"

# 2. Train model
python train.py --num_epochs 50

# 3. Generate samples
python generate.py --checkpoint checkpoints/best_model.pt --text "Test"

# 4. Evaluate and iterate
```

### Production Use

```bash
# 1. Train on large dataset
python train.py --data_path large_dataset.json --num_epochs 200

# 2. Test on validation set
python generate.py --mode file --text_file test_samples.txt

# 3. Deploy model
# - Save model: torch.save(model.state_dict(), 'production_model.pt')
# - Load in production: model.load_state_dict(torch.load('production_model.pt'))
# - Use model.generate() for inference
```

## Resources

- **Papers**: See README.md for citations
- **Datasets**: IAM-OnDB, CASIA-OLHWDB
- **Tutorials**: Check the demo.py for examples
- **Issues**: Open GitHub issues for problems

---

For more detailed information, see the main README.md file.
