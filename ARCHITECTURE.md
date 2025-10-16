# Architecture Diagram

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     HANDWRITING SYNTHESIS SYSTEM                          │
└─────────────────────────────────────────────────────────────────────────┘

INPUT PHASE
═══════════════════════════════════════════════════════════════════════════

┌──────────────────┐                      ┌──────────────────────┐
│  Input Text      │                      │  Style Samples       │
│  "Hello World"   │                      │  (Handwriting Imgs)  │
└────────┬─────────┘                      └──────────┬───────────┘
         │                                           │
         │ text_to_indices()                        │
         ▼                                           ▼
┌──────────────────┐                      ┌─────────────────────────┐
│ Character        │                      │ [N × 1 × H × W]        │
│ Indices Tensor   │                      │ Grayscale Images        │
└────────┬─────────┘                      └──────────┬──────────────┘
         │                                           │
         │                                           │

ENCODING PHASE
═══════════════════════════════════════════════════════════════════════════

         │                                           │
         ▼                                           ▼
┌─────────────────────────────────────────┐ ┌──────────────────────────┐
│      CONTENT ENCODER                    │ │   STYLE ENCODER          │
│                                         │ │                          │
│  ┌────────────────────────────┐        │ │  ┌────────────────────┐  │
│  │ Character Embeddings       │        │ │  │  CNN Backbone      │  │
│  │ [B × L × d_model]         │        │ │  │  (Conv + BatchNorm)│  │
│  └──────────┬─────────────────┘        │ │  └─────────┬──────────┘  │
│             │ +positional              │ │            │              │
│             ▼                          │ │  ┌─────────▼──────────────┤
│  ┌────────────────────────────┐        │ │  │   Dual-Head Split      │
│  │ Transformer Encoder        │        │ │  ├────────┬───────────────┤
│  │ (2 layers, 8 heads)       │        │ │  │        │               │
│  │  - Self-Attention         │        │ │  ▼        ▼               │
│  │  - Feed-Forward           │        │ │ ┌──────┐ ┌─────────────┐ │
│  └──────────┬─────────────────┘        │ │ │Writer│ │Glyph (Char) │ │
│             │                          │ │ │Head  │ │   Head      │ │
│             ▼                          │ │ │Global│ │   Local     │ │
│  [B × L × d_model]                    │ │ │Style │ │   Style     │ │
│  Content Features                     │ │ └───┬──┘ └──────┬──────┘ │
└────────────┬──────────────────────────┘ └─────┼───────────┼────────┘
             │                                   │           │
             │                                   ▼           ▼
             │                            [B×128]        [B×128]
             │                          Writer Style   Glyph Style
             │                                   │           │
             │                                   └─────┬─────┘
             │                                         │
             │                                         ▼
             │                                  ┌─────────────┐
             │                                  │ Style Fusion│
             │                                  │ (Linear+Add)│
             │                                  └──────┬──────┘
             │                                         │
             │                                    Combined Style
             │                                    [B × d_model]
             │                                         │

DECODING PHASE
═══════════════════════════════════════════════════════════════════════════

             │                                         │
             │                                         │
             └──────────────────┬──────────────────────┘
                                │
                                ▼
                    ┌──────────────────────────────┐
                    │  Positional Queries          │
                    │  + Style Information         │
                    │  [B × seq_len × d_model]    │
                    └───────────┬──────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER DECODER                                │
│                                                                       │
│  For each layer (6 layers):                                         │
│                                                                       │
│  ┌─────────────────────┐                                            │
│  │  Self-Attention     │  Look at previous outputs                  │
│  │  (8 heads)          │                                            │
│  └──────────┬──────────┘                                            │
│             ▼                                                         │
│  ┌─────────────────────┐                                            │
│  │  Cross-Attention    │  Attend to content features                │
│  │  (8 heads)          │                                            │
│  │  Q: from queries    │                                            │
│  │  K,V: from content  │                                            │
│  └──────────┬──────────┘                                            │
│             ▼                                                         │
│  ┌─────────────────────┐                                            │
│  │  Feed-Forward       │                                            │
│  │  (2048 hidden dim)  │                                            │
│  └──────────┬──────────┘                                            │
│             ▼                                                         │
│  [B × seq_len × d_model]                                            │
│  Decoded Features                                                    │
└──────────────────┬────────────────────────────────────────────────────┘
                   │
                   ▼

OUTPUT PHASE
═══════════════════════════════════════════════════════════════════════════

         ┌────────────────────────────────────────┐
         │         GMM DECODER                    │
         │                                        │
         │  ┌──────────────────────────┐         │
         │  │  Linear Layer            │         │
         │  │  [d_model → output_dim]  │         │
         │  └──────────┬───────────────┘         │
         │             │                         │
         │             ▼                         │
         │  Split into GMM Parameters:           │
         │                                       │
         │  ┌─ π (mixture weights)    [M]        │
         │  ├─ μ (means x,y)          [M×2]      │
         │  ├─ σ (std devs x,y)       [M×2]      │
         │  ├─ ρ (correlation)        [M]        │
         │  └─ pen (pen state)        [1]        │
         │             │                          │
         │             ▼                          │
         │  Apply Activations:                   │
         │  - Softmax on π                       │
         │  - Exp on σ                           │
         │  - Tanh on ρ                          │
         │  - Sigmoid on pen                     │
         └────────────┬─────────────────────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  SAMPLING           │
            │  (with temperature) │
            └─────────┬───────────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │  Stroke Sequence         │
         │  [B × seq_len × 3]      │
         │  (dx, dy, pen_state)    │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │  SVG PATH GENERATION     │
         │                          │
         │  1. Cumulative Sum       │
         │  2. Smooth Strokes       │
         │  3. Convert to Paths     │
         │  4. Create SVG           │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │  OUTPUT.SVG           │
         │  Vector Handwriting      │
         └──────────────────────────┘

═══════════════════════════════════════════════════════════════════════════

KEY DESIGN DECISIONS:
─────────────────────────────────────────────────────────────────────────

1. DUAL-HEAD STYLE ENCODING
   - Separates global (writer) and local (character) styles
   - Better captures individual writing characteristics
   - Improves style transfer quality

2. TRANSFORMER ARCHITECTURE
   - Replaces RNN/LSTM from older methods
   - Better long-range dependencies
   - Parallel training (faster)
   - More consistent outputs

3. MIXTURE DENSITY NETWORKS
   - Models natural variation in strokes
   - 20 Gaussian components (configurable)
   - Realistic, human-like trajectories
   - Temperature control for randomness

4. SVG VECTOR OUTPUT
   - Scalable to any resolution
   - Smooth, professional quality
   - Editable in graphics software
   - Small file size

═══════════════════════════════════════════════════════════════════════════

DATA FLOW DIMENSIONS:
─────────────────────────────────────────────────────────────────────────

Input Text:           [B × L]           B=batch, L=text length
Character Embeddings: [B × L × 512]     512=d_model
Content Features:     [B × L × 512]     
Style Images:         [B × N × 1 × 64 × 64]  N=style samples
Writer Style:         [B × 128]         
Glyph Style:          [B × 128]         
Decoder Queries:      [B × S × 512]     S=sequence length
GMM Parameters:       [B × S × 121]     121=(20×6)+1
Output Strokes:       [B × S × 3]       (dx, dy, pen)

═══════════════════════════════════════════════════════════════════════════
```

## Training Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                          TRAINING LOOP                               │
└──────────────────────────────────────────────────────────────────────┘

For each batch:

1. FORWARD PASS
   ├─ Encode style from reference images
   ├─ Encode content from text
   ├─ Decode with Transformer
   └─ Generate GMM parameters

2. COMPUTE LOSS
   ├─ Coordinate Loss: -log p(x,y | GMM)
   ├─ Pen State Loss: BCE(pred_pen, target_pen)
   └─ Total Loss = Coordinate + Pen

3. BACKWARD PASS
   ├─ Compute gradients
   ├─ Clip gradients (prevent explosion)
   └─ Update parameters

4. LOGGING
   └─ Track losses, save checkpoints

Repeat for 100+ epochs
```

## Generation Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GENERATION FLOW                               │
└──────────────────────────────────────────────────────────────────────┘

1. PREPARE INPUT
   ├─ Convert text to indices
   └─ Load style references

2. ENCODE
   ├─ Extract style embeddings
   └─ Encode text content

3. DECODE (Autoregressive)
   For t = 1 to sequence_length:
     ├─ Query position t
     ├─ Attend to content
     ├─ Generate GMM parameters
     ├─ Sample stroke point (dx, dy, pen)
     └─ Add to sequence

4. POST-PROCESS
   ├─ Smooth strokes (optional)
   ├─ Convert to absolute positions
   └─ Generate SVG paths

5. SAVE
   └─ Write to output.svg
```
