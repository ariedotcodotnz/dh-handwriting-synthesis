"""
Complete Handwriting Synthesis Model
Combines style encoder, content encoder, transformer, and GMM decoder
"""
import torch
import torch.nn as nn
from .style_encoder import DualHeadStyleEncoder
from .content_encoder import ContentEncoder
from .transformer import build_transformer, TransformerDecoderLayer, TransformerDecoder
from .gmm_decoder import GMMDecoder


class HandwritingSynthesisModel(nn.Module):
    """
    Complete model for handwriting synthesis
    
    Architecture:
    1. Style Encoder: Extracts writer-wise and character-wise styles from reference samples
    2. Content Encoder: Encodes input text into content features
    3. Transformer Decoder: Fuses style and content with self-attention
    4. GMM Decoder: Generates stroke sequences
    
    Args:
        vocab_size: Size of character vocabulary
        d_model: Model dimension (default: 512)
        nhead: Number of attention heads (default: 8)
        num_decoder_layers: Number of decoder layers (default: 6)
        dim_feedforward: Feedforward dimension (default: 2048)
        num_mixtures: Number of GMM components (default: 20)
        writer_style_dim: Writer style dimension (default: 128)
        glyph_style_dim: Glyph style dimension (default: 128)
    """
    def __init__(self, vocab_size=100, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048,
                 num_mixtures=20, writer_style_dim=128, glyph_style_dim=128):
        super().__init__()
        
        self.d_model = d_model
        self.writer_style_dim = writer_style_dim
        self.glyph_style_dim = glyph_style_dim
        
        # Style encoder
        self.style_encoder = DualHeadStyleEncoder(
            input_channels=1,
            feature_dim=256,
            writer_style_dim=writer_style_dim,
            glyph_style_dim=glyph_style_dim
        )
        
        # Content encoder
        self.content_encoder = ContentEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=2,
            dim_feedforward=dim_feedforward
        )
        
        # Style projection layers
        self.writer_style_proj = nn.Linear(writer_style_dim, d_model)
        self.glyph_style_proj = nn.Linear(glyph_style_dim, d_model)
        
        # Transformer decoder for fusing content and style
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Positional encoding for output sequence
        # FIXED: Increased to 5000 to handle longer sequences
        self.pos_encoding = nn.Parameter(torch.randn(5000, d_model))

        # GMM decoder for stroke generation
        self.gmm_decoder = GMMDecoder(
            d_model=d_model,
            num_mixtures=num_mixtures,
            hidden_dim=512
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, char_indices, style_images, char_mask=None,
                target_len=None, return_attention=False):
        """
        Forward pass for training

        Args:
            char_indices: [B, L] - character indices of input text
            style_images: [B, N, 1, H, W] - reference handwriting samples for style
            char_mask: [B, L] - mask for valid characters (1=valid, 0=padding)
            target_len: Target sequence length for generation
            return_attention: Whether to return attention weights

        Returns:
            gmm_params: GMM parameters for stroke generation
            attention_weights: (optional) Attention weights
        """
        B, L = char_indices.shape

        # 1. Encode style from reference images
        writer_style, glyph_styles = self.style_encoder(style_images)
        # writer_style: [B, writer_style_dim]
        # glyph_styles: [B, N, glyph_style_dim]

        # Project styles to model dimension
        writer_style_proj = self.writer_style_proj(writer_style)  # [B, d_model]
        glyph_styles_mean = glyph_styles.mean(dim=1)  # Average over samples
        glyph_style_proj = self.glyph_style_proj(glyph_styles_mean)  # [B, d_model]

        # Combine writer and glyph styles
        combined_style = writer_style_proj + glyph_style_proj  # [B, d_model]

        # 2. Encode content
        content_features = self.content_encoder(char_indices, char_mask)
        # content_features: [B, L, d_model]

        # 3. Prepare for decoder
        # Determine target sequence length (strokes per character)
        if target_len is None:
            # Estimate: approximately 10 strokes per character
            target_len = L * 10

        # FIXED: Safety check for positional encoding limit
        max_pos_len = self.pos_encoding.size(0)
        if target_len > max_pos_len:
            print(f"Warning: target_len ({target_len}) exceeds positional encoding size ({max_pos_len})")
            print(f"Clamping to maximum size. Consider using chunked generation for long text.")
            target_len = max_pos_len

        # Create query sequence (learnable queries initialized with pos encoding)
        queries = self.pos_encoding[:target_len].unsqueeze(0).expand(B, -1, -1)
        # queries: [B, target_len, d_model]

        # Add style information to queries
        queries = queries + combined_style.unsqueeze(1)

        # Transpose for transformer (expects [len, batch, dim])
        queries_t = queries.transpose(0, 1)  # [target_len, B, d_model]
        content_t = content_features.transpose(0, 1)  # [L, B, d_model]

        # Create masks
        memory_key_padding_mask = None
        if char_mask is not None:
            memory_key_padding_mask = ~char_mask.bool()

        # 4. Decode with transformer
        decoded_features = self.decoder(
            tgt=queries_t,
            memory=content_t,
            memory_key_padding_mask=memory_key_padding_mask
        )
        # decoded_features: [target_len, B, d_model]

        # Transpose back
        decoded_features = decoded_features.transpose(0, 1)  # [B, target_len, d_model]

        # 5. Generate GMM parameters
        gmm_params = self.gmm_decoder(decoded_features)

        if return_attention:
            # Note: attention weights would need to be extracted from decoder layers
            return gmm_params, None

        return gmm_params

    def generate(self, text_indices, style_images, char_mask=None,
                 points_per_char=10, temperature=0.8):
        """
        Generate handwriting strokes from text and style

        Args:
            text_indices: [B, L] - character indices
            style_images: [B, N, 1, H, W] - style reference images
            char_mask: [B, L] - character mask
            points_per_char: Number of stroke points per character
            temperature: Sampling temperature

        Returns:
            strokes: [B, seq_len, 3] - generated strokes (dx, dy, pen_state)
        """
        self.eval()
        with torch.no_grad():
            B, L = text_indices.shape
            target_len = L * points_per_char

            # Forward pass
            gmm_params = self.forward(text_indices, style_images, char_mask, target_len)

            # Sample strokes
            strokes = self.gmm_decoder.sample(gmm_params, temperature=temperature)

        return strokes

    def compute_loss(self, char_indices, style_images, target_strokes,
                    char_mask=None, stroke_mask=None):
        """
        Compute training loss
        FIXED: Now properly passes stroke_mask to GMM decoder

        Args:
            char_indices: [B, L] - character indices
            style_images: [B, N, 1, H, W] - style reference images
            target_strokes: [B, seq_len, 3] - ground truth strokes
            char_mask: [B, L] - character mask
            stroke_mask: [B, seq_len] - stroke mask (1=valid, 0=padding)

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual loss components
        """
        B, seq_len, _ = target_strokes.shape

        # Forward pass
        gmm_params = self.forward(char_indices, style_images, char_mask, target_len=seq_len)

        # FIXED: Pass stroke_mask to GMM decoder
        loss, loss_dict = self.gmm_decoder.compute_loss(
            gmm_params, target_strokes, mask=stroke_mask
        )

        return loss, loss_dict

    def adapt_style(self, style_images):
        """
        Extract and cache style from reference images
        Useful for generating multiple samples with the same style

        Args:
            style_images: [B, N, 1, H, W] - style reference images

        Returns:
            style_dict: Dictionary containing style embeddings
        """
        with torch.no_grad():
            writer_style, glyph_styles = self.style_encoder(style_images)

        return {
            'writer_style': writer_style,
            'glyph_styles': glyph_styles
        }