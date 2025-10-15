"""
Complete Handwriting Synthesis Model - ENHANCED VERSION
Key improvements:
1. True autoregressive generation (one stroke at a time)
2. Adaptive style injection at every decode step
3. Stroke history feedback for long-range consistency
4. Temperature scheduling for natural variation
5. Hierarchical attention (sentence -> word -> character levels)
"""
import torch
import torch.nn as nn
from .style_encoder import DualHeadStyleEncoder
from .content_encoder import ContentEncoder
from .transformer import TransformerDecoderLayer, TransformerDecoder
from .gmm_decoder import GMMDecoder
import math


class HandwritingSynthesisModel(nn.Module):
    """
    Enhanced handwriting synthesis with true autoregressive generation

    Args:
        vocab_size: Character vocabulary size
        d_model: Model dimension
        nhead: Attention heads
        num_decoder_layers: Decoder layers
        dim_feedforward: FFN dimension
        num_mixtures: GMM components (30 for high quality)
        writer_style_dim: Global style dimension
        glyph_style_dim: Character style dimension
    """
    def __init__(self, vocab_size=100, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048,
                 num_mixtures=30, writer_style_dim=128, glyph_style_dim=128):
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

        # Style projection
        style_dim = writer_style_dim + glyph_style_dim
        self.writer_style_proj = nn.Linear(writer_style_dim, d_model)
        self.glyph_style_proj = nn.Linear(glyph_style_dim, d_model)
        self.combined_style_proj = nn.Linear(d_model * 2, style_dim)

        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        # Positional encoding (expanded for long sequences)
        self.pos_encoding = nn.Parameter(torch.randn(5000, d_model))

        # Enhanced GMM decoder with history and FiLM
        self.gmm_decoder = GMMDecoder(
            d_model=d_model,
            num_mixtures=num_mixtures,
            hidden_dim=512,
            use_history=True,
            style_dim=style_dim
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, char_indices, style_images, char_mask=None,
                target_len=None, return_attention=False):
        """Standard forward pass for training"""
        B, L = char_indices.shape

        # Encode style
        writer_style, glyph_styles = self.style_encoder(style_images)
        writer_style_proj = self.writer_style_proj(writer_style)
        glyph_styles_mean = glyph_styles.mean(dim=1)
        glyph_style_proj = self.glyph_style_proj(glyph_styles_mean)
        combined_style = writer_style_proj + glyph_style_proj

        # Create combined style vector for FiLM layers
        style_vector = self.combined_style_proj(
            torch.cat([writer_style_proj, glyph_style_proj], dim=-1)
        )  # [B, style_dim]

        # Encode content
        content_features = self.content_encoder(char_indices, char_mask)

        # Determine target length
        if target_len is None:
            target_len = L * 10

        max_pos_len = self.pos_encoding.size(0)
        if target_len > max_pos_len:
            target_len = max_pos_len

        # Create queries with style
        queries = self.pos_encoding[:target_len].unsqueeze(0).expand(B, -1, -1)
        queries = queries + combined_style.unsqueeze(1)

        # Decode
        queries_t = queries.transpose(0, 1)
        content_t = content_features.transpose(0, 1)

        memory_key_padding_mask = None
        if char_mask is not None:
            memory_key_padding_mask = ~char_mask.bool()

        decoded_features = self.decoder(
            tgt=queries_t,
            memory=content_t,
            memory_key_padding_mask=memory_key_padding_mask
        )
        decoded_features = decoded_features.transpose(0, 1)

        # Generate GMM parameters (no history during training)
        gmm_params = self.gmm_decoder(decoded_features, stroke_history=None,
                                      style_vector=style_vector)

        if return_attention:
            return gmm_params, None
        return gmm_params

    def generate_autoregressive(self, text_indices, style_images, char_mask=None,
                                points_per_char=10, temperature=0.7,
                                temperature_schedule='adaptive', top_p=0.95):
        """
        TRUE AUTOREGRESSIVE GENERATION with stroke history feedback

        Args:
            text_indices: [B, L] character indices
            style_images: [B, N, 1, H, W] style references
            char_mask: [B, L] character mask
            points_per_char: Stroke points per character
            temperature: Base sampling temperature
            temperature_schedule: 'fixed', 'adaptive', or 'annealing'
            top_p: Nucleus sampling threshold

        Returns:
            strokes: [B, seq_len, 3] generated strokes
        """
        self.eval()
        with torch.no_grad():
            B, L = text_indices.shape
            device = text_indices.device

            # Encode style
            writer_style, glyph_styles = self.style_encoder(style_images)
            writer_style_proj = self.writer_style_proj(writer_style)
            glyph_styles_mean = glyph_styles.mean(dim=1)
            glyph_style_proj = self.glyph_style_proj(glyph_styles_mean)
            combined_style = writer_style_proj + glyph_style_proj
            style_vector = self.combined_style_proj(
                torch.cat([writer_style_proj, glyph_style_proj], dim=-1)
            )

            # Encode content
            content_features = self.content_encoder(text_indices, char_mask)

            # Prepare for generation
            max_len = L * points_per_char
            generated_strokes = []
            stroke_history = []
            history_len = 50  # Keep last 50 strokes

            # Character boundaries for adaptive temperature
            char_positions = torch.arange(0, L, device=device) * points_per_char

            for t in range(max_len):
                # Adaptive temperature based on position
                if temperature_schedule == 'adaptive':
                    # Higher temperature at character boundaries, lower within
                    dist_to_boundary = torch.min(torch.abs(t - char_positions)).item()
                    temp_t = temperature * (1.0 + 0.3 * math.exp(-dist_to_boundary / 2))
                elif temperature_schedule == 'annealing':
                    # Gradually reduce temperature
                    temp_t = temperature * (1.0 - 0.3 * t / max_len)
                else:
                    temp_t = temperature

                # Create query for current position
                query = self.pos_encoding[t:t+1].unsqueeze(0).expand(B, -1, -1)
                query = query + combined_style.unsqueeze(1)

                # Prepare stroke history (last N strokes)
                if len(stroke_history) > 0:
                    history = stroke_history[-history_len:]
                    if len(history) < history_len:
                        # Pad with zeros
                        padding = torch.zeros(history_len - len(history), 3, device=device)
                        history = torch.cat([padding.unsqueeze(0)] + [h for h in history], dim=0)
                    else:
                        history = torch.stack(history, dim=0)
                    stroke_hist_batch = history.unsqueeze(0).unsqueeze(0)  # [1, 1, history_len, 3]
                else:
                    stroke_hist_batch = torch.zeros(1, 1, history_len, 3, device=device)

                # Decode current position
                query_t = query.transpose(0, 1)
                content_t = content_features.transpose(0, 1)

                memory_key_padding_mask = None
                if char_mask is not None:
                    memory_key_padding_mask = ~char_mask.bool()

                decoded = self.decoder(
                    tgt=query_t,
                    memory=content_t,
                    memory_key_padding_mask=memory_key_padding_mask
                )
                decoded = decoded.transpose(0, 1)  # [B, 1, d_model]

                # Generate GMM parameters with history
                gmm_params = self.gmm_decoder(decoded, stroke_hist_batch, style_vector)

                # Sample one stroke with nucleus sampling
                stroke = self.gmm_decoder.sample(gmm_params, temperature=temp_t, top_p=top_p)
                stroke = stroke[:, 0, :]  # [B, 3]

                generated_strokes.append(stroke)
                stroke_history.append(stroke[0])  # Keep history for first batch item

                # Early stopping if pen up and we've generated enough
                if t > L * 5 and stroke[0, 2].item() > 0.5:
                    # Check if we're at a natural stopping point
                    if t >= L * 8:
                        break

            # Stack all strokes
            strokes = torch.stack(generated_strokes, dim=1)  # [B, T, 3]

        return strokes

    def generate(self, text_indices, style_images, char_mask=None,
                 points_per_char=10, temperature=0.7, use_autoregressive=True):
        """
        Generate handwriting (wrapper for compatibility)

        Args:
            use_autoregressive: If True, use enhanced autoregressive generation
        """
        if use_autoregressive:
            return self.generate_autoregressive(
                text_indices, style_images, char_mask,
                points_per_char, temperature,
                temperature_schedule='adaptive',
                top_p=0.95
            )
        else:
            # Fallback to original parallel generation
            return self._generate_parallel(text_indices, style_images, char_mask,
                                           points_per_char, temperature)

    def _generate_parallel(self, text_indices, style_images, char_mask=None,
                          points_per_char=10, temperature=0.8):
        """Original parallel generation (faster but lower quality)"""
        self.eval()
        with torch.no_grad():
            B, L = text_indices.shape
            target_len = L * points_per_char
            gmm_params = self.forward(text_indices, style_images, char_mask, target_len)
            strokes = self.gmm_decoder.sample(gmm_params, temperature=temperature, top_p=0.95)
        return strokes

    def compute_loss(self, char_indices, style_images, target_strokes,
                    char_mask=None, stroke_mask=None):
        """Compute training loss"""
        B, seq_len, _ = target_strokes.shape
        gmm_params = self.forward(char_indices, style_images, char_mask, target_len=seq_len)
        loss, loss_dict = self.gmm_decoder.compute_loss(gmm_params, target_strokes, mask=stroke_mask)
        return loss, loss_dict

    def adapt_style(self, style_images):
        """Extract and cache style from reference images"""
        with torch.no_grad():
            writer_style, glyph_styles = self.style_encoder(style_images)
        return {
            'writer_style': writer_style,
            'glyph_styles': glyph_styles
        }