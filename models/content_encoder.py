"""
Content Encoder
Processes the text content to be synthesized into handwriting
"""
import torch
import torch.nn as nn
from .transformer import TransformerEncoder, TransformerEncoderLayer


class ContentEncoder(nn.Module):
    """
    Encodes text content using character embeddings and Transformer
    
    Args:
        vocab_size: Size of character vocabulary
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        max_seq_len: Maximum sequence length
    """
    def __init__(self, vocab_size=100, d_model=512, nhead=8, 
                 num_layers=2, dim_feedforward=2048, max_seq_len=500):
        super().__init__()
        self.d_model = d_model
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, char_indices, char_mask=None):
        """
        Args:
            char_indices: [B, L] - character indices
            char_mask: [B, L] - padding mask (1 for valid, 0 for padding)
        
        Returns:
            content_features: [B, L, d_model] - encoded content features
        """
        B, L = char_indices.shape
        
        # Embed characters
        x = self.char_embedding(char_indices) * (self.d_model ** 0.5)  # [B, L, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:L].unsqueeze(0)  # [B, L, d_model]
        x = self.dropout(x)
        
        # Transpose for transformer (expects [L, B, d_model])
        x = x.transpose(0, 1)  # [L, B, d_model]
        
        # Create attention mask from padding mask
        key_padding_mask = None
        if char_mask is not None:
            key_padding_mask = ~char_mask.bool()  # True for padding positions
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # [L, B, d_model]
        
        # Transpose back
        x = x.transpose(0, 1)  # [B, L, d_model]
        
        return x


def create_vocabulary():
    """
    Create a character vocabulary for handwriting synthesis
    Includes: uppercase, lowercase, digits, punctuation, special chars
    """
    chars = []
    
    # Lowercase letters
    chars.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
    
    # Uppercase letters
    chars.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
    
    # Digits
    chars.extend([str(i) for i in range(10)])
    
    # Common punctuation and symbols
    punctuation = ['.', ',', '!', '?', ';', ':', "'", '"', '-', '(', ')', 
                   '[', ']', '{', '}', '/', '\\', '@', '#', '$', '%', 
                   '^', '&', '*', '+', '=', '<', '>', '|', '~', '`']
    chars.extend(punctuation)
    
    # Special tokens
    special_chars = ['<PAD>', '<UNK>', '<SOS>', '<EOS>', ' ']  # Space is important!
    chars = special_chars + chars
    
    # Create char to index mapping
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return char_to_idx, idx_to_char


def text_to_indices(text, char_to_idx, max_len=None):
    """
    Convert text string to character indices
    
    Args:
        text: Input text string
        char_to_idx: Character to index mapping
        max_len: Maximum length (pad or truncate)
    
    Returns:
        indices: List of character indices
        mask: Validity mask (1 for valid chars, 0 for padding)
    """
    unk_idx = char_to_idx.get('<UNK>', 0)
    pad_idx = char_to_idx.get('<PAD>', 0)
    
    # Convert characters to indices
    indices = [char_to_idx.get(char, unk_idx) for char in text]
    
    if max_len is not None:
        # Pad or truncate
        if len(indices) < max_len:
            mask = [1] * len(indices) + [0] * (max_len - len(indices))
            indices = indices + [pad_idx] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
            mask = [1] * max_len
    else:
        mask = [1] * len(indices)
    
    return indices, mask


def indices_to_text(indices, idx_to_char):
    """
    Convert character indices back to text
    
    Args:
        indices: List of character indices
        idx_to_char: Index to character mapping
    
    Returns:
        text: Decoded text string
    """
    return ''.join([idx_to_char.get(idx, '<UNK>') for idx in indices])
