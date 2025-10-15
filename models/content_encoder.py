"""Content Encoder - Processes text content"""
import torch
import torch.nn as nn
from .transformer import TransformerEncoder, TransformerEncoderLayer


class ContentEncoder(nn.Module):
    def __init__(self, vocab_size=100, d_model=512, nhead=8,
                 num_layers=2, dim_feedforward=2048, max_seq_len=500):
        super().__init__()
        self.d_model = d_model
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(0.1)

    def forward(self, char_indices, char_mask=None):
        B, L = char_indices.shape
        x = self.char_embedding(char_indices) * (self.d_model ** 0.5)
        x = x + self.pos_encoding[:L].unsqueeze(0)
        x = self.dropout(x)
        x = x.transpose(0, 1)
        key_padding_mask = None
        if char_mask is not None:
            key_padding_mask = ~char_mask.bool()
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = x.transpose(0, 1)
        return x


def create_vocabulary():
    chars = []
    chars.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
    chars.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
    chars.extend([str(i) for i in range(10)])
    punctuation = ['.', ',', '!', '?', ';', ':', "'", '"', '-', '(', ')',
                   '[', ']', '{', '}', '/', '\\', '@', '#', '$', '%',
                   '^', '&', '*', '+', '=', '<', '>', '|', '~', '`']
    chars.extend(punctuation)
    special_chars = ['<PAD>', '<UNK>', '<SOS>', '<EOS>', ' ']
    chars = special_chars + chars
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char


def text_to_indices(text, char_to_idx, max_len=None):
    unk_idx = char_to_idx.get('<UNK>', 0)
    pad_idx = char_to_idx.get('<PAD>', 0)
    indices = [char_to_idx.get(char, unk_idx) for char in text]
    if max_len is not None:
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
    return ''.join([idx_to_char.get(idx, '<UNK>') for idx in indices])