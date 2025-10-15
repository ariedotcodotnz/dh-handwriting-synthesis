"""
Dual-Head Style Encoder
Extracts both writer-wise and character-wise style representations
Based on SDT (Style-Disentangled Transformer) architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """CNN for extracting visual features from handwriting images"""
    def __init__(self, input_channels=1, feature_dim=256):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, feature_dim, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_dim)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # x: [B, 1, H, W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # [B, 64, H/2, W/2]
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # [B, 128, H/4, W/4]
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # [B, 256, H/8, W/8]
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # [B, feature_dim, H/16, W/16]
        
        return x


class WriterHead(nn.Module):
    """
    Writer-wise style head
    Learns global writing characteristics (slant, spacing, overall style)
    """
    def __init__(self, feature_dim=256, style_dim=128):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, style_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, features):
        # features: [B, feature_dim, H, W]
        x = self.global_pool(features)  # [B, feature_dim, 1, 1]
        x = x.flatten(1)  # [B, feature_dim]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # [B, style_dim]
        return x


class GlyphHead(nn.Module):
    """
    Character-wise (glyph) style head
    Learns fine-grained character-specific style details
    """
    def __init__(self, feature_dim=256, style_dim=128):
        super().__init__()
        # Spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.adaptive_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.fc1 = nn.Linear(feature_dim * 4 * 4, 512)
        self.fc2 = nn.Linear(512, style_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, features):
        # features: [B, feature_dim, H, W]
        # Apply spatial attention
        attention = self.spatial_attention(features)  # [B, 1, H, W]
        weighted_features = features * attention
        
        x = self.adaptive_pool(weighted_features)  # [B, feature_dim, 4, 4]
        x = x.flatten(1)  # [B, feature_dim * 16]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # [B, style_dim]
        return x


class DualHeadStyleEncoder(nn.Module):
    """
    Complete dual-head style encoder
    Extracts both writer-wise and character-wise style representations
    
    Args:
        input_channels: Number of input image channels (1 for grayscale)
        feature_dim: Dimension of CNN features
        writer_style_dim: Dimension of writer-wise style embedding
        glyph_style_dim: Dimension of character-wise style embedding
    """
    def __init__(self, input_channels=1, feature_dim=256, 
                 writer_style_dim=128, glyph_style_dim=128):
        super().__init__()
        self.cnn = CNNFeatureExtractor(input_channels, feature_dim)
        self.writer_head = WriterHead(feature_dim, writer_style_dim)
        self.glyph_head = GlyphHead(feature_dim, glyph_style_dim)
        
        self.writer_style_dim = writer_style_dim
        self.glyph_style_dim = glyph_style_dim
    
    def forward(self, style_images):
        """
        Args:
            style_images: [B, num_style_samples, C, H, W] - batch of style reference images
        
        Returns:
            writer_style: [B, writer_style_dim] - writer-wise style embedding
            glyph_style: [B, num_style_samples, glyph_style_dim] - character-wise style embeddings
        """
        B, N, C, H, W = style_images.shape
        
        # Reshape to process all samples
        x = style_images.view(B * N, C, H, W)
        
        # Extract CNN features
        features = self.cnn(x)  # [B*N, feature_dim, H', W']
        
        # Extract writer-wise style (aggregate over all samples)
        features_reshaped = features.view(B, N, features.size(1), features.size(2), features.size(3))
        # Average features across style samples for writer-wise style
        writer_features = features_reshaped.mean(dim=1)  # [B, feature_dim, H', W']
        writer_style = self.writer_head(writer_features)  # [B, writer_style_dim]
        
        # Extract character-wise style (per sample)
        glyph_styles = self.glyph_head(features)  # [B*N, glyph_style_dim]
        glyph_styles = glyph_styles.view(B, N, -1)  # [B, N, glyph_style_dim]
        
        return writer_style, glyph_styles
    
    def encode_single(self, image):
        """
        Encode a single handwriting sample
        
        Args:
            image: [B, C, H, W] - single image
        
        Returns:
            writer_style: [B, writer_style_dim]
            glyph_style: [B, glyph_style_dim]
        """
        features = self.cnn(image)
        writer_style = self.writer_head(features)
        glyph_style = self.glyph_head(features)
        return writer_style, glyph_style
