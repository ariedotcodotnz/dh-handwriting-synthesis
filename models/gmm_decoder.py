"""
Gaussian Mixture Model Decoder
Generates handwriting strokes as sequences of 2D points with pen states
Based on Alex Graves' approach and SDT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GMMDecoder(nn.Module):
    """
    Decodes content and style features into handwriting strokes
    Uses Mixture Density Network with Gaussian Mixture Model
    
    Output format for each stroke point:
    - (dx, dy): pen displacement
    - pen_state: 0 (pen down) or 1 (pen up/end of character)
    
    Args:
        d_model: Input feature dimension
        num_mixtures: Number of Gaussian mixture components
        hidden_dim: Hidden dimension for decoder
    """
    def __init__(self, d_model=512, num_mixtures=20, hidden_dim=512):
        super().__init__()
        self.d_model = d_model
        self.num_mixtures = num_mixtures
        
        # Project combined features to hidden dimension
        self.input_proj = nn.Linear(d_model, hidden_dim)
        
        # Mixture Density Network output layer
        # For each mixture: (mu_x, mu_y, sigma_x, sigma_y, rho, pi)
        # Plus pen state (end of stroke)
        self.output_size = num_mixtures * 6 + 1
        self.mdn_layer = nn.Linear(hidden_dim, self.output_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, features):
        """
        Args:
            features: [B, L, d_model] - combined content and style features
        
        Returns:
            gmm_params: Dictionary containing GMM parameters
                - pi: [B, L, num_mixtures] - mixture weights
                - mu: [B, L, num_mixtures, 2] - means (x, y)
                - sigma: [B, L, num_mixtures, 2] - std devs (x, y)
                - rho: [B, L, num_mixtures] - correlations
                - pen: [B, L, 1] - pen state logits
        """
        B, L, _ = features.shape
        
        # Project features
        h = F.relu(self.input_proj(features))  # [B, L, hidden_dim]
        h = self.dropout(h)
        
        # Generate MDN parameters
        out = self.mdn_layer(h)  # [B, L, output_size]
        
        # Split into components
        idx = 0
        pi = out[..., idx:idx+self.num_mixtures]  # Mixture weights
        idx += self.num_mixtures
        
        mu_x = out[..., idx:idx+self.num_mixtures]
        idx += self.num_mixtures
        mu_y = out[..., idx:idx+self.num_mixtures]
        idx += self.num_mixtures
        
        sigma_x = out[..., idx:idx+self.num_mixtures]
        idx += self.num_mixtures
        sigma_y = out[..., idx:idx+self.num_mixtures]
        idx += self.num_mixtures
        
        rho = out[..., idx:idx+self.num_mixtures]
        idx += self.num_mixtures
        
        pen = out[..., idx:]  # Pen state
        
        # Apply activation functions
        pi = F.softmax(pi, dim=-1)  # [B, L, num_mixtures]
        
        mu = torch.stack([mu_x, mu_y], dim=-1)  # [B, L, num_mixtures, 2]
        
        # Ensure positive standard deviations (minimum value to avoid numerical issues)
        sigma_x = torch.exp(sigma_x) + 1e-4  # [B, L, num_mixtures]
        sigma_y = torch.exp(sigma_y) + 1e-4
        sigma = torch.stack([sigma_x, sigma_y], dim=-1)  # [B, L, num_mixtures, 2]
        
        # Correlation coefficient bounded to [-1, 1]
        rho = torch.tanh(rho)  # [B, L, num_mixtures]
        
        # Pen state (will be thresholded during sampling)
        pen = torch.sigmoid(pen)  # [B, L, 1]
        
        return {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'rho': rho,
            'pen': pen
        }
    
    def sample(self, gmm_params, temperature=1.0):
        """
        Sample strokes from the GMM
        
        Args:
            gmm_params: Dictionary of GMM parameters
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            strokes: [B, L, 3] - (dx, dy, pen_state)
        """
        pi = gmm_params['pi']  # [B, L, num_mixtures]
        mu = gmm_params['mu']  # [B, L, num_mixtures, 2]
        sigma = gmm_params['sigma']  # [B, L, num_mixtures, 2]
        rho = gmm_params['rho']  # [B, L, num_mixtures]
        pen = gmm_params['pen']  # [B, L, 1]
        
        B, L, M = pi.shape
        
        # Apply temperature to mixture weights
        pi = pi / temperature
        pi = F.softmax(pi, dim=-1)
        
        # Sample mixture component for each point
        mixture_idx = torch.multinomial(pi.view(-1, M), 1).view(B, L)  # [B, L]
        
        # Gather parameters for selected mixture
        batch_idx = torch.arange(B).unsqueeze(1).expand(B, L)
        seq_idx = torch.arange(L).unsqueeze(0).expand(B, L)
        
        mu_selected = mu[batch_idx, seq_idx, mixture_idx]  # [B, L, 2]
        sigma_selected = sigma[batch_idx, seq_idx, mixture_idx]  # [B, L, 2]
        rho_selected = rho[batch_idx, seq_idx, mixture_idx]  # [B, L]
        
        # Sample from bivariate normal distribution
        # Using Cholesky decomposition for correlated sampling
        mean_x, mean_y = mu_selected[..., 0], mu_selected[..., 1]
        std_x, std_y = sigma_selected[..., 0], sigma_selected[..., 1]
        
        # Standard normal samples
        z = torch.randn_like(mu_selected)  # [B, L, 2]
        
        # Apply correlation and scaling
        x = mean_x + std_x * z[..., 0]
        y = mean_y + std_y * (rho_selected * z[..., 0] + 
                              torch.sqrt(1 - rho_selected**2) * z[..., 1])
        
        # Sample pen state
        pen_state = (pen.squeeze(-1) > 0.5).float()  # [B, L]
        
        # Combine into strokes
        strokes = torch.stack([x, y, pen_state], dim=-1)  # [B, L, 3]
        
        return strokes
    
    def compute_loss(self, gmm_params, target_strokes):
        """
        Compute negative log-likelihood loss
        
        Args:
            gmm_params: Dictionary of GMM parameters
            target_strokes: [B, L, 3] - ground truth (dx, dy, pen_state)
        
        Returns:
            loss: Scalar loss value
        """
        pi = gmm_params['pi']  # [B, L, M]
        mu = gmm_params['mu']  # [B, L, M, 2]
        sigma = gmm_params['sigma']  # [B, L, M, 2]
        rho = gmm_params['rho']  # [B, L, M]
        pen = gmm_params['pen']  # [B, L, 1]
        
        B, L, M = pi.shape
        
        target_xy = target_strokes[..., :2]  # [B, L, 2]
        target_pen = target_strokes[..., 2]  # [B, L]
        
        # Expand targets for mixture computation
        target_xy = target_xy.unsqueeze(2)  # [B, L, 1, 2]
        
        # Compute bivariate Gaussian PDF for each mixture
        mean_x, mean_y = mu[..., 0], mu[..., 1]  # [B, L, M]
        std_x, std_y = sigma[..., 0], sigma[..., 1]  # [B, L, M]
        
        target_x = target_xy[..., 0, 0].unsqueeze(-1)  # [B, L, 1]
        target_y = target_xy[..., 0, 1].unsqueeze(-1)  # [B, L, 1]
        
        # Normalized differences
        z_x = (target_x - mean_x) / (std_x + 1e-8)  # [B, L, M]
        z_y = (target_y - mean_y) / (std_y + 1e-8)  # [B, L, M]
        
        # Mahalanobis distance with correlation
        z = (z_x ** 2) + (z_y ** 2) - 2 * rho * z_x * z_y
        z = z / (1 - rho ** 2 + 1e-8)
        
        # Bivariate normal PDF
        norm = 1 / (2 * math.pi * std_x * std_y * torch.sqrt(1 - rho ** 2 + 1e-8))
        gaussian = norm * torch.exp(-z / 2)  # [B, L, M]
        
        # Mixture likelihood
        likelihood = torch.sum(pi * gaussian, dim=-1)  # [B, L]
        likelihood = torch.clamp(likelihood, min=1e-8)  # Avoid log(0)
        
        # Coordinate loss (negative log-likelihood)
        coord_loss = -torch.log(likelihood).mean()
        
        # Pen state loss (binary cross entropy)
        pen_loss = F.binary_cross_entropy(pen.squeeze(-1), target_pen)
        
        # Combined loss
        total_loss = coord_loss + pen_loss
        
        return total_loss, {'coord_loss': coord_loss.item(), 'pen_loss': pen_loss.item()}
