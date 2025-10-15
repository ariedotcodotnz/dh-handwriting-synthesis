"""
Gaussian Mixture Model Decoder
Generates handwriting strokes as sequences of 2D points with pen states
ENHANCED VERSION: Stroke history attention, FiLM layers, nucleus sampling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation - allows style to adaptively modulate features"""
    def __init__(self, feature_dim, style_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(style_dim, feature_dim)
        self.beta_fc = nn.Linear(style_dim, feature_dim)

    def forward(self, x, style):
        gamma = self.gamma_fc(style)
        beta = self.beta_fc(style)
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * x + beta


class StrokeHistoryEncoder(nn.Module):
    """Encodes recently generated strokes for context-aware generation"""
    def __init__(self, stroke_dim=3, hidden_dim=128, context_len=50):
        super().__init__()
        self.context_len = context_len
        self.conv1 = nn.Conv1d(stroke_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, stroke_history):
        """Args: stroke_history [B, context_len, 3] Returns: [B, hidden_dim]"""
        if stroke_history.size(1) == 0:
            return torch.zeros(stroke_history.size(0), 128, device=stroke_history.device)
        x = stroke_history.transpose(1, 2)  # [B, 3, context_len]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.pool(x).squeeze(-1)  # [B, hidden_dim]


class GMMDecoder(nn.Module):
    """
    Enhanced GMM decoder with:
    - 30 mixture components (vs 20 original)
    - Stroke history attention
    - FiLM style modulation
    - Nucleus sampling support

    Args:
        d_model: Input feature dimension
        num_mixtures: Number of Gaussian components (default: 30)
        hidden_dim: Hidden dimension
        use_history: Enable stroke history (default: True)
        style_dim: Style vector dimension for FiLM
    """
    def __init__(self, d_model=512, num_mixtures=30, hidden_dim=512,
                 use_history=True, style_dim=256):
        super().__init__()
        self.d_model = d_model
        self.num_mixtures = num_mixtures
        self.use_history = use_history

        # Stroke history encoder
        history_dim = 0
        if use_history:
            self.history_encoder = StrokeHistoryEncoder(stroke_dim=3, hidden_dim=128, context_len=50)
            history_dim = 128

        # Input projection
        self.input_proj = nn.Linear(d_model + history_dim, hidden_dim)

        # FiLM layers for style modulation
        self.film1 = FiLMLayer(hidden_dim, style_dim)
        self.film2 = FiLMLayer(hidden_dim, style_dim)

        # Hidden layers with residual connections
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # MDN output: mu_x, mu_y, sigma_x, sigma_y, rho, pi for each mixture + pen
        self.output_size = num_mixtures * 6 + 1
        self.mdn_layer = nn.Linear(hidden_dim, self.output_size)

        # Careful initialization
        nn.init.normal_(self.mdn_layer.weight, 0, 0.001)
        nn.init.constant_(self.mdn_layer.bias, 0)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features, stroke_history=None, style_vector=None):
        """
        Args:
            features: [B, L, d_model] - content+style features
            stroke_history: [B, L, context_len, 3] - recent strokes (optional)
            style_vector: [B, style_dim] - style for FiLM (optional)
        Returns:
            gmm_params: Dict with pi, mu, sigma, rho, pen
        """
        B, L, _ = features.shape

        # Encode stroke history if available
        if self.use_history and stroke_history is not None:
            history_flat = stroke_history.view(B * L, -1, 3)
            history_context = self.history_encoder(history_flat)  # [B*L, 128]
            history_context = history_context.view(B, L, -1)  # [B, L, 128]
            h = torch.cat([features, history_context], dim=-1)
        else:
            # No history available - pad with zeros if history encoder exists
            if self.use_history:
                # Create zero history context to match expected input size
                history_context = torch.zeros(B, L, 128, device=features.device, dtype=features.dtype)
                h = torch.cat([features, history_context], dim=-1)
            else:
                h = features

        # Project to hidden dim
        h = F.relu(self.input_proj(h))  # [B, L, hidden_dim]
        h = self.dropout(h)

        # First layer with FiLM modulation
        if style_vector is not None:
            h = self.film1(h, style_vector)
        h_res = h
        h = F.relu(self.fc1(h))
        h = self.layer_norm1(h + h_res)  # Residual
        h = self.dropout(h)

        # Second layer with FiLM modulation
        if style_vector is not None:
            h = self.film2(h, style_vector)
        h_res = h
        h = F.relu(self.fc2(h))
        h = self.layer_norm2(h + h_res)  # Residual
        h = self.dropout(h)

        # Generate MDN parameters
        out = self.mdn_layer(h)  # [B, L, output_size]

        # Split into components
        idx = 0
        pi = out[..., idx:idx+self.num_mixtures]
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
        pen = out[..., idx:]

        # Apply activations
        pi = F.softmax(pi, dim=-1)  # [B, L, M]
        mu = torch.stack([mu_x, mu_y], dim=-1)  # [B, L, M, 2]
        sigma_x = torch.exp(sigma_x) + 1e-4
        sigma_y = torch.exp(sigma_y) + 1e-4
        sigma = torch.stack([sigma_x, sigma_y], dim=-1)  # [B, L, M, 2]
        rho = torch.tanh(rho)  # [B, L, M]
        pen = torch.sigmoid(pen)  # [B, L, 1]

        return {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'rho': rho,
            'pen': pen
        }

    def sample(self, gmm_params, temperature=1.0, top_p=0.95):
        """
        Sample strokes with nucleus sampling for better quality

        Args:
            gmm_params: GMM parameters dict
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold (0.95 = keep top 95% probability mass)
        Returns:
            strokes: [B, L, 3] (dx, dy, pen_state)
        """
        pi = gmm_params['pi']  # [B, L, M]
        mu = gmm_params['mu']  # [B, L, M, 2]
        sigma = gmm_params['sigma']  # [B, L, M, 2]
        rho = gmm_params['rho']  # [B, L, M]
        pen = gmm_params['pen']  # [B, L, 1]

        B, L, M = pi.shape

        # Apply temperature to mixture weights
        pi = pi / temperature
        pi = F.softmax(pi, dim=-1)

        # Nucleus sampling on mixture weights
        if top_p < 1.0:
            sorted_pi, sorted_indices = torch.sort(pi, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_pi, dim=-1)

            # Remove mixtures below threshold
            sorted_pi_trimmed = sorted_pi.clone()
            sorted_pi_trimmed[cumulative_probs > top_p] = 0

            # Renormalize
            sorted_pi_trimmed = sorted_pi_trimmed / (sorted_pi_trimmed.sum(dim=-1, keepdim=True) + 1e-8)

            # Scatter back to original order
            pi = torch.zeros_like(pi)
            pi.scatter_(-1, sorted_indices, sorted_pi_trimmed)

        # Sample mixture component
        mixture_idx = torch.multinomial(pi.view(-1, M), 1).view(B, L)

        # Gather selected mixture parameters
        batch_idx = torch.arange(B, device=pi.device).unsqueeze(1).expand(B, L)
        seq_idx = torch.arange(L, device=pi.device).unsqueeze(0).expand(B, L)

        mu_selected = mu[batch_idx, seq_idx, mixture_idx]  # [B, L, 2]
        sigma_selected = sigma[batch_idx, seq_idx, mixture_idx]  # [B, L, 2]
        rho_selected = rho[batch_idx, seq_idx, mixture_idx]  # [B, L]

        # Sample from bivariate Gaussian
        mean_x, mean_y = mu_selected[..., 0], mu_selected[..., 1]
        std_x, std_y = sigma_selected[..., 0], sigma_selected[..., 1]

        # Apply temperature to std devs
        std_x = std_x * math.sqrt(temperature)
        std_y = std_y * math.sqrt(temperature)

        # Correlated sampling
        z = torch.randn_like(mu_selected)
        x = mean_x + std_x * z[..., 0]
        y = mean_y + std_y * (rho_selected * z[..., 0] +
                               torch.sqrt(1 - rho_selected**2 + 1e-8) * z[..., 1])

        # Sample pen state with temperature
        pen_probs = pen.squeeze(-1)  # [B, L]
        pen_probs = torch.clamp(pen_probs / temperature, 0, 1)
        pen_state = (pen_probs > 0.5).float()

        strokes = torch.stack([x, y, pen_state], dim=-1)
        return strokes

    def compute_loss(self, gmm_params, target_strokes, mask=None):
        """
        Compute negative log-likelihood loss

        Args:
            gmm_params: GMM parameters
            target_strokes: [B, L, 3] ground truth
            mask: [B, L] validity mask
        Returns:
            loss, loss_dict
        """
        pi = gmm_params['pi']
        mu = gmm_params['mu']
        sigma = gmm_params['sigma']
        rho = gmm_params['rho']
        pen = gmm_params['pen']

        B, L, M = pi.shape
        target_xy = target_strokes[..., :2]
        target_pen = target_strokes[..., 2]

        # Expand targets
        target_xy = target_xy.unsqueeze(2)  # [B, L, 1, 2]
        mean_x, mean_y = mu[..., 0], mu[..., 1]
        std_x, std_y = sigma[..., 0], sigma[..., 1]

        target_x = target_xy[..., 0, 0].unsqueeze(-1)
        target_y = target_xy[..., 0, 1].unsqueeze(-1)

        # Normalized differences
        z_x = (target_x - mean_x) / (std_x + 1e-8)
        z_y = (target_y - mean_y) / (std_y + 1e-8)

        # Mahalanobis distance
        z = (z_x ** 2) + (z_y ** 2) - 2 * rho * z_x * z_y
        z = z / (1 - rho ** 2 + 1e-8)

        # Bivariate Gaussian PDF
        norm = 1 / (2 * math.pi * std_x * std_y * torch.sqrt(1 - rho ** 2 + 1e-8))
        gaussian = norm * torch.exp(-z / 2)

        # Mixture likelihood
        likelihood = torch.sum(pi * gaussian, dim=-1)
        likelihood = torch.clamp(likelihood, min=1e-10)
        coord_loss_per_pos = -torch.log(likelihood)
        coord_loss_per_pos = torch.clamp(coord_loss_per_pos, max=10.0)

        # Pen loss
        pen_loss_per_pos = F.binary_cross_entropy(pen.squeeze(-1), target_pen, reduction='none')

        # Apply mask
        if mask is not None:
            valid_positions = mask.sum()
            if valid_positions > 0:
                coord_loss = (coord_loss_per_pos * mask).sum() / valid_positions
                pen_loss = (pen_loss_per_pos * mask).sum() / valid_positions
            else:
                coord_loss = coord_loss_per_pos.mean()
                pen_loss = pen_loss_per_pos.mean()
        else:
            coord_loss = coord_loss_per_pos.mean()
            pen_loss = pen_loss_per_pos.mean()

        total_loss = coord_loss + pen_loss

        return total_loss, {
            'coord_loss': coord_loss.item(),
            'pen_loss': pen_loss.item(),
            'total_loss': total_loss.item()
        }