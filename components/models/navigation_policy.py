import os

import torch
import torch.nn as nn

from components.models.feature_encoder import PositionalEncoding


class NavigationPolicy(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        use_transformer=False,
        value_head=False,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_len=256,
        device=None,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        self.hidden_dim = hidden_dim
        self.d_model = hidden_dim
        self.input_dim = input_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_transformer:
            self.input_proj = nn.Linear(input_dim, self.d_model)
            self.pos_encoder = PositionalEncoding(self.d_model, max_len=max_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
            )
            self.core = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.core_output_dim = self.d_model
        else:
            self.core = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.core_output_dim = hidden_dim

        self.shared = nn.Sequential(
            nn.Linear(self.core_output_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

        self.value_head = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            if value_head
            else None
        )

    def forward(self, seq, hidden=None, pad_mask=None):
        """
        seq: Tensor [B, T, D]
        hidden: (h, c) for LSTM, None for Transformer
        pad_mask: Optional [B, T] bool mask for Transformer (True=pad)
        """
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)

        if self.use_transformer:
            hidden = None
            seq = self.input_proj(seq)
            seq = self.pos_encoder(seq)
            # Transformer expects src_key_padding_mask (True = PAD)
            out = self.core(seq, src_key_padding_mask=pad_mask)
        else:
            out, hidden = self.core(seq, hidden)

        out = self.shared(out)
        logits = self.policy_head(out)
        value = self.value_head(out).squeeze(-1) if self.value_head is not None else None
        return logits, value, hidden

    def save_model(self, path):
        """
        Saves the model parameters and config.
        navigation_policy_{input_dim}_{hidden_dim}_{output_dim}_{use_transformer}.pth
        Example: navigation_policy_256_256_10_True.pth for input_dim=256, hidden_dim=256, output_dim=10
        """
        input_dim = self.input_dim
        hidden_dim = self.hidden_dim
        output_dim = self.policy_head[-1].out_features
        filename = f"navigation_policy_{input_dim}_{hidden_dim}_{output_dim}_{self.use_transformer}.pth"
        torch.save(self.state_dict(), os.path.join(path, filename))

    def load_weights(self, model_path, device="cpu"):
        """
        Loads model weights into an existing NavigationPolicy instance.
        """
        state_dict = torch.load(model_path, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)
