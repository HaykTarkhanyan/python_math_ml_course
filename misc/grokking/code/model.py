"""The smallest transformer needed for modular-addition grokking."""

import torch
from torch import nn


class ModularAdditionTransformer(nn.Module):
    """Predict x + y mod p from the token sequence [x, y, equals]."""

    def __init__(
        self, modulus: int, d_model: int = 128, n_heads: int = 4,
        d_mlp: int = 512, use_layer_norm: bool = True,
    ):
        super().__init__()
        self.modulus = modulus
        # Tokens 0..p-1 are numbers; token p is the '=' placeholder.
        self.equals_token = modulus
        self.token_embedding = nn.Embedding(modulus + 1, d_model)
        self.position_embedding = nn.Embedding(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_mlp,
            dropout=0.0, activation="relu", batch_first=True, norm_first=True,
        )
        # Nanda et al.'s mainline model has no LayerNorm.  Replacing these
        # modules avoids both normalization and their trainable parameters.
        if not use_layer_norm:
            layer.norm1 = nn.Identity()
            layer.norm2 = nn.Identity()
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.unembed = nn.Linear(d_model, modulus, bias=False)

    def forward(self, tokens: torch.Tensor, ablate: str | None = None) -> torch.Tensor:
        """Return answer logits only, with shape [batch, modulus]."""
        logits, _ = self.forward_with_cache(tokens, ablate=ablate)
        return logits

    def forward_with_cache(
        self, tokens: torch.Tensor, ablate: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the layer explicitly, retaining its attention/MLP residual streams.

        ``ablate`` may be ``"attention"`` or ``"mlp"``; this zeroes the
        corresponding residual update while keeping the remainder intact.
        """
        if ablate not in (None, "attention", "mlp"):
            raise ValueError("ablate must be None, 'attention', or 'mlp'")
        positions = torch.arange(3, device=tokens.device)
        token_residual = self.token_embedding(tokens) + self.position_embedding(positions)
        layer = self.transformer.layers[0]
        attention_input = layer.norm1(token_residual)
        attention_output, attention_patterns = layer.self_attn(
            attention_input, attention_input, attention_input,
            need_weights=True, average_attn_weights=False,
        )
        post_attention = token_residual if ablate == "attention" else token_residual + layer.dropout1(attention_output)
        mlp_input = layer.norm2(post_attention)
        mlp_hidden = layer.activation(layer.linear1(mlp_input))
        mlp_output = layer.linear2(layer.dropout(mlp_hidden))
        post_layer = post_attention if ablate == "mlp" else post_attention + layer.dropout2(mlp_output)
        logits = self.unembed(post_layer[:, -1])
        return logits, {
            "token_residual": token_residual,
            "attention_patterns": attention_patterns,
            "attention_output": attention_output,
            "post_attention": post_attention,
            "mlp_hidden": mlp_hidden,
            "mlp_output": mlp_output,
            "post_layer": post_layer,
        }
