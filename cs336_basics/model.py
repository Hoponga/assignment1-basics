import torch
import torch.nn as nn
import math


class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        sd = math.sqrt(2.0/(in_features + out_features))
        weight_tensor = nn.init.trunc_normal_(torch.empty(out_features, in_features, dtype=dtype), std=sd*sd, a=-3*sd, b=3*sd)
        self.weight = nn.Parameter(weight_tensor)
        if device:
            self.device = device
            self.weight = self.weight.to(device)

    # x is of shape [..., in_features]
    def forward(self, x):
        return x @ self.weight.T


# embed and unembed table for llm logits
class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.vocab_size = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        embedding_weights = nn.init.trunc_normal_(torch.empty(self.vocab_size, self.embedding_dim, dtype=dtype), a=-3, b=3)
        self.embedding_table = nn.Parameter(embedding_weights)

    def forward(self, token_ids: torch.Tensor):
        return self.embedding_table[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype))
        if device:
            self.weight = self.weight.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention.
    Q: [..., queries, d_k]
    K: [..., keys, d_k]
    V: [..., values, d_v]
    mask: [..., queries, keys] bool tensor (True = keep, False = mask out)
    Returns: [..., queries, d_v]
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.
    output = (silu(x @ W1.T) * (x @ W3.T)) @ W2.T
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


def apply_rope(
    x: torch.Tensor,
    token_positions: torch.Tensor,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    Apply Rotary Position Embeddings (RoPE).
    x: [..., seq_len, d_k]
    token_positions: [..., seq_len]  (broadcasts with x's leading dims)
    Returns: [..., seq_len, d_k]
    """
    d_k = x.shape[-1]
    assert d_k % 2 == 0, "d_k must be even for RoPE"

    # Frequencies: theta^(-2i/d_k) for i in [0, d_k/2)
    i = torch.arange(d_k // 2, device=x.device, dtype=torch.float32)
    freqs = theta ** (-2.0 * i / d_k)  # (d_k/2,)

    # Angles: [..., seq_len, d_k/2]
    angles = token_positions.float().unsqueeze(-1) * freqs

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    x_even = x[..., 0::2]  # [..., seq_len, d_k/2]
    x_odd  = x[..., 1::2]  # [..., seq_len, d_k/2]

    out_even = x_even * cos_a - x_odd * sin_a
    out_odd  = x_even * sin_a + x_odd * cos_a

    # Interleave back: [..., seq_len, d_k]
    out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
    return out.to(x.dtype)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention without RoPE. Applies causal mask by default."""

    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [..., seq_len, d_model]
        *batch, seq_len, d_model = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to [..., num_heads, seq_len, d_k]
        Q = Q.view(*batch, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch, seq_len, self.num_heads, self.d_k).transpose(-3, -2)

        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Merge heads: [..., seq_len, d_model]
        attn_out = attn_out.transpose(-3, -2).contiguous().view(*batch, seq_len, d_model)
        return self.output_proj(attn_out)


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """Multi-head self-attention with RoPE applied to Q and K. Applies causal mask."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [..., seq_len, d_model]
        *batch, seq_len, d_model = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to [..., num_heads, seq_len, d_k]
        Q = Q.view(*batch, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch, seq_len, self.num_heads, self.d_k).transpose(-3, -2)

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        # Broadcast token_positions over the num_heads dim if it has a batch dim
        if token_positions.dim() > 1:
            rope_positions = token_positions.unsqueeze(-2)  # [..., 1, seq_len]
        else:
            rope_positions = token_positions  # (seq_len,) broadcasts over all leading dims

        Q = apply_rope(Q, rope_positions, theta=self.theta)
        K = apply_rope(K, rope_positions, theta=self.theta)

        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Merge heads: [..., seq_len, d_model]
        attn_out = attn_out.transpose(-3, -2).contiguous().view(*batch, seq_len, d_model)
        return self.output_proj(attn_out)


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block:
      x = x + attn(ln1(x))
      x = x + ffn(ln2(x))
    Uses RoPE in self-attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer language model.
    Input:  token indices [batch, seq_len]
    Output: logits        [batch, seq_len, vocab_size]

    State dict keys:
      token_embeddings.weight   (vocab_size, d_model)
      layers.{i}.ln1.weight
      layers.{i}.attn.{q,k,v}_proj.weight
      layers.{i}.attn.output_proj.weight
      layers.{i}.ln2.weight
      layers.{i}.ffn.w{1,2,3}.weight
      ln_final.weight
      lm_head.weight
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length

        # nn.Embedding gives state dict key `token_embeddings.weight`
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        if device:
            self.token_embeddings = self.token_embeddings.to(device)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        # in_indices: [batch, seq_len]
        seq_len = in_indices.shape[-1]
        x = self.token_embeddings(in_indices)  # [batch, seq_len, d_model]

        token_positions = torch.arange(seq_len, device=in_indices.device)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)
        return self.lm_head(x)  # [batch, seq_len, vocab_size]




def init_model_from_config(config):
    model_cfg = config['model']
    data_cfg = config['data']
    device = config['training']['device']

    model = TransformerLM(
        vocab_size=int(model_cfg['vocab_size']),
        context_length=int(data_cfg['context_length']),
        d_model=int(model_cfg['d_model']),
        num_layers=int(model_cfg['num_layers']),
        num_heads=int(model_cfg['num_heads']),
        d_ff=int(model_cfg['d_ff']),
        rope_theta=float(model_cfg.get('rope_theta', 10000.0)),
        device=str(device),
    )

    return model

