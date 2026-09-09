"""
LinformerSelfAttention: BERT-compatible self-attention with learned projection
matrices E_i, F_i (Linformer, Wang et al. 2020).

Replaces standard O(n^2) attention with O(nk) by projecting K and V from n to
k dimensions using per-head learned projection matrices.

Design: k = 32, no parameter sharing (independent E_i, F_i per head).

Masking convention (docs/linformer_dev_plan.md):
  - Q_proc and K_proc are pre-masked by attention_mask * 1/sqrt(sqrt(D)).
  - V is pre-masked (unlike BERT, because F@V mixes all positions
    without softmax gating, so padding V would contaminate the output).
  - EK = E_i @ K_proc: padding zeroed through K_proc pre-masking.
  - FV = F_i @ V: padding zeroed through V pre-masking.
  - Softmax: no mask parameter. K_proc pre-masking handles padding.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class LinformerSelfAttention(nn.Module):
    """Drop-in replacement for BertSelfAttention with Linformer projection."""

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 seq_len: int, proj_dim: int = 32):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size
        self.seq_len = seq_len
        self.proj_dim = proj_dim  # k

        # Standard Q/K/V projections (same as BertSelfAttention)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Linformer projection matrices: E_i [H, k, n], F_i [H, k, n]
        self.E = nn.Parameter(torch.randn(
            num_attention_heads, proj_dim, seq_len,
        ))
        self.F = nn.Parameter(torch.randn(
            num_attention_heads, proj_dim, seq_len,
        ))

    def forward(self, hidden_states, attention_mask=None,
                output_attentions=False, **kwargs):
        B, S, _ = hidden_states.shape
        H = self.num_attention_heads
        D = self.attention_head_size

        # Q/K/V linear projections -> [B, H, S, D]
        q = self.query(hidden_states).view(B, S, H, D).transpose(1, 2)
        k = self.key(hidden_states).view(B, S, H, D).transpose(1, 2)
        v = self.value(hidden_states).view(B, S, H, D).transpose(1, 2)

        # Scaling: split 1/sqrt(D) into two factors on Q and K
        scale = 1.0 / math.sqrt(math.sqrt(D))
        q_proc = q * scale
        k_proc = k * scale

        # Pre-mask Q_proc and K_proc (multiplicative)
        if attention_mask is not None:
            # BERT passes extended mask [B,1,1,S]; also accept [B,S]
            if attention_mask.dim() == 4:
                mask = (attention_mask.view(B, -1)[:, -S:] >= 0).float()
            elif attention_mask.dim() == 2:
                mask = attention_mask.float()
            else:
                mask = attention_mask.view(B, S).float()
            mask = mask.view(B, 1, S, 1)  # [B, 1, S, 1]
            q_proc = q_proc * mask
            k_proc = k_proc * mask
            v = v * mask

        # Linformer projections: E_i [H, k, S] @ K_proc [B, H, S, D]
        ek = torch.matmul(
            self.E.unsqueeze(0), k_proc,
        )  # [1, H, k, S] @ [B, H, S, D] -> [B, H, k, D]
        fv = torch.matmul(
            self.F.unsqueeze(0), v,
        )  # [1, H, k, S] @ [B, H, S, D] -> [B, H, k, D]

        # Attention logit: Q_proc [B,H,S,D] @ EK^T [B,H,D,k] -> [B,H,S,k]
        logit = torch.matmul(q_proc, ek.transpose(-1, -2))

        # Softmax (no mask: K_proc pre-masking handles padding)
        p = torch.nn.functional.softmax(logit, dim=-1)

        # Output: P [B,H,S,k] @ FV [B,H,k,D] -> [B,H,S,D]
        x = torch.matmul(p, fv)

        # Reshape [B, H, S, D] -> [B, S, HD]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, S, self.all_head_size)

        outputs = (x, p) if output_attentions else (x,)
        return outputs
