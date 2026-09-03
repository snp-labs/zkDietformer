"""SOFTSelfAttention: BERT-compatible self-attention with SOFT kernel.

SOFT (Lu et al., NeurIPS 2021): Softmax-free Transformer via Gaussian kernel.
No row-sum normalization. W_q = W_k weight sharing enforced.
Newton-Raphson 2nd order pseudoinverse (Eq. 10), NOT HuggingFace 4th order.

Masking convention (docs/soft_dev_plan.md):
  - Q_proc (= K_proc): pre-masked by attention_mask * 1/sqrt(sqrt(D)).
    Padding positions become 0 in float domain (z_q in quantized domain).
    K_proc = Q_proc so separate K masking is unnecessary.
  - V: pre-masked. Padding positions become 0 in float domain.
    In chain matmul, (V - z_v) = 0 at padding, so no contamination.
  - Landmarks are generated from masked Q_proc via segment averaging.
    Padding positions (zeros) pull down segment means. This is a modeling
    choice: reference model and witness pipeline must handle it identically.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def newton_raphson_pseudoinverse(A, n_iter=20):
    """SOFT Eq. 10: 2nd order Newton-Raphson for Moore-Penrose pseudoinverse.

    A_{k+1} = 2 A_k - A_k A A_k
    Initialization: A_0 = alpha * A, alpha = 2 / ||A||_1^2

    Args:
        A: (B, H, m, m) symmetric PSD Gaussian kernel matrix.
        n_iter: number of iterations (default 20, from SOFT Figure 4).

    Returns:
        A_k: (B, H, m, m) approximate pseudoinverse.
    """
    norm_A_1 = torch.norm(A, p=1, dim=(-2, -1), keepdim=True)
    alpha = 2.0 / (norm_A_1 ** 2 + 1e-12)
    A_k = alpha * A
    for _ in range(n_iter):
        A_k = 2 * A_k - A_k @ A @ A_k
    return A_k


class SOFTSelfAttention(nn.Module):
    """Drop-in replacement for BertSelfAttention with SOFT Gaussian kernel."""

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 seq_len: int, num_landmarks: int = 32, n_iter: int = 20):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})."
            )
        if seq_len % num_landmarks != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by "
                f"num_landmarks ({num_landmarks})."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size
        self.seq_len = seq_len
        self.num_landmarks = num_landmarks
        self.n_iter = n_iter

        # W_q = W_k: same nn.Linear object (shared reference)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = self.query  # weight sharing enforced
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None,
                output_attentions=False, **kwargs):
        B, S, _ = hidden_states.shape
        H = self.num_attention_heads
        D = self.attention_head_size

        # Q/K/V projections -> [B, H, S, D]
        q = self.query(hidden_states).view(B, S, H, D).transpose(1, 2)
        k = self.key(hidden_states).view(B, S, H, D).transpose(1, 2)
        v = self.value(hidden_states).view(B, S, H, D).transpose(1, 2)

        # Scaling: 1/sqrt(sqrt(D)) on both Q and K
        scale = 1.0 / math.sqrt(math.sqrt(D))
        q_proc = q * scale
        k_proc = k * scale

        # Pre-mask Q_proc, K_proc, V
        if attention_mask is not None:
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

        # Landmarks: segment averaging of Q_proc (= K_proc since W_q = W_k)
        seg = S // self.num_landmarks
        q_landmarks = q_proc.reshape(B, H, self.num_landmarks, seg, D).mean(dim=-2)

        # Gaussian kernel: exp(-||q_i - k~_j||^2 / (2 * sqrt(D)))
        denom = 2.0 * math.sqrt(D)

        # B kernel: Q_proc [B,H,S,D] vs Q~ [B,H,m,D]
        norm_q = (q_proc ** 2).sum(dim=-1, keepdim=True)          # [B,H,S,1]
        norm_qtilde = (q_landmarks ** 2).sum(dim=-1, keepdim=True)  # [B,H,m,1]
        cross_B = torch.matmul(q_proc, q_landmarks.transpose(-1, -2))  # [B,H,S,m]
        dist_B = norm_q + norm_qtilde.transpose(-1, -2) - 2 * cross_B  # [B,H,S,m]
        kernel_B = torch.exp(-dist_B / denom)

        # A kernel: Q~ [B,H,m,D] vs Q~ [B,H,m,D]
        cross_A = torch.matmul(q_landmarks, q_landmarks.transpose(-1, -2))  # [B,H,m,m]
        dist_A = norm_qtilde + norm_qtilde.transpose(-1, -2) - 2 * cross_A  # [B,H,m,m]
        kernel_A = torch.exp(-dist_A / denom)

        # Pseudoinverse: Newton-Raphson 2nd order (SOFT Eq. 10)
        inv_A = newton_raphson_pseudoinverse(kernel_A, n_iter=self.n_iter)

        # Chain matmul: Y = B @ A^{+} @ B^T @ V
        temp = torch.matmul(kernel_B.transpose(-1, -2), v)   # [B,H,m,D]
        mid = torch.matmul(kernel_B, inv_A)                    # [B,H,S,m]
        Y = torch.matmul(mid, temp)                            # [B,H,S,D]

        # Reshape [B, H, S, D] -> [B, S, HD]
        Y = Y.permute(0, 2, 1, 3).contiguous()
        Y = Y.view(B, S, self.all_head_size)

        outputs = (Y, mid) if output_attentions else (Y,)
        return outputs
