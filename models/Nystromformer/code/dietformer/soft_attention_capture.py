"""Monkey-patch capture for SOFTSelfAttention.

Replaces SOFTSelfAttention.forward with a version that captures all
intermediate values needed for the quantization witness pipeline.

Pattern follows nystrom_attention_capture.py and linformer_attention_capture.py.
"""
from __future__ import annotations

import math
import types

import numpy as np
import torch

from soft_attention import SOFTSelfAttention, newton_raphson_pseudoinverse


class SOFTCaptureRegistry:
    """layer_name -> {npz key -> numpy array} container."""

    def __init__(self):
        self._captures: dict[str, dict[str, np.ndarray]] = {}

    def reset(self):
        self._captures.clear()

    def get(self, layer_name: str) -> dict[str, np.ndarray]:
        return dict(self._captures.get(layer_name, {}))

    def all(self) -> dict[str, dict[str, np.ndarray]]:
        return {k: dict(v) for k, v in self._captures.items()}

    def __contains__(self, layer_name: str) -> bool:
        return layer_name in self._captures

    def __len__(self) -> int:
        return len(self._captures)


def _patched_soft_forward(self, hidden_states, attention_mask=None,
                          output_attentions=False, **kwargs):
    """SOFTSelfAttention.forward with full intermediate capture."""
    capture: dict[str, np.ndarray] = {}

    B, S, _ = hidden_states.shape
    H = self.num_attention_heads
    D = self.attention_head_size

    # Phase 0: projection inputs and weights
    capture["hidden_states"] = hidden_states.detach().cpu().numpy()
    capture["W_q"] = self.query.weight.data.cpu().numpy()
    capture["b_q"] = (self.query.bias.data.cpu().numpy()
                      if self.query.bias is not None else None)
    # W_k = W_q (shared), capture explicitly for witness clarity
    capture["W_v"] = self.value.weight.data.cpu().numpy()
    capture["b_v"] = (self.value.bias.data.cpu().numpy()
                      if self.value.bias is not None else None)
    capture["W_o"] = self._output_dense.weight.data.cpu().numpy()
    capture["b_o"] = (self._output_dense.bias.data.cpu().numpy()
                      if self._output_dense.bias is not None else None)

    # Q/K/V projections -> [B, H, S, D]
    q = self.query(hidden_states).view(B, S, H, D).transpose(1, 2)
    k = self.key(hidden_states).view(B, S, H, D).transpose(1, 2)
    v = self.value(hidden_states).view(B, S, H, D).transpose(1, 2)

    capture["V"] = v.detach().cpu().numpy()

    # Scaling
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
        mask_broad = mask.view(B, 1, S, 1)
        q_proc = q_proc * mask_broad
        k_proc = k_proc * mask_broad
        v = v * mask_broad

    capture["Q_proc"] = q_proc.detach().cpu().numpy()
    # K_proc = Q_proc (W_q = W_k), captured for explicit verification
    capture["K_proc"] = k_proc.detach().cpu().numpy()

    # Landmarks from masked Q_proc via segment averaging
    seg = S // self.num_landmarks
    q_landmarks = q_proc.reshape(
        B, H, self.num_landmarks, seg, D
    ).mean(dim=-2)
    capture["Q_landmarks"] = q_landmarks.detach().cpu().numpy()

    # Gaussian kernel distance: ||q_i - k~_j||^2 / (2*sqrt(D))
    denom = 2.0 * math.sqrt(D)

    # Squared norms
    norm_q = (q_proc ** 2).sum(dim=-1)             # [B,H,S]
    norm_qtilde = (q_landmarks ** 2).sum(dim=-1)    # [B,H,m]
    capture["norm_q"] = norm_q.detach().cpu().numpy()
    capture["norm_qtilde"] = norm_qtilde.detach().cpu().numpy()

    # Cross-terms
    cross_B = torch.matmul(q_proc, q_landmarks.transpose(-1, -2))  # [B,H,S,m]
    cross_A = torch.matmul(q_landmarks, q_landmarks.transpose(-1, -2))  # [B,H,m,m]
    capture["cross_B"] = cross_B.detach().cpu().numpy()
    capture["cross_A"] = cross_A.detach().cpu().numpy()

    # Distances (before dividing by denom)
    dist_B_raw = (norm_q.unsqueeze(-1) + norm_qtilde.unsqueeze(-2)
                  - 2 * cross_B)  # [B,H,S,m]
    dist_A_raw = (norm_qtilde.unsqueeze(-1) + norm_qtilde.unsqueeze(-2)
                  - 2 * cross_A)  # [B,H,m,m]
    capture["dist_B_raw"] = dist_B_raw.detach().cpu().numpy()
    capture["dist_A_raw"] = dist_A_raw.detach().cpu().numpy()

    # Kernel values
    kernel_B = torch.exp(-dist_B_raw / denom)
    kernel_A = torch.exp(-dist_A_raw / denom)
    capture["kernel_B"] = kernel_B.detach().cpu().numpy()
    capture["kernel_A"] = kernel_A.detach().cpu().numpy()

    # Newton-Raphson pseudoinverse (2nd order, SOFT Eq. 10)
    inv_A = newton_raphson_pseudoinverse(kernel_A, n_iter=self.n_iter)
    capture["inv_A"] = inv_A.detach().cpu().numpy()

    # Chain matmul
    temp = torch.matmul(kernel_B.transpose(-1, -2), v)   # [B,H,m,D]
    mid = torch.matmul(kernel_B, inv_A)                    # [B,H,S,m]
    Y = torch.matmul(mid, temp)                            # [B,H,S,D]
    capture["temp"] = temp.detach().cpu().numpy()
    capture["mid"] = mid.detach().cpu().numpy()
    capture["Y"] = Y.detach().cpu().numpy()

    # Reshape for O-projection
    context_layer = Y.permute(0, 2, 1, 3).contiguous()
    context_layer = context_layer.view(B, S, self.all_head_size)
    capture["attn_output_for_proj"] = context_layer.detach().cpu().numpy()

    # Register captures
    self._capture_registry._captures[self._capture_layer_name] = capture

    outputs = (context_layer, mid) if output_attentions else (context_layer,)
    return outputs


def apply_soft_monkey_patch(model) -> SOFTCaptureRegistry:
    """Replace SOFTSelfAttention.forward with capture version.

    Args:
        model: BertForMaskedLM with attention layers replaced by SOFTSelfAttention.

    Returns:
        SOFTCaptureRegistry for retrieving captured intermediates.
    """
    registry = SOFTCaptureRegistry()
    n_patched = 0

    for name, module in model.named_modules():
        if isinstance(module, SOFTSelfAttention):
            module._capture_registry = registry
            module._capture_layer_name = name

            # Find output projection dense layer (parent.attention.output.dense)
            parts = name.split('.')
            # Expected: bert.encoder.layer.{i}.attention.self
            layer_idx = int(parts[3])
            parent_attn = model.bert.encoder.layer[layer_idx].attention
            module._output_dense = parent_attn.output.dense

            module.forward = types.MethodType(_patched_soft_forward, module)
            n_patched += 1

    if n_patched == 0:
        raise RuntimeError(
            "No SOFTSelfAttention modules found. "
            "Replace attention layers before calling apply_soft_monkey_patch."
        )

    return registry
