"""
Monkey-patch capture for LinformerSelfAttention.

Follows nystrom_attention_capture.py pattern:
  1. LinformerCaptureRegistry stores per-layer captures.
  2. _patched_forward replaces LinformerSelfAttention.forward with a version
     that captures all intermediates for witness generation.
  3. apply_linformer_monkey_patch patches all LinformerSelfAttention modules.

Capture target list matches dev plan A-2.
"""
from __future__ import annotations

import math
import types
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from linformer_attention import LinformerSelfAttention


class LinformerCaptureRegistry:
    """layer_name -> {npz_key -> numpy array} container."""

    def __init__(self) -> None:
        self._captures: dict[str, dict[str, np.ndarray]] = {}

    def reset(self) -> None:
        self._captures.clear()

    def get(self, layer_name: str) -> dict[str, np.ndarray]:
        return dict(self._captures.get(layer_name, {}))

    def all(self) -> dict[str, dict[str, np.ndarray]]:
        return {k: dict(v) for k, v in self._captures.items()}

    def __contains__(self, layer_name: str) -> bool:
        return layer_name in self._captures

    def __len__(self) -> int:
        return len(self._captures)


def _patched_forward(self, hidden_states, attention_mask=None,
                     output_attentions=False, **kwargs):
    """Fork of LinformerSelfAttention.forward with intermediate capture.

    Numerically identical to the clean forward. All intermediate values
    needed by the quantization pipeline are captured into the registry.
    """
    capture: dict[str, np.ndarray] = {}

    B, S, _ = hidden_states.shape
    H = self.num_attention_heads
    D = self.attention_head_size

    # -- Phase 0: Projection inputs/weights capture --
    capture["hidden_states"] = hidden_states.detach().cpu().numpy()
    capture["W_q"] = self.query.weight.data.cpu().numpy()
    capture["W_k"] = self.key.weight.data.cpu().numpy()
    capture["W_v"] = self.value.weight.data.cpu().numpy()
    capture["b_q"] = (self.query.bias.data.cpu().numpy()
                      if self.query.bias is not None else None)
    capture["b_k"] = (self.key.bias.data.cpu().numpy()
                      if self.key.bias is not None else None)
    capture["b_v"] = (self.value.bias.data.cpu().numpy()
                      if self.value.bias is not None else None)
    capture["W_o"] = self._output_dense.weight.data.cpu().numpy()
    capture["b_o"] = (self._output_dense.bias.data.cpu().numpy()
                      if self._output_dense.bias is not None else None)

    # Linformer projection matrices
    capture["E"] = self.E.data.cpu().numpy()  # [H, k, n]
    capture["F_proj"] = self.F.data.cpu().numpy()  # [H, k, n]

    # Q/K/V projections -> [B, H, S, D]
    q = (self.query(hidden_states)
         .view(B, S, H, D).transpose(1, 2))
    k = (self.key(hidden_states)
         .view(B, S, H, D).transpose(1, 2))
    v = (self.value(hidden_states)
         .view(B, S, H, D).transpose(1, 2))

    capture["V"] = v.detach().cpu().numpy()

    # Scaling
    scale = 1.0 / math.sqrt(math.sqrt(D))
    q_proc = q * scale
    k_proc = k * scale

    # Pre-mask Q_proc and K_proc
    if attention_mask is not None:
        # BERT passes extended mask [B,1,1,S] with -inf for pad.
        # Convert back to binary [B,S]: valid=1, pad=0.
        if attention_mask.dim() == 4:
            bin_mask = (attention_mask.view(B, -1)[:, -S:] >= 0).long()
        elif attention_mask.dim() == 2:
            bin_mask = attention_mask.long()
        else:
            bin_mask = attention_mask.view(B, S).long()
        mask_np = bin_mask.detach().cpu().numpy().astype(np.int64)
        mask_t = bin_mask.float().view(B, 1, S, 1)
        q_proc = q_proc * mask_t
        k_proc = k_proc * mask_t
        v = v * mask_t
    else:
        mask_np = np.ones((B, S), dtype=np.int64)

    capture["Q_proc"] = q_proc.detach().cpu().numpy()
    capture["K_proc"] = k_proc.detach().cpu().numpy()
    capture["mask"] = mask_np

    # Linformer projections
    ek = torch.matmul(self.E.unsqueeze(0), k_proc)  # [B, H, k, D]
    fv = torch.matmul(self.F.unsqueeze(0), v)         # [B, H, k, D]
    capture["EK"] = ek.detach().cpu().numpy()
    capture["FV"] = fv.detach().cpu().numpy()

    # Attention logit: Q_proc @ EK^T -> [B, H, S, k]
    logit = torch.matmul(q_proc, ek.transpose(-1, -2))
    capture["logit"] = logit.detach().cpu().numpy()

    # Softmax (no mask)
    p = torch.nn.functional.softmax(logit, dim=-1)
    capture["P"] = p.detach().cpu().numpy()

    # Output: P @ FV -> [B, H, S, D]
    x = torch.matmul(p, fv)
    capture["X"] = x.detach().cpu().numpy()

    # Reshape -> [B, S, HD]
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(B, S, self.all_head_size)
    capture["attn_output_for_proj"] = x.detach().cpu().numpy()

    # Store in registry
    self._capture_registry._captures[self._capture_layer_name] = capture

    outputs = (x, p) if output_attentions else (x,)
    return outputs


def apply_linformer_monkey_patch(
    model: nn.Module,
) -> LinformerCaptureRegistry:
    """Patch all LinformerSelfAttention modules with capture-enabled forward.

    Args:
        model: BertForMaskedLM with LinformerSelfAttention modules.

    Returns:
        LinformerCaptureRegistry for post-forward capture retrieval.
    """
    registry = LinformerCaptureRegistry()
    n_patched = 0

    layers = model.bert.encoder.layer

    for name, module in model.named_modules():
        if isinstance(module, LinformerSelfAttention):
            module._capture_registry = registry
            module._capture_layer_name = name

            # Output projection: parent.output.dense
            layer_idx = int(name.split('.')[3])
            parent_attn = layers[layer_idx].attention
            module._output_dense = parent_attn.output.dense

            module.forward = types.MethodType(_patched_forward, module)
            n_patched += 1

    expected = len(layers)
    if n_patched != expected:
        raise RuntimeError(
            f"monkey-patched {n_patched} layers, expected {expected}."
        )

    return registry
