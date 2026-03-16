import torch
import torch.nn as nn
import math
import numpy as np

class BertStandardAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.seq_len = config["seq_len"]
        
        # BERT는 Landmark를 사용하지 않지만, 인터페이스 통일을 위해 정의
        self.num_landmarks = self.seq_len 

    def forward(self, Q, K, V, mask, collect_intermediates: bool = False, save_path: str = None):
        """
        Q, K, V: (B, H, S, D) 형태의 입력
        mask: (B, S) 또는 (B, S, 1) 형태의 마스크
        """
        debug = {}

        # 1. 마스크 정규화 (NystromAttention과 동일 로직)
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask2 = mask.squeeze(-1)
        elif mask.dim() == 2:
            mask2 = mask
        else:
            raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}")

        mask2 = mask2.to(Q.dtype).to(Q.device)

        # 2. 스케일링 적용 (Nystromformer와 동일한 sqrt(sqrt(d)) 또는 표준 sqrt(d) 선택 가능)
        # 여기서는 비교의 공정성을 위해 Nystrom 코드의 scale 방식을 따릅니다.
        scale = 1.0 / math.sqrt(math.sqrt(self.head_dim))
        
        Q_proc = Q * mask2[:, None, :, None] * scale
        K_proc = K * mask2[:, None, :, None] * scale

        # 중간값 저장 시작
        if collect_intermediates:
            debug["Q_proc"] = Q_proc.detach().cpu().numpy()
            debug["K_proc"] = K_proc.detach().cpu().numpy()
            debug["V"] = V.detach().cpu().numpy()
            debug["mask"] = mask2.detach().cpu().numpy()

        # 3. 표준 Full-Attention 연산 (O(S^2))
        # raw_scores: (B, H, S, S)
        raw_scores = torch.matmul(Q_proc, K_proc.transpose(-1, -2))
        
        # 마스킹 적용 (음의 무한대 합산)
        logits = raw_scores - 1e9 * (1.0 - mask2[:, None, None, :].to(raw_scores.dtype))
        
        # Softmax: (B, H, S, S)
        attn_weights = torch.nn.functional.softmax(logits, dim=-1)
        
        # Context Vector 계산: (B, H, S, D)
        X = torch.matmul(attn_weights, V)

        # 4. 중간값 저장 (BERT 전용)
        if collect_intermediates:
            debug["raw_scores"] = raw_scores.detach().cpu().numpy()
            debug["attn_weights"] = attn_weights.detach().cpu().numpy()
            debug["X"] = X.detach().cpu().numpy()

        # 5. NPZ 파일 저장
        if collect_intermediates and save_path is not None:
            self._save_debug_npz(debug, save_path)

        return X

    def _save_debug_npz(self, debug, save_path):
        np_save = {}
        for k, v in debug.items():
            try:
                np_save[k] = np.asarray(v)
            except Exception as e:
                print(f"Failed to convert {k}: {e}")
        
        try:
            np.savez_compressed(save_path, **np_save)
        except Exception as e:
            print(f"Warning: failed to save BERT attention debug npz to {save_path}: {e}")

    def extra_repr(self):
        return f'standard_attention, seq_len={self.seq_len}, heads={self.num_head}'