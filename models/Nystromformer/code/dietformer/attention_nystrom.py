
import torch
import torch.nn as nn
import math
import numpy as np

# 과정중 입력값, 중간값, 출력값들을 저장하도록 수정
class NystromAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.num_landmarks = config["num_landmarks"]
        self.seq_len = config["seq_len"]
        
        if "inv_coeff_init_option" in config:
            self.init_option = config["inv_init_coeff_option"]
        else:
            self.init_option = "original"

        self.use_conv = "conv_kernel_size" in config
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = self.num_head, out_channels = self.num_head,
                kernel_size = (config["conv_kernel_size"], 1), padding = (config["conv_kernel_size"] // 2, 0),
                bias = False,
                groups = self.num_head)

    def forward(self, Q, K, V, mask, collect_intermediates: bool = False, save_path: str = None):

        debug = {}

        # 마스크의 shape을 (B,S)로 정규화 + dtype을 Q,K와 맞춤
        if mask.dim() == 3 and mask.size(-1) == 1:
            # (B, S, 1) -> (B, S)
            mask2 = mask.squeeze(-1)
        elif mask.dim() == 2:
            mask2 = mask
        else:
            raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}; expected (B,S) or (B,S,1)")

        # dtype이 잘 맞춰져 있는지 확인
        mask2 = mask2.to(Q.dtype).to(Q.device)

        scale = 1.0 / math.sqrt(math.sqrt(self.head_dim))
        # apply mask and scale
        Q_proc = Q * mask2[:, None, :, None] * scale
        K_proc = K * mask2[:, None, :, None] * scale

        # 중간값 저장
        if collect_intermediates:
            debug["Q_proc"] = Q_proc.detach().cpu().numpy() # 커밋
            debug["K_proc"] = K_proc.detach().cpu().numpy() # 커밋
            debug["V"] = V.detach().cpu().numpy() # 커밋
            debug["mask"] = mask2.detach().cpu().numpy()

        if self.num_landmarks == self.seq_len:
            raw_scores = torch.matmul(Q_proc, K_proc.transpose(-1, -2))
            logits = raw_scores - 1e9 * (1.0 - mask2[:, None, None, :].to(raw_scores.dtype))
            attn = torch.nn.functional.softmax(logits, dim = -1)
            X = torch.matmul(attn, V)

            if collect_intermediates:
                debug["raw_scores"] = raw_scores.detach().cpu().numpy()
                debug["attn"] = attn.detach().cpu().numpy()
                debug["X"] = X.detach().cpu().numpy()
        else:
            # compute landmarks (mean over segments)
            seg = self.seq_len // self.num_landmarks
            # Q_proc: (B,H,S,D) -> Q_landmarks: (B,H,L,D)
            Q_landmarks = Q_proc.reshape(-1, self.num_head, self.num_landmarks, seg, self.head_dim).mean(dim = -2)
            K_landmarks = K_proc.reshape(-1, self.num_head, self.num_landmarks, seg, self.head_dim).mean(dim = -2)

            # kernel_1: (B,H,S,L) = softmax(Q * K_landmarks^T)
            kernel_1_scores = torch.matmul(Q_proc, K_landmarks.transpose(-1, -2))
            kernel_1 = torch.nn.functional.softmax(kernel_1_scores, dim = -1)

            # kernel_2: (B,H,L,L) = softmax(Q_landmarks * K_landmarks^T)
            kernel_2_scores = torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2))
            kernel_2 = torch.nn.functional.softmax(kernel_2_scores, dim = -1)

            # kernel_3: (B,H,L,S) = softmax(Q_landmarks * K^T - mask)
            kernel_3_scores = torch.matmul(Q_landmarks, K_proc.transpose(-1, -2))
            kernel_3_logits = kernel_3_scores - 1e9 * (1.0 - mask2[:, None, None, :].to(kernel_3_scores.dtype))
            kernel_3 = torch.nn.functional.softmax(kernel_3_logits, dim = -1)

            # iterative inverse of kernel_2 -> P approx (B,H,L,L)
            inv_kernel_2 = self.iterative_inv(kernel_2)

            # temp = kernel_3 @ V  -> (B,H,L,D)
            temp = torch.matmul(kernel_3, V)
            # mid = kernel_1 @ inv_kernel_2 -> (B,H,S,L)
            mid = torch.matmul(kernel_1, inv_kernel_2)
            # X = mid @ temp -> (B,H,S,D)
            X = torch.matmul(mid, temp)

            if collect_intermediates:
                debug['Q_landmarks'] = Q_landmarks.detach().cpu().numpy()
                debug['K_landmarks'] = K_landmarks.detach().cpu().numpy()
                debug['kernel_1'] = kernel_1.detach().cpu().numpy()
                debug['kernel_2'] = kernel_2.detach().cpu().numpy()
                debug['kernel_3'] = kernel_3.detach().cpu().numpy()
                debug['kernel_1_scores'] = kernel_1_scores.detach().cpu().numpy()
                debug['kernel_2_scores'] = kernel_2_scores.detach().cpu().numpy()
                debug['kernel_3_scores'] = kernel_3_scores.detach().cpu().numpy()
                debug['inv_kernel_2'] = inv_kernel_2.detach().cpu().numpy()
                debug['temp'] = temp.detach().cpu().numpy()
                debug['mid'] = mid.detach().cpu().numpy()
                debug['X'] = X.detach().cpu().numpy()

        # optional conv term
        if self.use_conv:
            conv_term = self.conv(V * mask2[:, None, :, None])
            X = X + conv_term
            if collect_intermediates:
                debug['conv_term'] = conv_term.detach().cpu().numpy()
                debug['X_final'] = X.detach().cpu().numpy()

        # npz로 중간값 저장
        if collect_intermediates and save_path is not None:
            np_save = {}
            for k, v in debug.items():
                try:
                    np_save[k] = np.asarray(v)
                except Exception:
                    try:
                        np_save[k] = np.array(v)
                    except Exception:
                        pass
            try:
                np.savez_compressed(save_path, **np_save)
            except Exception as e:
                # don't fail forward if saving fails; just print warning
                print(f"Warning: failed to save attention debug npz to {save_path}: {e}")

        if collect_intermediates:
            return X, debug
        else:
            return X


    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        
        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        if self.init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0. 
            V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence. 
            V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)
            
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, seq_len={self.seq_len}'
