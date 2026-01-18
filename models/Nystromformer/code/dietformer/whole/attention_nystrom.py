
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
        
        if "inv_init_coeff_option" in config:
            self.init_option = config["inv_init_coeff_option"]
        elif "inv_coeff_init_option" in config:
            self.init_option = config["inv_coeff_init_option"]
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

        Q = Q * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        K = K * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))

        if collect_intermediates:
            debug["Q_proc"] = Q_proc.detach().cpu()
            debug["K_proc"] = K_proc.detach().cpu()
            debug["V"] = V.detach().cpu()
            debug["mask"] = mask.detach().cpu()
        
        if self.num_landmarks == self.seq_len:
            attn = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            X = torch.matmul(attn, V)

            if collect_intermediates:
                debug["raw_scores"] = raw_scores.detach().cpu()
                debug["attn"] = attn.detach().cpu()
                debug["X"] = X.detach().cpu()
        else:
            Q_landmarks = Q.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)
            K_landmarks = K.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)

            kernel_1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

            if collect_intermediates:
                debug['Q_landmarks'] = Q_landmarks.detach().cpu()
                debug['K_landmarks'] = K_landmarks.detach().cpu()
                debug['kernel_1'] = kernel_1.detach().cpu()
                debug['kernel_2'] = kernel_2.detach().cpu()
                debug['kernel_3'] = kernel_3.detach().cpu()
                debug['inv_kernel_2'] = inv_kernel_2.detach().cpu()
                debug['temp_kernel3_V'] = temp.detach().cpu()
                debug['mid_after_inv'] = mid.detach().cpu()
                debug['X'] = X.detach().cpu()

        if self.use_conv:
            X += self.conv(V * mask[:, None, :, None])
            if collect_intermediates:
                debug['conv_term'] = conv_term.detach().cpu()
                debug['X_final'] = X.detach().cpu()

        if collect_intermediates and save_path is not None:
            # convert debug to numpy where possible and save
            np_save = {}
            for k, v in debug.items():
                if isinstance(v, torch.Tensor):
                    np_save[k] = v.cpu().numpy()
                else:
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
