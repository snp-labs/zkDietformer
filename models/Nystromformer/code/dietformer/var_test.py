import torch
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased").eval()
input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]])  # 짧은 sample

# Hook으로 LayerNorm 입력의 var 측정
all_vars = []
def capture_var(module, input, output):
    x = input[0]
    var = x.var(dim=-1)
    all_vars.append(var.flatten().cpu().numpy())

for layer in model.encoder.layer:
    layer.attention.output.LayerNorm.register_forward_hook(capture_var)
    layer.output.LayerNorm.register_forward_hook(capture_var)

with torch.no_grad():
    model(input_ids)

import numpy as np
all_vars = np.concatenate(all_vars)
print(f"var 분포:")
print(f"  min:    {all_vars.min():.4f}")
print(f"  5%:     {np.percentile(all_vars, 5):.4f}")
print(f"  median: {np.median(all_vars):.4f}")
print(f"  95%:    {np.percentile(all_vars, 95):.4f}")
print(f"  max:    {all_vars.max():.4f}")