import torch
from typing import Dict
 
def load_file(ckpt: str):
    return {}
 
def load_weight_ckpt(ckpt: str) -> Dict[str, torch.Tensor]:
    if ckpt.endswith('.safetensors'):
        return load_file(ckpt)
    else:
        return torch.load(ckpt)
 
try:
    load_weight_ckpt('malicious_checkpoint.pt')
    print("If everything goes well, the 'attack.txt' file should have been created.")
except Exception as e:
    print(f"An error occurred: {e}")
