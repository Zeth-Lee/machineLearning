import torch
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def save_checkpoint(state, ckpt_dir, name='checkpoint.pth'):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(state, os.path.join(ckpt_dir, name))

def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt

def tensor_to_base64(tensor):
    # tensor expected CxHxW, values 0..1
    arr = tensor.squeeze().cpu().numpy()
    arr = (arr*255.0).clip(0,255).astype(np.uint8)
    img = Image.fromarray(arr, mode='L')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode('ascii')
    return b64