import numpy as np
from PIL import Image
import os

def analyze_noise(noise_dir, sample_n=200):
    files = sorted([f for f in os.listdir(noise_dir) if f.endswith('_n.png')])[:sample_n]
    variances = []
    sp_counts = []
    for f in files:
        img = np.array(Image.open(os.path.join(noise_dir,f)).convert('L')).astype(np.float32)
        # estimate variance by local differences
        diffs = img.astype(np.float32) - Image.fromarray(img).filter(Image.Filter.BLUR).resize(img.shape[::-1]) if False else None
        variances.append(np.var(img))
        # detect extreme pixels near 0 or 255
        sp = np.mean((img<10) | (img>245))
        sp_counts.append(sp)
    return {'mean_variance': float(np.mean(variances)), 'std_variance': float(np.std(variances)), 'mean_sp_ratio': float(np.mean(sp_counts))}
