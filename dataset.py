import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class DenoiseDataset(Dataset):
    def __init__(self, noise_dir, clean_dir=None, transform=None, augment=False):
        self.noise_dir = noise_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.augment = augment
        self.ids = sorted([f for f in os.listdir(noise_dir) if f.endswith('_n.png')])
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        nid = self.ids[idx]
        nid_path = os.path.join(self.noise_dir, nid)
        noisy = Image.open(nid_path).convert('L')
        noisy = np.array(noisy).astype(np.float32)/255.0
        if self.clean_dir is not None:
            cid = nid.replace('_n.png','_o.png')
            clean = Image.open(os.path.join(self.clean_dir, cid)).convert('L')
            clean = np.array(clean).astype(np.float32)/255.0
        else:
            clean = noisy.copy()
        # augmentation: random small rotations and flips
        if self.augment:
            if np.random.rand() < 0.5:
                noisy = np.fliplr(noisy).copy()
                clean = np.fliplr(clean).copy()
            r = np.random.choice([0,1,2,3])
            noisy = np.rot90(noisy, r).copy()
            clean = np.rot90(clean, r).copy()
        noisy = torch.from_numpy(noisy).unsqueeze(0).float()
        clean = torch.from_numpy(clean).unsqueeze(0).float()
        return noisy, clean, int(nid.replace('_n.png',''))
