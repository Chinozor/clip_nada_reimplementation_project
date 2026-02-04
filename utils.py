import os
import random
from typing import Optional, Dict, Any, Union

import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def tensor_to_pil(img):
    img = (img.clamp(-1, 1) + 1) / 2.0  # [1, 3, H, W] -> [0, 1]
    img = img.squeeze(0).detach().cpu()  # [3, H, W]
    return transforms.ToPILImage()(img)

def show_img(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def save_img(img, path):
    ensure_dir(os.path.dirname(path))
    pil = tensor_to_pil(img)
    pil.save(path)

def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(path: str, step: int, generator_t, optimizer, extra: Optional[Dict[str, Any]] = None) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {
        "step": step,
        "generator_t": generator_t.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)