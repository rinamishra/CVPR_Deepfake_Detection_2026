# ntire_dataset.py
# ============================================================
# PLACE THIS IN THE REPO ROOT (same level as predict_ntire.py)
# DO **NOT** PUT IT INSIDE datasets/ — that folder's __init__.py
# will trigger the broken register/ import chain.
# ============================================================

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class NTIREDataset(Dataset):
    """
    Flat-folder loader for Deep_Dataset.
      train/  → filenames like 0000fake.jpg, 0001real.jpg  (label parsed from name)
      test/   → filenames like 0000.jpg                     (label = -1, unknown)
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.split = split
        self.folder = os.path.join(root_dir, split)

        # Alphabetical sort — CRITICAL: submission order must match this
        self.image_files = sorted([
            f for f in os.listdir(self.folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # 384×384 — matches the paper's training resolution exactly
        self.transform = transform or transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.folder, fname)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Label parsing (train only)
        if self.split == "train":
            if "fake" in fname.lower():
                label = 1
            elif "real" in fname.lower():
                label = 0
            else:
                raise ValueError(f"Cannot parse label from: {fname}")
        else:
            label = -1

        return image, label, fname