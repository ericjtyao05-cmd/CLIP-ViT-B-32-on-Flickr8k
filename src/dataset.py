from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


@dataclass
class Flickr8kSample:
    pixel_values: torch.FloatTensor
    text: str
    image_name: str
    caption_idx: int  # 0..4 (typically), caption index within an image


class Flickr8kDataset(Dataset):
    """
    Expects:
      data/flickr8k/
        images/
        captions.txt               # TSV: image_name \\t caption
        splits/{train,val,test}.txt  # one image name per line

    Returns one sample per (image, caption) pair:
      - pixel_values: FloatTensor [3, H, W]
      - text: str
      - image_name: str
      - caption_idx: int (per-image caption id)
    """

    def __init__(self, root: str | Path, split: str = "train", image_size: int = 224):
        self.root = Path(root)
        self.split = split

        self.images_dir = self.root / "images"
        self.captions_path = self.root / "captions.txt"
        self.split_path = self.root / "splits" / f"{split}.txt"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {self.images_dir}")
        if not self.captions_path.exists():
            raise FileNotFoundError(f"Missing captions file: {self.captions_path}")
        if not self.split_path.exists():
            raise FileNotFoundError(f"Missing split file: {self.split_path}")

        # Basic image preprocessing (Phase 2.5 baseline).
        # In Phase 3 we may switch to CLIPProcessor/CLIPImageProcessor for exact CLIP normalization.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Load split image names
        self.split_images: List[str] = [
            ln.strip()
            for ln in self.split_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if ln.strip()
        ]
        self.split_images_set = set(self.split_images)

        # Load captions TSV and keep only those in current split
        # captions.txt lines: image_name \t caption
        self.samples: List[Tuple[str, str, int]] = []

        # caption counter per image to assign caption_idx deterministically
        cap_counter: Dict[str, int] = {}

        for ln in self.captions_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not ln.strip():
                continue
            parts = ln.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue

            img_name = parts[0].strip()
            caption = parts[1].strip()

            if img_name not in self.split_images_set:
                continue

            cap_counter.setdefault(img_name, 0)
            cap_idx = cap_counter[img_name]
            cap_counter[img_name] += 1

            self.samples.append((img_name, caption, cap_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split='{split}'. "
                f"Check captions.txt and splits/{split}.txt alignment."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Flickr8kSample:
        img_name, caption, cap_idx = self.samples[idx]
        img_path = self.images_dir / img_name

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        return Flickr8kSample(
            pixel_values=pixel_values,
            text=caption,
            image_name=img_name,
            caption_idx=cap_idx,
        )


