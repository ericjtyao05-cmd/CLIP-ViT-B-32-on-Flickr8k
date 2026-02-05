#!/usr/bin/env python3
"""
Prepare Flickr8k (Kaggle layout) into a standardized project layout.

Input (Kaggle):
  <input_dir>/
    Images/
    Text/
      Flickr8k.token.txt
      Flickr_8k.trainImages.txt (optional)
      Flickr_8k.devImages.txt   (optional)
      Flickr_8k.testImages.txt  (optional)

Output (standard):
  <output_dir>/flickr8k/
    images/
    captions.txt      (TSV: image_name \\t caption)
    splits/
      train.txt
      val.txt
      test.txt

Usage examples:
  python tools/prepare_flickr8k.py --input data/raw/flickr8k_kaggle --output data
  python tools/prepare_flickr8k.py --input ~/Downloads/Flickr8k --output data

If official split files are missing, it will do a deterministic random split.
"""

from __future__ import annotations
import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def read_lines(p: Path) -> List[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]


def parse_kaggle_token_file(token_path: Path) -> Dict[str, List[str]]:
    """
    Kaggle token format:
      1000268201_693b08cb0e.jpg#0 A child in a pink dress is climbing up a set of stairs in an entry way .
    We convert to:
      image_name -> [caption1, caption2, ...]
    """
    mapping: Dict[str, List[str]] = {}
    for ln in read_lines(token_path):
        # Split only on first whitespace
        parts = ln.split(maxsplit=1)
        if len(parts) != 2:
            continue
        img_id, caption = parts
        # Remove trailing #N
        img_name = img_id.split("#")[0]
        mapping.setdefault(img_name, []).append(caption)
    return mapping


def write_captions_tsv(out_path: Path, mapping: Dict[str, List[str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for img_name in sorted(mapping.keys()):
            for cap in mapping[img_name]:
                f.write(f"{img_name}\t{cap}\n")


def copy_images(src_images_dir: Path, dst_images_dir: Path, image_names: List[str]) -> Tuple[int, int]:
    """
    Copy only images referenced by split lists.
    Returns (copied, missing).
    """
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    for name in image_names:
        src = src_images_dir / name
        dst = dst_images_dir / name
        if not src.exists():
            missing += 1
            continue
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied, missing


def find_kaggle_paths(input_dir: Path) -> Tuple[Path, Path]:
    """
    Accept both 'Images'/'Text' and lowercase variants.
    """
    candidates = [
        ("Images", "Text"),
        ("images", "text"),
        ("Flickr8k_Dataset", "Flickr8k_text"),  # some variants
        ("Flicker8k_Images", "Flickr8k_text"),  # Kaggle variant (note: Flicker)
        ("Flickr8k_Images", "Flickr8k_text"),
    ]
    for img_dir_name, text_dir_name in candidates:
        img_dir = input_dir / img_dir_name
        text_dir = input_dir / text_dir_name
        if img_dir.exists() and text_dir.exists():
            return img_dir, text_dir
    raise FileNotFoundError(
        f"Could not find Kaggle-style subfolders under: {input_dir}\n"
        "Expected something like <input>/Images and <input>/Text."
    )


def load_official_splits(text_dir: Path) -> Tuple[List[str], List[str], List[str]] | None:
    train_p = text_dir / "Flickr_8k.trainImages.txt"
    val_p = text_dir / "Flickr_8k.devImages.txt"
    test_p = text_dir / "Flickr_8k.testImages.txt"
    if train_p.exists() and val_p.exists() and test_p.exists():
        return read_lines(train_p), read_lines(val_p), read_lines(test_p)
    return None


def random_split(image_names: List[str], seed: int, train_ratio: float, val_ratio: float) -> Tuple[List[str], List[str], List[str]]:
    assert 0 < train_ratio < 1
    assert 0 < val_ratio < 1
    assert train_ratio + val_ratio < 1
    rng = random.Random(seed)
    names = list(sorted(set(image_names)))
    rng.shuffle(names)
    n = len(names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = names[:n_train]
    val = names[n_train:n_train + n_val]
    test = names[n_train + n_val:]
    return train, val, test


def write_split(out_path: Path, names: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(names) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Kaggle Flickr8k folder containing Images/ and Text/")
    ap.add_argument("--output", default="data", help="Output root dir (default: data)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (used only if official split files are missing)")
    ap.add_argument("--train_ratio", type=float, default=0.70, help="Train ratio if random split is used")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="Val ratio if random split is used")
    ap.add_argument("--copy_all_images", action="store_true", help="Copy all images (not just split-referenced)")
    args = ap.parse_args()

    input_dir = Path(os.path.expanduser(args.input)).resolve()
    out_root = Path(os.path.expanduser(args.output)).resolve()
    out_dir = out_root / "flickr8k"
    out_images = out_dir / "images"
    out_splits = out_dir / "splits"

    img_dir, text_dir = find_kaggle_paths(input_dir)

    token_path = text_dir / "Flickr8k.token.txt"
    if not token_path.exists():
        raise FileNotFoundError(f"Missing token file: {token_path}")

    mapping = parse_kaggle_token_file(token_path)

    # Write captions.tsv (named captions.txt per our project spec)
    captions_out = out_dir / "captions.txt"
    write_captions_tsv(captions_out, mapping)

    # Determine splits
    official = load_official_splits(text_dir)
    if official is not None:
        train_names, val_names, test_names = official
        split_mode = "official"
    else:
        all_names = list(mapping.keys())
        train_names, val_names, test_names = random_split(
            all_names, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
        )
        split_mode = f"random(seed={args.seed}, train={args.train_ratio}, val={args.val_ratio})"

    # Write split files
    write_split(out_splits / "train.txt", train_names)
    write_split(out_splits / "val.txt", val_names)
    write_split(out_splits / "test.txt", test_names)

    # Copy images
    if args.copy_all_images:
        # Copy everything under Images/
        out_images.mkdir(parents=True, exist_ok=True)
        copied = 0
        missing = 0
        for p in img_dir.glob("*"):
            if p.is_file():
                dst = out_images / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)
                    copied += 1
        print(f"[images] copied all files: {copied}")
    else:
        needed = sorted(set(train_names + val_names + test_names))
        copied, missing = copy_images(img_dir, out_images, needed)
        print(f"[images] copied referenced: {copied}, missing referenced: {missing}")

    print("\n✅ Prepared Flickr8k dataset")
    print(f"Input:  {input_dir}")
    print(f"Output: {out_dir}")
    print(f"Splits: {split_mode}")
    print(f"Captions lines: {sum(len(v) for v in mapping.values())}")
    print(f"Unique images in captions: {len(mapping)}")
    print("\nExpected layout:")
    print(f"  {out_dir}/images/")
    print(f"  {out_dir}/captions.txt")
    print(f"  {out_dir}/splits/train.txt, val.txt, test.txt")


if __name__ == "__main__":
    main()
