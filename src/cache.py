from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from transformers import CLIPModel, CLIPProcessor

from src.dataset import Flickr8kDataset
from src.clip_utils import l2_normalize


def collate_fn(batch):
    # batch is list[Flickr8kSample]
    pixel_values = torch.stack([b.pixel_values for b in batch], dim=0)
    texts = [b.text for b in batch]
    image_names = [b.image_name for b in batch]
    return pixel_values, texts, image_names


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/flickr8k")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)  # macOS safe default
    ap.add_argument("--out_dir", type=str, default="results/embeddings")
    ap.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset returns per-caption samples; we embed each caption and its paired image.
    ds = Flickr8kDataset(args.data_root, split=args.split, image_size=224)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = CLIPModel.from_pretrained(args.model_name)
    processor = CLIPProcessor.from_pretrained(args.model_name, use_fast=False)

    model.eval()
    model.to(device)

    all_img_embs: List[torch.Tensor] = []
    all_txt_embs: List[torch.Tensor] = []
    all_img_names: List[str] = []
    all_texts: List[str] = []

    for pixel_values, texts, image_names in dl:
        pixel_values = pixel_values.to(device)

        # text tokenization (CPU tensors are okay; we move to device)
        text_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        outputs = model(
            pixel_values=pixel_values,
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            return_dict=True,
        )

        # CLIP image/text embeddings
        img_emb = outputs.image_embeds  # [B, D]
        txt_emb = outputs.text_embeds   # [B, D]

        # Normalize for cosine similarity
        img_emb = l2_normalize(img_emb).to("cpu")
        txt_emb = l2_normalize(txt_emb).to("cpu")

        all_img_embs.append(img_emb)
        all_txt_embs.append(txt_emb)
        all_img_names.extend(image_names)
        all_texts.extend(texts)

    image_embeddings = torch.cat(all_img_embs, dim=0)  # [Ncap, D] (duplicated per caption)
    text_embeddings = torch.cat(all_txt_embs, dim=0)   # [Ncap, D]
    # We also store image_names aligned with each caption row
    # Later we'll deduplicate image embeddings for image->text retrieval
    torch.save(image_embeddings, out_dir / f"{args.split}_image_embeds.pt")
    torch.save(text_embeddings, out_dir / f"{args.split}_text_embeds.pt")
    (out_dir / f"{args.split}_image_names.txt").write_text("\n".join(all_img_names) + "\n", encoding="utf-8")
    (out_dir / f"{args.split}_texts.txt").write_text("\n".join(all_texts) + "\n", encoding="utf-8")

    print("✅ Cached embeddings")
    print("split:", args.split)
    print("rows (captions):", len(all_img_names))
    print("image_embeds:", tuple(image_embeddings.shape))
    print("text_embeds :", tuple(text_embeddings.shape))
    print("out_dir:", str(out_dir.resolve()))


if __name__ == "__main__":
    main()
