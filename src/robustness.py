from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import CLIPModel, CLIPProcessor

from src.clip_utils import l2_normalize, cosine_sim_matrix


def load_txt_lines(p: Path) -> List[str]:
    return [ln.rstrip("\n") for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]


def build_unique_images(
    img_emb_cap: torch.Tensor, img_names_cap: List[str]
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    From per-caption image embeddings, deduplicate to unique image embeddings.

    Returns:
      img_emb: [Nimg, D]
      unique_img_names: list[str] length Nimg
      gt_img_index_for_caption: [Ncap] long, mapping caption-row -> unique image idx
    """
    img_to_first_idx: Dict[str, int] = {}
    unique_img_names: List[str] = []
    unique_img_embs: List[torch.Tensor] = []

    for i, name in enumerate(img_names_cap):
        if name not in img_to_first_idx:
            img_to_first_idx[name] = i
            unique_img_names.append(name)
            unique_img_embs.append(img_emb_cap[i])

    img_emb = torch.stack(unique_img_embs, dim=0)  # [Nimg, D]
    name_to_imgidx = {n: i for i, n in enumerate(unique_img_names)}
    gt_img_index_for_caption = torch.tensor([name_to_imgidx[n] for n in img_names_cap], dtype=torch.long)
    return img_emb, unique_img_names, gt_img_index_for_caption


def ranks_text_to_image(sim_t2i: torch.Tensor, gt_img_index_for_caption: torch.Tensor) -> torch.Tensor:
    """
    sim_t2i: [Ncap, Nimg]
    gt_img_index_for_caption: [Ncap]
    returns ranks (1-based) [Ncap]
    """
    sorted_idx = torch.argsort(sim_t2i, dim=1, descending=True)  # [Ncap, Nimg]
    inv_rank = torch.empty_like(sorted_idx)
    inv_rank.scatter_(
        1,
        sorted_idx,
        torch.arange(sorted_idx.size(1)).unsqueeze(0).expand_as(sorted_idx),
    )
    ranks = inv_rank[torch.arange(inv_rank.size(0)), gt_img_index_for_caption] + 1
    return ranks


def recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= k).float().mean().item())


def median_rank(ranks: torch.Tensor) -> float:
    return float(ranks.median().item())


def perturb_dropout_words(text: str, p: float, rng: random.Random) -> str:
    """
    Drop each word with probability p. Ensure at least 1 token remains.
    """
    words = text.strip().split()
    if not words:
        return text
    kept = [w for w in words if rng.random() > p]
    if len(kept) == 0:
        kept = [rng.choice(words)]
    return " ".join(kept)


def perturb_shuffle_words(text: str, rng: random.Random) -> str:
    words = text.strip().split()
    if len(words) <= 2:
        return text
    rng.shuffle(words)
    return " ".join(words)


@torch.no_grad()
@torch.no_grad()
def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: List[str],
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """
    Returns l2-normalized text embeddings: [N, D] on CPU.
    """
    out: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        text_inputs = processor(
            text=chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # ✅ CLIP text-only path
        emb = model.get_text_features(**text_inputs)  # [B, D]
        emb = l2_normalize(emb).to("cpu")
        out.append(emb)

    return torch.cat(out, dim=0)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", type=str, default="results/embeddings")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--text_batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--mode", type=str, required=True, choices=["dropout", "shuffle"])
    ap.add_argument("--dropout_ps", type=str, default="0,0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = random.Random(args.seed)

    emb_dir = Path(args.emb_dir)
    img_emb_cap = torch.load(emb_dir / f"{args.split}_image_embeds.pt")  # [Ncap, D]
    img_names_cap = load_txt_lines(emb_dir / f"{args.split}_image_names.txt")
    texts_clean = load_txt_lines(emb_dir / f"{args.split}_texts.txt")

    if not (len(img_names_cap) == len(texts_clean) == img_emb_cap.size(0)):
        raise RuntimeError("Cached files misaligned: check image_names/texts/embeddings lengths.")

    # Build unique image embeddings
    img_emb, unique_img_names, gt_img_index_for_caption = build_unique_images(img_emb_cap, img_names_cap)

    # Load CLIP for text encoding
    model = CLIPModel.from_pretrained(args.model_name)
    processor = CLIPProcessor.from_pretrained(args.model_name, use_fast=False)  # fixed for reproducibility
    model.eval()
    model.to(device)

    # Prepare output
    out_dir = Path("results/robustness")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        out_csv = out_dir / f"zero_shot_{args.split}_text_{args.mode}.csv"

    # Parse dropout probabilities
    if args.mode == "dropout":
        ps = [float(x.strip()) for x in args.dropout_ps.split(",") if x.strip() != ""]
    else:
        ps = [0.0]  # single run for shuffle

    rows = []
    ks = [1, 5, 10]

    for p in ps:
        if args.mode == "dropout":
            perturbed = [perturb_dropout_words(t, p=p, rng=rng) for t in texts_clean]
            tag = f"dropout_p={p:.2f}"
        else:
            perturbed = [perturb_shuffle_words(t, rng=rng) for t in texts_clean]
            tag = "shuffle"

        txt_emb = encode_texts(
            model=model,
            processor=processor,
            texts=perturbed,
            device=device,
            batch_size=args.text_batch_size,
        )

        sim_t2i = cosine_sim_matrix(txt_emb, img_emb)  # [Ncap, Nimg]
        ranks = ranks_text_to_image(sim_t2i, gt_img_index_for_caption)

        result = {
            "mode": args.mode,
            "setting": tag,
            "captions": len(texts_clean),
            "images": len(unique_img_names),
            "medr": int(median_rank(ranks)),
        }
        for k in ks:
            result[f"r@{k}"] = recall_at_k(ranks, k)

        rows.append(result)

        print(f"\n[{tag}] Text → Image")
        for k in ks:
            print(f"Recall@{k}: {result[f'r@{k}']:.4f}")
        print(f"MedR: {result['medr']}")

    # Write CSV
    fieldnames = ["mode", "setting", "captions", "images", "r@1", "r@5", "r@10", "medr"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n✅ Saved: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
