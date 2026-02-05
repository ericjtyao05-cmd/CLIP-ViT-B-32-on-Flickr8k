from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from src.clip_utils import cosine_sim_matrix


def recall_at_k(ranks: torch.Tensor, k: int) -> float:
    # ranks: [N] (1-based rank of the correct item)
    return float((ranks <= k).float().mean().item())


def median_rank(ranks: torch.Tensor) -> float:
    return float(ranks.median().item())


def load_txt_lines(p: Path) -> List[str]:
    return [ln.rstrip("\n") for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", type=str, default="results/embeddings")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir)
    img_emb_cap = torch.load(emb_dir / f"{args.split}_image_embeds.pt")  # [Ncap, D]
    txt_emb = torch.load(emb_dir / f"{args.split}_text_embeds.pt")       # [Ncap, D]
    img_names_cap = load_txt_lines(emb_dir / f"{args.split}_image_names.txt")

    # Build unique image embeddings by taking the first occurrence for each image
    img_to_first_idx: Dict[str, int] = {}
    unique_img_names: List[str] = []
    unique_img_embs: List[torch.Tensor] = []
    for i, name in enumerate(img_names_cap):
        if name not in img_to_first_idx:
            img_to_first_idx[name] = i
            unique_img_names.append(name)
            unique_img_embs.append(img_emb_cap[i])

    img_emb = torch.stack(unique_img_embs, dim=0)  # [Nimg, D]

    # Map each caption row to its ground-truth image index in unique_img_names
    name_to_imgidx = {n: i for i, n in enumerate(unique_img_names)}
    gt_img_index_for_caption = torch.tensor([name_to_imgidx[n] for n in img_names_cap], dtype=torch.long)

    # For image->text: each image has multiple correct captions (typically 5)
    imgidx_to_caption_rows: Dict[int, List[int]] = defaultdict(list)
    for cap_row, img_idx in enumerate(gt_img_index_for_caption.tolist()):
        imgidx_to_caption_rows[img_idx].append(cap_row)

    # ---------- Text -> Image retrieval ----------
    # Similarity: captions as queries vs unique images as candidates
    sim_t2i = cosine_sim_matrix(txt_emb, img_emb)  # [Ncap, Nimg]
    # rank (1-based) of correct image for each caption
    sorted_idx_t2i = torch.argsort(sim_t2i, dim=1, descending=True)  # [Ncap, Nimg]
    # find position of gt in sorted list
    # create inverse ranks
    inv_rank = torch.empty_like(sorted_idx_t2i)
    inv_rank.scatter_(1, sorted_idx_t2i, torch.arange(sorted_idx_t2i.size(1)).unsqueeze(0).expand_as(sorted_idx_t2i))
    ranks_t2i = inv_rank[torch.arange(inv_rank.size(0)), gt_img_index_for_caption] + 1  # 1-based

    # ---------- Image -> Text retrieval ----------
    # Similarity: images as queries vs captions as candidates
    sim_i2t = cosine_sim_matrix(img_emb, txt_emb)  # [Nimg, Ncap]
    sorted_idx_i2t = torch.argsort(sim_i2t, dim=1, descending=True)  # [Nimg, Ncap]

    # For each image query, the "correct" set is multiple caption rows.
    # Rank for image query is the best (minimum) rank among its correct captions.
    ranks_i2t_list: List[int] = []
    for img_idx in range(img_emb.size(0)):
        correct_caps = set(imgidx_to_caption_rows[img_idx])
        # find first position where retrieved caption is in correct set
        row = sorted_idx_i2t[img_idx].tolist()
        best_rank = None
        for pos, cap_row in enumerate(row):
            if cap_row in correct_caps:
                best_rank = pos + 1  # 1-based
                break
        ranks_i2t_list.append(best_rank if best_rank is not None else sim_i2t.size(1) + 1)
    ranks_i2t = torch.tensor(ranks_i2t_list, dtype=torch.long)

    # ---------- Report ----------
    ks = [1, 5, 10]
    print(f"Split: {args.split}")
    print(f"Captions: {txt_emb.size(0)} | Unique images: {img_emb.size(0)}\n")

    print("Text → Image Retrieval")
    for k in ks:
        print(f"Recall@{k}: {recall_at_k(ranks_t2i, k):.4f}")
    print(f"MedR: {median_rank(ranks_t2i):.0f}\n")

    print("Image → Text Retrieval")
    for k in ks:
        print(f"Recall@{k}: {recall_at_k(ranks_i2t, k):.4f}")
    print(f"MedR: {median_rank(ranks_i2t):.0f}")


if __name__ == "__main__":
    main()
