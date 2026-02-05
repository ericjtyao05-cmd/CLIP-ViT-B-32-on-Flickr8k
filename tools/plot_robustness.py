from __future__ import annotations
import csv
from pathlib import Path

import matplotlib.pyplot as plt

csv_path = Path("results/robustness/zero_shot_test_text_dropout.csv")
out_dir = Path("results/robustness")
out_dir.mkdir(parents=True, exist_ok=True)

ps, r1, r5, r10, medr = [], [], [], [], []
with csv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        setting = row["setting"]  # dropout_p=0.10
        p = float(setting.split("=")[1])
        ps.append(p)
        r1.append(float(row["r@1"]))
        r5.append(float(row["r@5"]))
        r10.append(float(row["r@10"]))
        medr.append(float(row["medr"]))

# Recall curves
plt.figure()
plt.plot(ps, r1, marker="o", label="Recall@1")
plt.plot(ps, r5, marker="o", label="Recall@5")
plt.plot(ps, r10, marker="o", label="Recall@10")
plt.xlabel("Word dropout probability p")
plt.ylabel("Recall")
plt.title("Zero-shot CLIP robustness (Text→Image) under word dropout")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "zero_shot_text_dropout_recall.png", dpi=200)

# MedR curve
plt.figure()
plt.plot(ps, medr, marker="o")
plt.xlabel("Word dropout probability p")
plt.ylabel("Median Rank (MedR)")
plt.title("Zero-shot CLIP robustness (Text→Image) MedR under word dropout")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "zero_shot_text_dropout_medr.png", dpi=200)

print("Saved plots to:", out_dir.resolve())
