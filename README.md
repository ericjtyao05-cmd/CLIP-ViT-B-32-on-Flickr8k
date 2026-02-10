# Multimodal Retrieval Robustness Study (CLIP on Flickr8k)

## 1. Introduction

Multimodal retrieval models such as CLIP have demonstrated strong zero-shot alignment between images and text.
However, their robustness under corrupted or noisy textual input remains less explored in small-scale empirical settings.

This project conducts a systematic study of:
- Zero-shot CLIP performance on Flickr8k
- Robustness under word dropout and word shuffle
- Lightweight text-only alignment fine-tuning
- Failure modes under multi-caption (multi-positive) contrastive learning

The goal is not to train a large model from scratch, but to analyze behavior and failure patterns under controlled perturbations.

---

## 2. Method

### 2.1 Baseline: Zero-shot CLIP

We use:
- Model: `openai/clip-vit-base-patch32`
- Dataset: Flickr8k
- Metrics:
  - Recall@1 / Recall@5 / Recall@10
  - Median Rank (MedR)

All embeddings are cached for reproducibility.

---

### 2.2 Robustness Experiments

We evaluate text corruption using:

1. Word Dropout  
   Randomly remove each word with probability p ∈ {0.0, 0.1, ..., 0.5}

2. Word Shuffle  
   Randomly permute word order in captions

Image embeddings remain fixed. Only text embeddings are recomputed.

---

### 2.3 Lightweight Alignment Fine-Tuning

We introduce a small 2-layer MLP projection head on top of frozen CLIP text embeddings:

- CLIP backbone is frozen
- Only text projection head is trained
- Training objective: batch-wise contrastive loss (InfoNCE)

We evaluate:
- Clean performance
- Robustness under corruption

---

## 3. Experimental Setup

Dataset: Flickr8k  
Images: 8,000  
Captions: 5 per image  
Test split: 1,215 images (6,075 captions)

Training:
- Epochs: 3
- Batch size: 64
- Optimizer: AdamW
- Temperature: 0.07
- Backbone: Frozen

Hardware:
- Apple Silicon (MPS backend)

---

## 4. Results

### 4.1 Zero-shot CLIP (Text → Image)

| Metric | Score |
|--------|-------|
| Recall@1 | 0.5109 |
| Recall@5 | 0.7793 |
| Recall@10 | 0.8644 |
| MedR | 1 |

---

### 4.2 Zero-shot Robustness (Word Dropout)

At p = 0.5:

| Metric | Score |
|--------|-------|
| Recall@1 | 0.1939 |
| Recall@5 | 0.3942 |
| Recall@10 | 0.4981 |
| MedR | 11 |

Performance degrades significantly under severe text corruption.

---

### 4.3 Text Projection Head (Frozen CLIP)

Test performance:

| Metric | Score |
|--------|-------|
| Recall@1 | 0.4352 |
| Recall@5 | 0.7378 |
| Recall@10 | 0.8413 |
| MedR | 2 |

The lightweight head does not outperform zero-shot CLIP on clean test data.

---

### 4.4 Robustness After Fine-Tuning

Under word dropout (p=0.5):

| Metric | Score |
|--------|-------|
| Recall@1 | 0.1536 |
| Recall@5 | 0.3539 |
| Recall@10 | 0.4612 |
| MedR | 13 |

The fine-tuned head is more sensitive to text corruption than zero-shot CLIP.

---

## 5. Discussion

The degradation after fine-tuning can be explained by a fundamental issue:

Flickr8k is a multi-positive dataset:
Each image has 5 valid captions.

Standard batch-wise contrastive loss treats other captions in the batch as negatives.
When multiple captions of the same image appear in a batch, they become false negatives.

This misalignment pressure:

- Reduces clean retrieval performance
- Decreases robustness under text corruption

This highlights the importance of multi-positive contrastive learning strategies.

---

## 6. Key Observations

1. Zero-shot CLIP is surprisingly robust to moderate text corruption.
2. Naive head-only fine-tuning can harm both clean accuracy and robustness.
3. False negatives in multi-caption datasets are a critical issue.
4. Lightweight alignment strategies must consider multi-positive structure.

---

## 7. Future Work

- Multi-positive contrastive loss
- Debiased negative sampling
- Text noise-aware training
- Vision-side alignment fine-tuning
- Cross-dataset robustness evaluation

---

## 8. Repository Structure


src/
dataset.py
cache.py
eval.py
robustness.py
head_model.py
train_head.py
results/
embeddings/
robustness/

tools/
prepare_flickr8k.py
plot_robustness.py

## 9. Reproducibility

To reproduce zero-shot baseline:

python -m src.cache --split test
python -m src.eval

To run robustness experiments:

python -m src.robustness --mode dropout
python -m src.robustness --mode shuffle

To train projection head:

python -m src.train_head

---

## Conclusion

This project demonstrates that small-scale empirical analysis can reveal meaningful failure modes in multimodal retrieval systems, even without large-scale training.

Understanding robustness and contrastive learning structure is as important as optimizing raw performance.