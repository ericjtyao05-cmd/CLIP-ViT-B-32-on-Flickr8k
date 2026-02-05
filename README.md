# CLIP-based Multimodal Retrieval

## Task
Bidirectional image-text retrieval on Flickr8k.

## Metrics
Recall@K for K ∈ {1, 5, 10} for both directions:
- Image → Text
- Text → Image

Also report:
- Median Rank (MedR)

## Data Split
Use official dataset splits if available. Otherwise, use a fixed random split with a fixed seed.

## Results

### Zero-shot CLIP (openai/clip-vit-base-patch32) on Flickr8k (test)

- **Text → Image**
  - R@1: 0.5109
  - R@5: 0.7793
  - R@10: 0.8644
  - MedR: 1

- **Image → Text**
  - R@1: 0.6823
  - R@5: 0.8840
  - R@10: 0.9457
  - MedR: 1

See `results/zero_shot_test.md` for the full evaluation log.
