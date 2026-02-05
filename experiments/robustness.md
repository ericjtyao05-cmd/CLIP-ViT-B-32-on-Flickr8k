# Robustness Experiments

## Zero-shot CLIP (Text → Image) under word dropout (Flickr8k test)

We evaluate how retrieval performance degrades when randomly dropping words from captions.
As dropout probability increases from 0.0 to 0.5, Recall@1 drops from 0.5109 to 0.1939 and MedR increases from 1 to 11,
showing substantial sensitivity to text corruption.

Artifacts:
- results/robustness/zero_shot_test_text_dropout.csv
- results/robustness/zero_shot_text_dropout_recall.png
- results/robustness/zero_shot_text_dropout_medr.png

## Zero-shot CLIP (Text → Image) under word shuffle (Flickr8k test)

Shuffling word order reduces performance but less severely than word dropout:
Recall@1 decreases to 0.3661 and MedR increases to 3, suggesting CLIP relies more on lexical content than strict word order
for this dataset.
Artifacts:
- results/robustness/zero_shot_test_text_shuffle.csv
