# Research workflows

## Project idea discovery

Generate 5–10 concrete problems. Score each (1–5) by user impact, data availability, feasible compute, novelty potential, demonstrability, and privacy/ethics. Recommend top three with explicit assumptions and a two-week starter milestone.

## Paper review

Define query strings and inclusion criteria, search at least two distinct indexes, verify important full texts, inspect citations and limitations. Aim for 5–12 relevant papers, but disclose sparse coverage. Output a comparison matrix and what remains uncertain.

## Dataset shortlist

Collect 3–5 candidates. Inspect official cards for actual availability, licenses, annotations, leakage, preprocessing, original train/test split and domain relevance. Recommend a fast iteration set plus an independent generalization set where possible.

## Model selection

Choose 3–4 candidate families ranging from cheap baseline to strong/efficient model. Link original methodology and implementation. Specify shared evaluation conditions, expected engineering difficulty, memory footprint when documented, and fallback if a repo or weights are missing.

## Gap analysis

Write each hypothesis as: Observation → supporting paper → counterevidence → testable improvement → experiment → rejection threshold.

## POC planning

Define milestones: data audit, baseline, stronger model, hyperparameter/ablation, robustness testing, deployment demo. Specify compute and delivery estimates as estimates; no training or spending without explicit approval.

## Example: deepfake media detector

**Phase 1 (image):** investigate manipulated/real image collections; prevent identity and source leakage, evaluate compression and unseen manipulation families, compare CNN, vision transformer and specialized artifact approaches subject to verified literature.

**Phase 2 (audio):** research spoofed-speech corpora, speaker-separated splits, vocoder and codec variation; examine speech anti-spoofing baselines and benchmark-specific EER/t-DCF.

**Phase 3 (multimodal):** locate synchronized audio/video datasets, assess pairing and provenance; compare late fusion against cross-modal alignment; test missing modality and lip-sync inconsistencies. Never create false synchronized pairs from unrelated audio/image datasets.

**Optional self-training:** use confidence-filtered pseudo-labeling on separate unlabeled pools, manual audit, rollback criteria and untouched holdout evaluation.

Per phase deliver 5–12 papers, 3–5 datasets, 3–4 models, experimental protocol and go/no-go criteria, adjusted for actual literature and resource limitations.
