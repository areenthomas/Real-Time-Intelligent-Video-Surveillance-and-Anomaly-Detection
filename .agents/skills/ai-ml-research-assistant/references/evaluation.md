# Quality and reproducibility standard

## Paper evidence register

`paper_id | title | authors | year | venue_or_preprint | url | method | datasets | metric_and_protocol | result | limitation | official_code | verified_at | evidence_level`

Evidence level: full-text verified / abstract verified / metadata only / unverified. Do not supply fabricated values. Different dataset splits, class distributions or metrics cannot be compared directly.

## Dataset audit

`name | original_url | sample_count | modality | labels | license | download_status | split | class_balance | privacy | leakage | domain_match`

Check duplicates and overlaps by identity, source video, speaker, recording session, generator and time as appropriate. Preserve untouched held-out evaluation data.

## Model comparison

`model | original_paper | code | weights | baseline_role | training_data | reported_protocol | score | compute | license | reproducibility`

Report scores only with protocol, and disclose missing data.

## Evaluation protocol

1. Set primary metric and acceptance threshold before training.
2. Benchmark a trivial/cheap baseline first.
3. Fix splits, seeds and preprocessing for model comparisons.
4. Use precision, recall, macro-F1, confusion matrix, ROC-AUC/PR-AUC and calibration as suitable; for spoofed speech, consider EER and t-DCF if benchmark requires.
5. Evaluate unseen domains, classes, compression and quality changes when relevant.
6. Track model size, runtime, peak memory, environment and energy/cost where possible.
7. Repeat runs or bootstrap uncertainty intervals when feasible.
8. Use ablations and error slicing to test the hypothesized innovation.

## Gap criteria

Every proposed gap must show observed limitation → evidence citation → counterexample search → hypothesis → falsifying experiment. Avoid claiming 'first ever' without substantiation.

## Self-training

Teacher predictions can label only a separate authorized, unlabeled training pool. Record confidence thresholds, selection rate, class distribution, auditing outcomes, model versions and drift. Never pseudo-label validation or final test. Require measurable holdout improvement before deployment.
