---
name: hypothesis-generation
description: Systematic methodology for formulating testable, falsifiable scientific hypotheses and experimental validation roadmaps. Guards against HARKing (Hypothesizing After Results are Known) and p-hacking. Use when defining research questions, proposing novel models/algorithms, designing ablation studies, or establishing POC go/no-go criteria.
---

# Scientific Hypothesis Generation & Experimental Framing

This skill establishes a rigorous framework for formulating falsifiable, theoretically grounded scientific hypotheses and designing pre-registered experimental validation protocols. It prevents post-hoc rationalization (HARKing) and ensures that all empirical investigations yield conclusive, reproducible findings.

---

## 1. The 5E Hypothesis Standard

Every proposed research hypothesis must satisfy the five foundational criteria:

1. **Explicit:** Stated with unambiguous operational definitions and quantitative directions.
2. **Evidence-Grounded:** Logically derived from gaps, inconsistencies, or theoretical foundations identified during the literature review.
3. **Ex-Ante:** Formulated and committed to prior to running tests or inspecting validation data (eliminating hindsight bias).
4. **Explanatory:** Articulates a clear causal mechanism explaining *why* the effect is expected to occur, not merely that a correlation might exist.
5. **Empirically Testable:** Capable of being conclusively supported or refuted with accessible data, observable metrics, and feasible compute budgets.

---

## 2. Hypothesis Specification Template

When generating hypotheses for a study or project, format each hypothesis using this specification:

```markdown
### Hypothesis [H-ID]: [Descriptive Title]

- **Theoretical Rationale:** [Brief explanation of the underlying scientific mechanism or architectural intuition.]
- **Independent Variable ($X$):** [What is intentionally manipulated, e.g., attention mechanism type, loss weighting, dataset diversity.]
- **Dependent Variable ($Y$):** [What is measured, e.g., cross-dataset AUROC, inference latency, sample efficiency.]
- **Controlled Variables ($Z$):** [Factors held constant across conditions, e.g., backbone architecture, optimizer, batch size, random seeds.]

- **Formal Formulations:**
  - **Null Hypothesis ($H_0$):** [State of no effect, e.g., "Replacing standard spatial convolution with cross-spectral attention yields no statistically significant difference in cross-dataset AUROC ($\Delta \text{AUROC} \le 0$)."]
  - **Alternative Hypothesis ($H_1$):** [Directional prediction, e.g., "Cross-spectral attention improves cross-dataset AUROC by at least 3.0 percentage points over the baseline spatial convolution model on the unseen benchmark test set."]

- **Falsification & Failure Conditions (Pre-defined):**
  - If $\Delta \text{AUROC} < 3.0\%$ or if latency increases by $> 50\%$, $H_1$ is refuted.
  - If performance gains disappear when evaluated across unseen compression codecs, the generalization claim is considered falsified.

- **Experimental Protocol:**
  - *Dataset / Splits:* [Specify exact benchmarks and leak-free split protocols.]
  - *Statistical Significance Test:* [e.g., Paired t-test / Wilcoxon signed-rank test across 5 random seeds with $\alpha = 0.05$.]
  - *Ablation Matrix:* [List specific component toggle experiments.]
```

---

## 3. Safeguards Against Common Research Biases

- **Anti-HARKing Discipline:** Never modify the original hypothesis after seeing experimental results. If surprising or contradictory patterns emerge, document the original hypothesis as refuted, and report the new observation as an *exploratory finding* that generates a subsequent hypothesis for future testing.
- **Confounding Variable Control:** Actively search for hidden variables that could mimic the expected outcome (e.g., differences in parameter count, pretraining data exposure, batch size discrepancies, or data leakage across train/test splits).
- **Leakage-Safe Validation:** Guard against subtle leakage (such as identical speakers, video clips, temporal timestamps, or synthetic generators shared between training and test sets).
- **Go / No-Go Decision Gates:** Define quantitative thresholds that determine whether a Proof of Concept (POC) proceeds to the next developmental phase or is pivoted.

