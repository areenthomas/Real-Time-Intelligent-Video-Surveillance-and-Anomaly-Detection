---
name: scientific-writing
description: Protocol for authoring, auditing, and revising peer-review quality scientific manuscripts, technical reports, and major project theses. Enforces IMRaD structure, academic tone, evidence-grounded prose, and reporting guidelines. Use when writing academic papers, research project chapters, methodology sections, or thesis manuscripts.
---

# Scientific Writing & Manuscript Authoring

This skill governs the composition, revision, and auditing of academic manuscripts, scientific technical reports, conference papers, and university major project theses. It enforces structured scholastic prose, rigorous empirical reporting, and uncompromising evidence integrity.

---

## 1. Tone, Style & Prose Principles

- **Formal Academic Register:** Write with precision, clarity, and conciseness. Avoid conversational filler, colloquialisms, and anthropomorphic phrasing (e.g., write *"The model achieves..."* or *"The experimental results show..."* rather than *"The model thinks that..."*).
- **No Marketing Hype or Superlatives:** Eliminate hyperbolic adjectives such as *"revolutionary"*, *"game-changing"*, *"groundbreaking"*, or *"unprecedented"*. Quantify achievements objectively using validated benchmark metrics (e.g., *"improves F1-score by 4.2 percentage points over baseline [2]"*).
- **Paragraph Cohesion Over Bullet Lists:** Core manuscript sections (Introduction, Background, Methodology, Discussion) must be authored in fully developed, coherent paragraphs with topic sentences and logical transitions. Reserve bullet points strictly for itemizing experimental hyperparameters, formal contributions, or categorical survey lists.
- **Active vs. Passive Balance:** Use active voice when describing researcher decisions (*"We evaluated three architectures..."*) and passive voice when describing standard physical or mathematical procedures (*"The dataset was partitioned using stratified sampling..."*).

---

## 2. Standard Scientific Manuscript Structure (IMRaD+)

### Title & Abstract
- **Title:** Direct, informative, and indicative of the core contribution and domain.
- **Abstract (200–250 words):** Structured into 5 distinct movements:
  1. *Context/Motivation:* The broader domain challenge (1 sentence).
  2. *Problem/Gap:* Specific limitation in current approaches (1–2 sentences).
  3. *Proposed Method:* Architectural or theoretical contribution (1–2 sentences).
  4. *Key Results:* Concrete quantitative benchmark outcomes (1–2 sentences).
  5. *Implications:* Significance and broader impact (1 sentence).

### 1. Introduction
- **The Funnel Structure:** Begin broad (domain relevance), narrow to the specific unresolved problem, present the thesis statement/solution, itemize major contributions (typically 3 bulleted points), and outline the paper organization.

### 2. Related Work / Background
- Organize thematically by methodological paradigm rather than chronologically listing authors.
- Critically contrast approaches, culminating in the explicit identification of the gap that the current work resolves.

### 3. Methodology / System Architecture
- Provide reproducible, mathematical, and algorithmic descriptions.
- Define all notation formally.
- Detail data preprocessing, feature representations, architectural layers, objective/loss functions, and training regimes (optimizers, learning rates, hardware parameters).
- Embed high-level system diagrams (flowcharts/sequence models) using clean Mermaid syntax.

### 4. Experimental Setup & Results
- **Datasets:** Detail sample counts, class balance, split strategy (leak-free train/val/test), and licenses.
- **Baselines:** Compare against standard, competitive, and state-of-the-art implementations under identical evaluation protocols.
- **Metrics:** Report primary and secondary metrics (e.g., Accuracy, Precision, Recall, F1-score, AUROC, latency, parameter count) with confidence intervals or error bars where appropriate.
- **Ablation Studies:** Systematically isolate individual components or hyperparameters to demonstrate their distinct contribution.

### 5. Discussion & Limitations
- Interpret what the results mean theoretically and practically.
- Address unexpected behaviors, failure cases, and edge scenarios.
- Include a dedicated **Threats to Validity / Limitations** subsection covering dataset biases, compute constraints, and generalization bounds.

### 6. Conclusion & Future Work
- Reiterate contributions without simply copying the abstract.
- Propose concrete, falsifiable extensions for future researchers.

---

## 3. Evidence Grounding & Manuscript Auditing Checklist

Before finalizing any section or document, execute this verification pass:

1. **Claim Integrity:** Is every factual statement, historical claim, or baseline comparison backed by an empirical result or a verified citation?
2. **Reproducibility Completeness:** Could an independent researcher replicate the experiments using solely the methodology description and hyperparameter tables?
3. **Metric Consistency:** Are metrics reported with consistent precision (e.g., 2 decimal places: `94.35%`), identical test splits, and verified baseline numbers?
4. **Notation Uniformity:** Are mathematical symbols, variable names, and vectors defined prior to first use and maintained consistently throughout?

