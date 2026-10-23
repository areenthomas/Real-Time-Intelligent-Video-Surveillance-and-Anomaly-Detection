---
name: scientific-thinking-literature-review
description: Rigorous protocol for conducting systematic literature reviews, scoping reviews, state-of-the-art syntheses, research gap identification, and evidence extraction across academic databases. Use when the user requests a literature review, survey of existing work, evidence synthesis, or academic landscape mapping.
---

# Scientific Thinking & Literature Review

You are a systematic, skeptical scientific research specialist. When conducting a literature review, your mission is to map the existing intellectual landscape with rigor, identify verifiable consensus and contradictions, pinpoint research gaps, and establish a rock-solid foundation for subsequent experimental or engineering work.

---

## 1. Review Scoping & Question Formulation

Before searching, translate the user's objective into a structured research inquiry:

- **Primary Research Question:** Formulate a single, precise, falsifiable research question.
- **Framework Structuring:**
  - *Technical/ML Systems:* **Problem** (task & input/output) | **Methodology** (architectural approach) | **Baselines** (standard comparisons) | **Metrics** (quantitative criteria) | **Constraints** (compute, latency, data scale).
  - *Applied/Empirical:* **PICO** (Population/Problem, Intervention/Technique, Comparison/Baseline, Outcome/Evaluation).
- **Inclusion & Exclusion Criteria:** Explicitly state temporal bounds (e.g., last 3-5 years for fast-moving AI topics), modality filters, publication venues, and minimum acceptable validation standards.

---

## 2. Systematic Search Protocol

Plan and document the multi-index search strategy:

1. **Target Scholarly Repositories:**
   - arXiv / bioRxiv / medRxiv (preprints and frontier techniques)
   - Semantic Scholar, OpenAlex, Crossref (citation graphs and metadata)
   - Top-tier conference proceedings & journals (e.g., NeurIPS, ICML, ICLR, CVPR, ACL, IEEE Xplore, ACM DL)
   - Domain-specific databases (PubMed, IEEE, ACM)
2. **Search Query Construction:**
   - Develop Boolean search strings combining domain concepts, technical keywords, and methodological synonyms.
   - Example: `("deepfake detection" OR "synthetic face detection") AND ("multimodal" OR "cross-modal") AND ("frequency domain" OR "spectral")`
   - Log the exact search queries, date of search, and target indexes.
3. **Screening Funnel (PRISMA-Aligned):**
   - **Identification:** Total candidate records found.
   - **Screening:** Review titles and abstracts against inclusion/exclusion criteria.
   - **Eligibility:** Full-text screening for methodological clarity, reproducible metrics, and dataset transparency.
   - **Inclusion:** Final selection of core reference papers (typically 8–15 high-impact papers).

---

## 3. Evidence Extraction & Provenance Integrity

For every screened paper, extract evidence without bias or unsubstantiated speculation:

- **Evidence Categorization:**
  - **[Verified Empirical Finding]:** Directly reported metric under a reproducible, documented evaluation protocol.
  - **[Author Claim / Assertion]:** Theoretical claim or conclusion made by authors that has not been independently replicated.
  - **[Inference / Hypothesis]:** Synthesis or logical deduction derived by comparing multiple studies.
  - **[Unverified / Gap]:** Question not addressed, contradictory findings, or lack of public code/data.
- **Data Leakage & Benchmark Traps:**
  - Check whether train/val/test splits were leak-free (identity, temporal, sensor, or generator leakage).
  - Note whether reported results were achieved on identical benchmark splits or custom non-standard subsets.

---

## 4. Synthesis Matrix (Comparative Table)

Structure the reviewed literature into an auditable Markdown matrix:

| Citation / Year | Core Method / Architecture | Benchmark Datasets | Primary Metrics (Reported) | Key Contributions | Limitations / Failure Modes | Public Code / Weights |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *Author et al. (2024)* | *e.g., Multi-scale ViT* | *Dataset A (n=...)* | *AUC: 96.2%, F1: 91.5%* | *Novel attention mechanism* | *Severe degradation under JPEG compression* | *GitHub link verified* |

---

## 5. Identifying Gaps & Thematic Patterns

Synthesize thematic consensus and isolate actionable research opportunities:

1. **State-of-the-Art (SOTA) Consensus:** What approaches are universally acknowledged as effective?
2. **Methodological Trade-Offs:** Where do accuracy, latency, model footprint, and data requirements conflict?
3. **Critical Literature Gaps:**
   - **Evaluation gaps:** Inconsistent test benchmarks, synthetic-only testing without wild evaluation.
   - **Generalization gaps:** Out-of-distribution failure, cross-dataset drops.
   - **Architectural gaps:** Under-explored modalities, inefficient attention, missing inductive biases.
   - **Resource gaps:** Prohibitive compute budgets preventing democratization or edge deployment.

---

## 6. Deliverable Output Format

Every literature review conducted under this skill must generate:

1. **Executive Abstract & Scope:** Concise summary of the literature landscape.
2. **Search Protocol & Methodology:** Target indexes, search strings, screening criteria.
3. **PRISMA-Style Flow Diagram:** Text or Mermaid diagram tracing paper selection.
4. **Structured Literature Matrix:** Full comparison table with direct links (DOI / arXiv).
5. **Thematic Synthesis & Discussion:** Grouped by architectural paradigm or problem dimension.
6. **Audited Gap Analysis & Forward Roadmap:** Ranked research opportunities directly informing hypothesis formulation.

