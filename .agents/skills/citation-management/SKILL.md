---
name: citation-management
description: Comprehensive reference tracking, bibliography auditing, metadata validation, and BibTeX/CSL generation. Enforces zero citation hallucination and ensures every claim has verified source attribution. Use when managing references, validating DOIs/arXiv IDs, generating BibTeX entries, or formatting citations.
---

# Citation Management & Bibliography Integrity

Accurate attribution and reference verification are foundational to academic and technical integrity. This skill establishes rigorous rules for discovering, formatting, validating, and maintaining bibliographic entries, citations, and reference catalogs.

---

## 1. Zero-Tolerance Citation Hallucination Policy

- **Never Fabricate Metadata:** Never guess, autocomplete, or invent DOIs, arXiv IDs, author lists, publication years, venue titles, volume numbers, or page counts.
- **Verification Requirement:** Every cited paper must correspond to a verified, published paper or preprint indexed in recognized academic databases (CrossRef, OpenAlex, Semantic Scholar, arXiv, DBLP, PubMed).
- **Missing Information Disclosure:** If a specific field (such as publication date, issue number, or direct DOI) cannot be verified from available tools or web search, explicitly flag the missing field with `[Unverified]` or omit the optional tag rather than guessing.

---

## 2. Standardized BibTeX Formatting

All BibTeX (`.bib`) entries must adhere to standard schema specifications:

```bibtex
@article{vaswani2017attention,
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  title     = {Attention Is All You Need},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017},
  url       = {https://arxiv.org/abs/1706.03762},
  doi       = {10.48550/arXiv.1706.03762}
}
```

### BibTeX Key & Syntax Rules:
- **Citation Key Format:** Use `[FirstAuthorSurname][Year][FirstMeaningfulTitleWord]` in lowercase (e.g., `he2016deep`, `devlin2018bert`, `brown2020language`).
- **Author Names:** Separate all authors with `and`. Format as `Surname, Given Names` or `Given Names Surname`.
- **Title Capitalization:** Protect domain-specific proper nouns, acronyms, and formulas with curly braces `{}` (e.g., `{BERT}`, `{Transformer}`, `{COVID-19}`, `{PyTorch}`).
- **Preprints:** When citing preprints, set `archivePrefix = {arXiv}`, `eprint = {XXXX.XXXXX}`, and include the direct URL.

---

## 3. Supported In-Text Citation Styles

When formatting citations in reports, manuscripts, or markdown, follow the user's requested style or default to IEEE/APA standards:

| Style | In-Text Citation Example | Reference List Entry Format |
| :--- | :--- | :--- |
| **IEEE (Numeric)** | "...as demonstrated in recent benchmarks [1], [2]..." | `[1] A. Vaswani et al., "Attention is all you need," in Adv. Neural Inf. Process. Syst., vol. 30, 2017.` |
| **APA 7th (Author-Date)** | "...is consistent with prior findings (Vaswani et al., 2017)..." | `Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.` |
| **ACM** | "...as detailed by Vaswani et al. [2017]..." | `[1] Ashish Vaswani et al. 2017. Attention is all you need. In Proc. NeurIPS '17.` |
| **Nature / Science** | "...novel attention-based architectures¹'²..." | `1. Vaswani, A. et al. Attention is all you need. Adv. Neural Inf. Process. Syst. 30 (2017).` |

---

## 4. Claim-to-Citation Auditing Workflow

Whenever reviewing a draft or generating scientific prose:

1. **Sentence-Level Audit:** Scan for empirical claims, statistical assertions, benchmark scores, or comparisons.
2. **Attribution Check:** Ensure every substantive claim is directly accompanied by its supporting citation.
3. **Contextual Alignment:** Verify that the cited paper actually supports the specific claim being made (prevent "citation diversion" or citing a paper that contradicted or did not measure the claimed property).
4. **Distinguish Secondary vs. Primary Sources:** Cite the original research paper that introduced a dataset, algorithm, or metric, rather than a secondary paper that merely mentioned it.

---

## 5. Reference Library Maintenance

When organizing references for a project:
- Maintain a centralized `references.bib` file in the project workspace root or docs folder.
- Deduplicate entries across different citation keys.
- Check for updated publication venues when converting preprints that have since been formally published in peer-reviewed journals or conferences.

