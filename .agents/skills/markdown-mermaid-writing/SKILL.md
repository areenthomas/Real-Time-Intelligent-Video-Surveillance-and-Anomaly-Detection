---
name: markdown-mermaid-writing
description: Standards and templates for authoring clean, version-controlled technical and scientific Markdown documentation enriched with native Mermaid diagrams. Use when creating technical specifications, research documentation, architecture designs, data pipeline flowcharts, or system state machines.
---

# Markdown & Mermaid Technical Writing

This skill defines the technical standards, stylistic conventions, and diagramming best practices for creating reproducible, Git-diffable, publication-grade Markdown documentation with embedded Mermaid visualizations.

---

## 1. Markdown Structuring Guidelines

- **Semantic Hierarchy:** Use consistent `#` header levels without skipping ranks (`#` -> `##` -> `###`).
- **Mathematical Equations:** Render formulas using LaTeX/KaTeX:
  - Inline equations: `$E = mc^2$` or `\( \mathcal{L}_{\text{total}} \)`
  - Block equations on dedicated lines:
    $$
    \mathcal{L}_{\text{CE}} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
    $$
  - Literal dollar signs in prose must be escaped (`\$100`) or backtick-wrapped (`$USD`) to avoid triggering math parsing.
- **GitHub-Style Callouts:** Emphasize operational notes and warnings:
  ```markdown
  > [!NOTE] Background context or implementation details.
  > [!IMPORTANT] Essential requirements or non-negotiable constraints.
  > [!WARNING] Potential pitfalls, data leakage risks, or stability issues.
  ```
- **Auditable Tables:** Align table columns cleanly with descriptive header labels and consistent units.

---

## 2. Mermaid Diagramming Standards

Mermaid enables text-based, version-controlled visual models directly inside Markdown code blocks (` ```mermaid `).

### Supported Diagram Types & Archetypes:

1. **System & Data Pipelines (`flowchart LR` / `flowchart TD`):**
   ```mermaid
   flowchart LR
       subgraph Ingestion["Data Ingestion"]
           A["Raw Audio / Video"] --> B["Pre-processing & Normalization"]
       end
       subgraph Modeling["Feature Extraction"]
           B --> C["Spectrogram STFT"]
           B --> D["Spatial Face Cropping"]
       end
       subgraph Fusion["Multimodal Fusion"]
           C --> E["Cross-Attention Transformer"]
           D --> E
       end
       E --> F{"Classification Head"}
       F -->|"Score > 0.5"| G["Deepfake Detected"]
       F -->|"Score <= 0.5"| H["Authentic Media"]
   ```

2. **Temporal & Protocol Interactions (`sequenceDiagram`):**
   ```mermaid
   sequenceDiagram
       autonumber
       actor Client as Client App
       participant Gateway as API Gateway
       participant Worker as ML Inference Worker
       participant Storage as Model / Cache Store

       Client->>Gateway: POST /detect (media_url)
       Gateway->>Worker: Dispatch job (task_id)
       Worker->>Storage: Fetch cached embeddings
       Storage-->>Worker: Return embeddings
       Worker->>Worker: Run ensemble inference
       Worker-->>Gateway: Result (confidence: 0.982)
       Gateway-->>Client: JSON {status: "fake", score: 0.982}
   ```

3. **Lifecycle & State Transitions (`stateDiagram-v2`):**
   ```mermaid
   stateDiagram-v2
       [*] --> Initialized
       Initialized --> DataLoading : Load Dataset Splits
       DataLoading --> Training : Start Epoch
       Training --> Validating : Epoch Complete
       Validating --> Training : Loss Decreasing
       Validating --> Checkpointing : New Best Metric
       Validating --> EarlyStopping : Patience Exceeded (Early Exit)
       Checkpointing --> Training
       EarlyStopping --> Evaluated
       Evaluated --> [*]
   ```

4. **Entity-Relationship & Data Schemas (`erDiagram`):**
   ```mermaid
   erDiagram
       DATASET ||--o{ SAMPLE : contains
       SAMPLE ||--o{ ANNOTATION : has
       SAMPLE {
           string sample_id PK
           string modality
           string file_path
           int duration_ms
       }
       ANNOTATION {
           string annot_id PK
           string sample_id FK
           string label
           float bounding_box
       }
   ```

---

## 3. Mermaid Syntax Guardrails & Safety Rules

To prevent compilation or rendering errors across markdown engines:
- **Always Quote Labels with Special Characters:** Parentheses, brackets, hyphens, and slashes will break Mermaid parsing unless enclosed in double quotes:
  - ❌ `A[Convolution (Kernel 3x3)] --> B`
  - ✅ `A["Convolution (Kernel 3x3)"] --> B`
- **Avoid Raw HTML Inside Labels:** Do not use `<b>`, `<br>`, or `<span>` tags unless strictly supported; prefer standard clean strings.
- **Directional Clarity:** Specify orientation clearly (`TD` for top-down hierarchies; `LR` for horizontal sequential pipelines).
- **Subgraphs for Modularity:** Group logically related components inside named `subgraph Name["Display Title"] ... end` blocks.
- **Textual Fallback:** Accompany every major architectural or pipeline diagram with 1–2 paragraphs of explanatory text detailing data flow and component roles.

