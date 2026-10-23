# AI/ML Research Assistant for Google Antigravity

An installable Agent Skill for finding AI/ML projects, papers, datasets, implementation code, model candidates, research gaps, and resource-aware POC plans.

## Installation

1. Extract `ai-ml-research-assistant.zip`.
2. Copy the extracted `ai-ml-research-assistant` directory into your Google Antigravity project's `.agents/skills/` folder.
3. Confirm the file is at `.agents/skills/ai-ml-research-assistant/SKILL.md`.
4. Open the project in Antigravity and describe an AI/ML research task. The skill should activate when relevant; you can explicitly ask to use it.

Official documentation: https://antigravity.google/docs/skills

## Test prompt

Use the ai-ml-research-assistant skill. Research a deepfake media detection POC in three phases: image, then audio, then multimodal. Find verified papers and datasets, shortlist 3–4 models per phase, identify research gaps, compare compute/implementation feasibility, design leakage-safe evaluation, and propose carefully gated self-training. Cite sources and distinguish evidence from assumptions.

## Prerequisites and limitations

No Python dependencies are needed for the instruction-only skill. Live literature and dataset research requires Antigravity to have web browsing or appropriately configured search/API tools. The skill does not itself confer access to restricted resources, paid GPU services or external accounts.

## Files

- `SKILL.md`: skill trigger and end-to-end requirements.
- `references/sources.md`: discovery source list.
- `references/evaluation.md`: research and experimental safeguards.
- `references/research-workflows.md`: task-specific playbooks and deepfake example.
