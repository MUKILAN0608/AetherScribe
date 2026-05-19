# AetherScribe Lab

### Quantum Narrative Intelligence · Master Render Edition (v6.0)

**AetherScribe Lab** is a research platform for **explainable, multi-agent narrative generation** driven by **Quantum Reinforcement Learning (QRL)** and **large language models (LLMs)**. It resolves *what happens* in each scene with a **Variational Quantum Circuit (VQC)** before any prose is written, then renders locked actions into publication-style scenes and surfaces a full audit trail: measured paths, alternate branches, posteriors, trait influence, and counterfactual rationale.

This repository (`quantum_rl`) contains the Streamlit application, quantum policy, environment, LLM renderer, and research dashboard. The interactive UI runs locally via `streamlit run app.py`.

**Stack:** [PennyLane](https://pennylane.ai/) · [Streamlit](https://streamlit.io/) · [Google Gemini](https://ai.google.dev/) (with graceful fallback when no API key is set)

---

## Table of contents

- [About AetherScribe](#about-aetherscribe)
- [Research highlights](#research-highlights)
- [Triad of authority](#triad-of-authority)
- [What AetherScribe does per scene](#what-aetherscribe-does-per-scene)
- [System architecture](#system-architecture)
- [Quantum explainability panel](#quantum-explainability-panel)
- [Understanding posteriors](#understanding-posteriors)
- [AetherScribe Lab UI](#aetherscribe-lab-ui)
- [Manuscript and exports](#manuscript-and-exports)
- [Dashboard telemetry](#dashboard-telemetry)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Episode pipeline](#episode-pipeline)
- [Project structure](#project-structure)
- [Development and verification](#development-and-verification)
- [Troubleshooting](#troubleshooting)
- [Citation and further reading](#citation-and-further-reading)

---

## About AetherScribe

Traditional story generators let an LLM freely choose plot beats, which makes decisions opaque and hard to reproduce. **AetherScribe** inverts that order:

1. **Quantum policy first** — A hybrid quantum–classical policy outputs a probability distribution over **joint actions** (protagonist, antagonist, and ally act simultaneously).
2. **Measurement** — One joint action is drawn from that distribution and **locked** for the scene.
3. **Rendering second** — The narrative engine writes 5–8 sentences that must respect the locked actions and premise.
4. **Explainability third** — Every step records *why* the measured path was taken and *why* notable alternates were not.

That design supports academic and industry use cases where stakeholders need **traceable creative AI**: reproducible trajectories, counterfactual branches, and charts that map quantum diagnostics to character names the author chose.

**Design goals**

| Goal | How AetherScribe achieves it |
|------|------------------------------|
| **Transparency** | Decisions are logged before LLM rendering; quantum panel shows posteriors and alternates. |
| **Narrative integrity** | Prose is conditioned on measured actions, not free-form LLM plotting. |
| **Reproducibility** | Episode logs store `probs`, `action_idx`, weights, tension, and rejections per scene. |
| **Human-readable XAI** | Explanation engine + UI translate VQC output into plain English and structured cards. |

---

## Research highlights

- **Variational Quantum Circuit (VQC) policy** — 4-qubit narrative state encoding with character trait biasing and trainable rotations (`quantum_policy/quantum_policy.py`).
- **Joint action space** — Cartesian product of per-role actions; one index selects a synchronized triad of character moves (`quantum_policy/quantum_action_space.py`).
- **Stochastic measurement** — Stories sample from the policy distribution (not silent argmax), with UI labels that prevent misreading peak vs selected posterior.
- **Per-scene quantum decision cards** — Chosen path, policy rationale, character influence bars, top alternates, and full “why not chosen” paragraphs (`ui/quantum_panel.py`, `ui/decisions.py`).
- **Research dashboard** — Plotly telemetry: tension, coherence, utility, entropy, risk, attribution, bonds, control heatmap, trajectory embedding (`ui/research_views.py`, `ui/charts.py`).
- **Premise-locked manuscript** — Full-story compilation plus Markdown/LaTeX/plain-text export helpers (`ui/manuscript.py`).
- **Entity-aware naming** — Detect character names from the premise (Gemini extraction or heuristics); sync names across prose, tables, and charts (`llm/entity_extractor.py`, `ui/seed_analysis.py`).
- **Audit logging** — Research runs append to `logs/research_audit.log` under the `AetherScribeLab` logger (`architecture_utils.py`).

---

## Triad of authority

AetherScribe separates responsibilities so no single component can silently override the others.

### A) Decision authority — Quantum RL

| Item | Detail |
|------|--------|
| **Component** | `QuantumPolicy`, `QuantumActionSpace`, `EpisodeRunner` |
| **Library** | PennyLane (hybrid quantum–classical) |
| **Input** | Encoded narrative state + combined trait vector |
| **Output** | Action probabilities, entropy, risk, attribution, sensitivity |
| **Rule** | Plot branch is chosen **before** scene text is generated |

The VQC uses three conceptual layers: **state encoding** (tension, phase, progress, genre on four qubits), **trait biasing** (protagonist/antagonist/ally traits as rotations), and **variational weights** updated from scene rewards.

### B) Rendering authority — LLM screenwriter

| Item | Detail |
|------|--------|
| **Component** | `StoryRenderer` (`llm/story_renderer.py`) |
| **Primary API** | Google Gemini (`GOOGLE_API_KEY` in `.env`) |
| **Input** | Locked joint action, rationale, prior summary, director brief |
| **Output** | Scene prose (5–8 sentences), running summary, coherence score |
| **Rule** | Must adhere to the quantum-resolved action set; cannot invent a different branch |

Coherence scores feed back into the composite reward, anchoring learning to narrative quality as well as environment utility.

### C) Explainability authority — Interpretation layer

| Item | Detail |
|------|--------|
| **Component** | `ExplanationEngine`, `ui/decisions.py`, `ui/quantum_panel.py` |
| **Input** | VQC diagnostics, rejections, attribution, agents |
| **Output** | Policy rationale, alternate paths, influence breakdown, narrative trace |
| **Rule** | Technical traces are distilled for the UI; alternates always include grounded “why not” text |

---

## What AetherScribe does per scene

For each scene index `step` in an episode:

1. **Encode** — `StateRepresentation.encode(phase, tension, genre, step)` → classical vector fed to the VQC.
2. **Diagnose** — `QuantumPolicy.compute_step_diagnostics(...)` → `probs`, entropy, risk, attribution, sensitivity.
3. **Measure** — `np.random.choice(len(probs), p=probs)` → `action_idx` and joint action dict.
4. **Reject** — Build sorted `rejection_data` for other branches (with template suppression reasons).
5. **Explain** — `ExplanationEngine.generate_explanation(...)` → multi-line rationale.
6. **Enrich** (optional) — `StoryRenderer.analyze_rejections(...)` when `render_depth="full"`.
7. **Render** — `StoryRenderer.render_scene(...)` → story block + coherence.
8. **Learn** — Policy `update(reward, state, action_idx)` using environment utility × coherence.
9. **Log** — Append full scene record for Manuscript and Dashboard tabs.

The deliverable is a **complete manuscript** plus an **auditable quantum trace** per scene.

---

## System architecture

```
Director brief (premise, genre, character names)
        │
        ▼
┌───────────────────┐     ┌─────────────────────┐
│  StoryEnvironment │◄────│  QuantumActionSpace │
│  (narrative physics)     │  (joint action grid)│
└─────────┬─────────┘     └─────────────────────┘
          │
          ▼
┌───────────────────┐     ┌─────────────────────┐
│   QuantumPolicy   │────►│ ExplanationEngine   │
│   (PennyLane VQC) │     │ (AetherScribe XAI)  │
└─────────┬─────────┘     └─────────────────────┘
          │
          ▼
┌───────────────────┐     ┌─────────────────────┐
│  EpisodeRunner    │────►│   StoryRenderer     │
│  sample + log     │     │   (Gemini / fallback)│
└─────────┬─────────┘     └─────────────────────┘
          │
          ▼
   AetherScribe Lab UI — Manuscript · Dashboard
```

### State encoding (4 qubits)

| Feature | Role |
|---------|------|
| **Tension** | Dramatic intensity in \([0,1]\); drives urgency of actions |
| **Phase** | Narrative arc position (intro / rising / climax) |
| **Progress** | `step / max_steps`; pacing toward resolution |
| **Genre** | Thriller, horror, drama, etc.; modulates dynamics |

### Agents and traits

Three agents—**protagonist**, **antagonist**, **ally**—each carry trait matrices (`agents/`). Traits are aggregated into a combined vector that **biases** the quantum circuit, so characters pull the distribution toward disposition-aligned joint actions.

### Reward signal

\[
\text{composite\_reward} = \text{env\_reward} \times \text{coherence\_score}
\]

Environment reward reflects narrative physics (tension, relationships); coherence is judged by the LLM auditor on how well prose matches locked actions.

---

## Quantum explainability panel

Open **Manuscript → Scenes & quantum trace** and expand any scene. The **Quantum decision** card is generated from logged episode data—not static placeholder text.

### Panel sections

| Section | Description |
|---------|-------------|
| **Legend** | How to read posteriors; clarifies that a high alternate % is not a UI bug under stochastic sampling. |
| **Metrics strip** | Selected branch P, peak branch P, entropy, risk, margin ΔP, coherence. |
| **Sampling note** | Shown when the measured branch’s posterior is below the VQC peak; explains stochastic draw vs distribution mode. |
| **Chosen path (measured)** | Joint action actually written into the story (character names from your brief). |
| **Policy rationale** | Measurement context, trait-channel attribution, dominant disposition, narrative phase. |
| **Character influence** | Normalized influence bars per character name. |
| **Other paths not chosen** | Up to three alternates (posterior ≥ 5% threshold in logging), each with a full **why not chosen** paragraph. |
| **Narrative trace** | One grammatical English sentence summarizing the measured path. |

### Metric definitions

| Metric | Meaning |
|--------|---------|
| **Selected branch P** | Posterior of the **sampled** joint action used in the scene. |
| **Peak branch P** | Maximum posterior over **all** joint actions in the superposition. |
| **Entropy** | Shannon entropy of the action distribution. |
| **Risk** | Policy risk diagnostic from the VQC step. |
| **Margin ΔP** | Gap between largest and second-largest posterior (decision sharpness). |
| **Coherence** | LLM score for prose ↔ measured action alignment. |

### “Why this path was not chosen”

Each alternate card includes text that:

- Names the alternate’s posterior and joint actions.
- Contrasts them with the **measured** path and its posterior.
- If the alternate’s posterior **exceeds** the measured branch’s, states explicitly that AetherScribe uses **random sampling** from the full distribution, not argmax.
- Appends VQC-grounded template reasons (trait misalignment, low mass, utility gap, etc.).
- Merges optional LLM rejection prose only when it passes quality checks (minimum length, no truncation artifacts).

Implementation: `build_scene_decision_record()` in `ui/decisions.py`, rendered by `render_quantum_explain_panel()` in `ui/quantum_panel.py`.

---

## Understanding posteriors

**Frequently asked:** *Why does the chosen path show 13% posterior while an alternate shows 49%?*

**Answer:** That is **correct AetherScribe behavior**, not corrupted data.

```python
# evaluation/episode_runner.py
action_idx = int(np.random.choice(len(probs), p=probs))
```

The story follows a **stochastic measurement** of the VQC distribution. The **peak** branch is the mode of the superposition; the **measured** branch is the single draw used for rendering. They often differ—similar to probabilistic collapse rather than always selecting the highest-probability outcome.

The UI communicates this through:

- **Selected branch P** vs **Peak branch P**
- A highlighted **sampling note** when the gap is meaningful
- Alternate cards that explain higher-posterior branches that were not selected

For deterministic experiments (always pick the mode), replace sampling with `int(np.argmax(probs))` in `episode_runner.py` and document that change in any publication, since it removes exploration behavior.

---

## AetherScribe Lab UI

Launch the lab:

```bash
streamlit run app.py
```

The browser title may show **Narrative Studio**; the research branding in logs and documentation is **AetherScribe Lab**.

### Sidebar — director brief

| Control | Purpose |
|---------|---------|
| **Premise** | Story setup; include character names for best extraction and coherence. |
| **Detect names from premise** | Runs `analyze_seed()` → Gemini entity extraction when `GOOGLE_API_KEY` is set, else heuristics. |
| **Characters** | Editable protagonist, antagonist, ally **display names** (propagate to prose and charts). |
| **Genre** | Thriller, Horror, Comedy, Romance, Drama, Fantasy. |
| **Language** | English, Spanish, French, German, Japanese. |
| **Number of scenes** | 6–12 scenes per episode. |
| **Generate** | Full episode at `render_depth="full"`. |
| **Connection status** | Whether the narrative engine can reach Gemini. |

### Manuscript tab

- **KPI strip** — Mean coherence, cumulative utility, tension, posterior/entropy/risk (extended mode on dashboard).
- **Full story** — Compiled manuscript from all scenes.
- **Scenes & quantum trace** — Per-scene prose + expandable **Quantum decision** panel.

### Dashboard tab

Research-grade Plotly charts for the active episode (see [Dashboard telemetry](#dashboard-telemetry)). All series use **your character names**, not internal role keys like `Protagonist`.

---

## Manuscript and exports

AetherScribe can assemble the generated episode into export-friendly formats (`ui/manuscript.py`):

| Function | Output |
|----------|--------|
| `compile_full_manuscript()` | Continuous story text for reading |
| `compile_plain_text_export()` | Plain-text bundle for archiving |
| `generate_manuscript_markdown()` | Markdown structured manuscript |
| `generate_manuscript_latex()` | LaTeX source for papers or reports |

Exports respect the **same measured trajectory** shown in the quantum trace—they do not rewrite plot decisions.

---

## Dashboard telemetry

The Dashboard tab renders an extended KPI grid and chart panels (titles centered via `chart_with_title()`). Typical visualizations:

| Row / theme | Charts |
|-------------|--------|
| **Narrative dynamics** | Dramatic tension, narrative coherence, scene utility, decision margin ΔP |
| **Quantum diagnostics** | Selected-branch posterior per scene, policy entropy, narrative risk |
| **Characters** | Trait pulse, final trait attribution (pie), relationship bonds (trust/enmity) |
| **Structure** | Control heatmap (influence over time), trajectory embedding (PCA when enough steps) |

Charts call `has_chart_data()` and show an empty-state message instead of blank panels when data is insufficient.

---

## Quick start

### Prerequisites

- Python **3.10+** (3.11 recommended)
- `pip`

### Clone and install

```bash
git clone https://github.com/MUKILAN0608/AetherScribe.git
cd AetherScribe   # or your local folder name, e.g. quantum_rl
pip install -r requirements.txt
```

### API key (strongly recommended)

Create `.env` in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Optional:

```env
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///./aetherscribe_lab.db
```

Without a valid `GOOGLE_API_KEY`, scene quality and entity detection are limited; the sidebar will warn that the narrative engine is not connected.

### Run AetherScribe Lab

```bash
streamlit run app.py
```

1. Enter a **premise** with named characters.  
2. Click **Detect names from premise** (optional).  
3. Set genre, language, and scene count.  
4. Click **Generate** and wait for all scenes (several minutes for 8–12 scenes at full depth).  
5. Review **Manuscript** and **Dashboard**.

**Important:** After updating explainability or UI code, run **Generate** again. Cached Streamlit session episodes do not pick up new formatting.

---

## Configuration

### Environment variables

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` | Gemini: scenes, coherence, rejection analysis, entity extraction |
| `LOG_LEVEL` | Logging verbosity (`architecture_utils.py`) |
| `DATABASE_URL` | SQLite path for trajectory archive (default `sqlite:///./aetherscribe_lab.db`) |

**Security:** Never commit `.env` to Git. Rotate keys if exposed.

### Render depth (`EpisodeRunner`)

Set in `evaluation/episode_runner.py` or when calling `run_episode()`:

| Mode | Behavior |
|------|----------|
| `full` | Scene + summary + coherence LLM + rejection LLM (default in `app.py`) |
| `standard` | Scene + summary + coherence; template-only rejections |
| `fast` | Scene only; heuristic coherence; template rejections |

### Default learning hyperparameters (`app.py`)

| Parameter | Default | Role |
|-----------|---------|------|
| `lr` | `0.01` | Policy weight update rate |
| `eps` | `0.01` | Exploration epsilon passed to diagnostics |
| `temperature` | `0.88` | LLM rendering temperature |

---

## Episode pipeline

Detailed step list for one scene:

```
state → encode → VQC diagnostics → sample action_idx
      → rejection_data → ExplanationEngine.rationale
      → [optional] analyze_rejections → render_scene
      → composite_reward → policy.update → log_entry
```

Each `log_entry` includes at minimum:

`step`, `state`, `action`, `action_idx`, `probs`, `entropy`, `risk`, `attribution`, `rejections`, `deep_rejections`, `sensitivity`, `coherence`, `rationale`, `story`, `summary`, `reward`, `weights`.

---

## Project structure

```
quantum_rl/                          # AetherScribe Lab codebase
├── app.py                           # Streamlit entry — AetherScribe Lab UI
├── architecture_utils.py            # .env loader, AetherScribeLab logging
├── agents/                          # Protagonist, antagonist, ally traits
├── environment/
│   ├── story_environment.py         # Narrative physics, relationships
│   └── state_representation.py      # 4-feature state encoding
├── quantum_policy/
│   ├── quantum_policy.py            # PennyLane VQC policy
│   ├── quantum_action_space.py      # Joint action enumeration
│   └── action_registry.py           # Action → environment mapping
├── explainability/
│   └── explanation_engine.py        # Research-tone rationales
├── llm/
│   ├── story_renderer.py            # Gemini scene + coherence pipeline
│   ├── prompt_templates.py          # Manuscript & analysis prompts
│   └── entity_extractor.py          # Premise → character names
├── evaluation/
│   ├── episode_runner.py            # Episode loop, stochastic sampling
│   └── database.py                  # Trajectory persistence (SQLite)
├── ui/
│   ├── research_views.py            # Manuscript + dashboard layouts
│   ├── quantum_panel.py             # Quantum decision HTML card
│   ├── decisions.py                 # Scene records, traces, why-not text
│   ├── charts.py                    # Plotly helpers
│   ├── manuscript.py                # Compile & export manuscript
│   ├── seed_analysis.py             # Premise / genre analysis
│   └── styles.py                    # Research dashboard CSS
├── experiments/
│   └── run_single_episode.py        # CLI trajectory runner
├── tests/
│   └── verify_system.py             # Integration smoke test
├── logs/
│   └── research_audit.log           # AetherScribeLab audit trail
├── TECHNICAL_SPEC.md                # Architecture specification
└── DETAILED_SYSTEM_FUNCTIONALITY.txt  # Full academic-style system doc
```

---

## Development and verification

```bash
# Integration test (AetherScribe Lab v5.4.2+ components)
python tests/verify_system.py

# CLI single-episode run (archives to aetherscribe_lab.db)
python experiments/run_single_episode.py
```

Smoke-test explainability builders:

```bash
python -c "from ui.decisions import build_scene_decision_record, build_name_map; print('ok')"
```

Audit logs: `logs/research_audit.log` (logger name `AetherScribeLab`).

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| Empty or very short scenes | Missing/invalid API key or token limits | Set `GOOGLE_API_KEY`; confirm sidebar **Narrative engine connected** |
| Selected P &lt; alternate P | Stochastic sampling (expected) | Read [Understanding posteriors](#understanding-posteriors); check sampling note |
| Truncated “why not chosen” | Old episode in session | Click **Generate** for a fresh run |
| Empty chart panels | No data yet or insufficient steps | Complete generation; check `has_chart_data` guards |
| Names wrong in charts | Brief not synced | Use **Detect names from premise** and regenerate |
| `GOOGLE_API_KEY missing` | No `.env` | Add key to project root `.env` |

---

## Citation and further reading

If you use **AetherScribe Lab** in academic work, cite the system and note whether actions were **sampled** or **argmax-selected**:

```bibtex
@software{aetherscribe2026,
  author = {Mukilan},
  title = {AetherScribe Lab: Quantum Narrative Intelligence System},
  year = {2026},
  version = {6.0},
  url = {https://github.com/MUKILAN0608/AetherScribe}
}
```

**In-repo documentation**

| Document | Contents |
|----------|----------|
| `TECHNICAL_SPEC.md` | Triad of authority, VQC overview, governance |
| `DETAILED_SYSTEM_FUNCTIONALITY.txt` | Full v6.0 academic-style system reference |

**Suggested citation sentence**

> We generated narratives using AetherScribe Lab (v6.0), a quantum reinforcement learning framework that resolves joint character actions via a variational quantum circuit before LLM rendering, with per-scene explainability over posteriors and counterfactual branches.

---

© 2026 **AetherScribe Research** · Master Render Edition v6.0 · Quantum Narrative Intelligence
