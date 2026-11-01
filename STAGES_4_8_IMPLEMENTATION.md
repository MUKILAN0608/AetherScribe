# AetherScribe Stages 4–8 implementation and verification

## Authority and execution path

The VQC still selects the action before narrative generation. Agent review is advisory until all existing prose, secret, continuity, and quality gates pass. `NarrativeCoordinator.commit()` is the only path that advances its versioned canonical narrative state. A coordinator-issued `StateUpdate` is required before the corresponding structured world version can advance. Prediction objects are explicitly marked hypothetical and have no commit API. Image observations are read-only.

## Implemented components

| Stage | Implementation | Current boundary |
|---|---|---|
| 4 — Strategic coordination | Typed events, character/world-facing state, proposals, revisions, disagreements, specialist continuity/intent/conflict/quality agents, bounded coordinator review, versioned commits and update IDs. Runner commits only accepted scenes and stores coordination data. | Local rule-based agents; not hosted specialist LLMs. The continuity and conflict agents implement bounded structural checks, not full semantic understanding. |
| 5 — World representation | `WorldState` accepts only a matching coordinator update, records entities, positions and object conditions, and exports USDA text. SQLite stores world and canonical state metadata per episode. | `pxr` is not installed, so USD SDK parsing and Omniverse visualization could not be verified; Omniverse remains optional and is accurately reported unavailable. |
| 6 — Visual observation | Validates PNG/JPEG frames, records a digest and dimensions, and accepts an injected detector contract to compare expected and observed entity state with confidence/evidence. Reports discrepancies without changing canonical state. | No semantic vision detector or local vision weights are configured. Without an injected detector, the result is `unavailable`; image semantics are not fabricated. |
| 7 — Future prediction | Bounded, action-conditioned, rule-based counterfactual branches from top VQC candidates. Branches carry risks, source version and `prediction` status. The predictor checks source immutability and cannot commit actions. | These are transparent heuristics, not calibrated or research-grade forecasts; plausibility scores are unavailable. |
| 8 — Temporal adaptation | Character profiles keep decisions, knowledge, event provenance and metrics. Measurable resolve, goal-consistency, trust, or emotional changes require an explicit effect supported by an event fact. A controlled paired-condition experiment runs the same policy over both event histories. | The prose renderer does not yet emit validated structured emotion/effect annotations, so unavailable measures remain unset; automatic profile updates are limited to accepted decisions and knowledge. Controlled experiments accept explicit, evidence-grounded event effects. |

The Streamlit research expander surfaces the coordinator version, committed updates, world version and USDA download, visual-observation availability, predictions, and adaptation evidence. Main manuscript exports remain reader-facing.

## Verification

Run from the repository root:

```powershell
$env:PYTHONPATH='.'
pytest -q -p no:cacheprovider tests
python -m compileall -q agents world_representation.py visual_observation.py future_prediction.py temporal_adaptation.py evaluation app.py ui tests
```

The Stage 4–8 regression tests exercise coordinator-only commits, duplicate rejection, world version checks and USDA serialization, prediction immutability, visual discrepancy detection and detector-unavailable reporting, evidence-grounded adaptation, controlled behavior changes, and SQLite system-state persistence.

## End-to-end model limitation

The configured runtime reports `models/OLMoE-1B-7B-0125-Instruct.Q4_K_M.gguf`. A six-scene fantasy acceptance run was persisted as episode 29 and exported to `acceptance_runs/generic_fantasy_manuscript.txt`. It accepted 1/6 scenes and correctly reported `partial`; five scenes failed length/truncation validation. This is not a complete-manuscript acceptance pass. `pxr` and a semantic vision detector are also absent, so those integrations are capability-gated rather than claimed as verified.
