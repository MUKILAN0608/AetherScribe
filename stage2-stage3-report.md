# AetherScribe Stage 2 / Stage 3 Validation Report

## Executive Summary

Persistent story-scoped memory and advisory narrative planning are implemented
without changing VQC authority. The controlled benchmark replayed identical
locked action indices in memory-off and memory-on conditions. Stage 3 planning
was then enabled for a hosted Qwen3.5 episode and its validated advisory trace
was persisted with the scene.

## Repository Files Inspected

- `evaluation/episode_runner.py`
- `evaluation/database.py`
- `memory/`
- `llm/prompt_templates.py`
- `llm/story_renderer.py`
- `app.py`
- `architecture_utils.py`
- `tests/`

## Stage 2 Audit Findings

- SQLite memory records are story-scoped and status-filtered.
- Retrieval is deterministic, bounded, and reports diagnostics.
- Records retain category, confidence, origin, episode, scene, and status.
- Existing episode records are preserved through additive `story_id` migration.
- Memory retrieval is optional and failures continue without memory.
- Memory is passed as delimited advisory context, not system authority.

## Controlled Memory Benchmark

Command:

```powershell
python benchmarks\memory_benchmark.py
```

Configuration:

| Field | Value |
|---|---|
| Scenes | 3 |
| Seed | 17 |
| Memory budget | 800 |
| Provider | Deterministic fixture |
| Locked actions | `[6, 7, 13]` |

| Metric | Memory off | Memory on |
|---|---:|---:|
| Runtime | 6.327 ms | 93.346 ms |
| Retrieved per scene | `[0, 0, 0]` | `[0, 6, 11]` |
| Memory updates | 0 | 18 |
| Persisted records | 0 | 16 |
| Actions identical | Yes | Yes |
| Character names retained | Yes | Yes |

The memory-on overhead in this deterministic run was approximately 87.019 ms.
Token and provider-cost measurements are unavailable because this benchmark
intentionally uses a local deterministic fixture and makes no paid calls.

## Stage 3 Architecture and Implementation

Added:

- `planning/planner.py`
- `planning/config.py`
- `planning/__init__.py`
- `tests/test_narrative_planner.py`

Planner input includes:

- story and episode identifiers
- scene index
- current state
- validated story-scoped memory
- character knowledge constraints
- immutable locked action
- token budget
- reproducibility seed

Planner output includes:

- scene objective
- pacing
- tension
- relevant memory references
- knowledge constraints
- open threads
- continuity risks
- rendering guidance
- secret/reveal guidance
- confidence
- validation status
- provider/model/latency/fallback metadata

The planner is deterministic and local by default. It does not expose a
replacement action field and cannot mutate the locked action snapshot.

## Hosted Planner-Enabled Verification

Hosted episode:

- Episode ID: `11`
- Story ID: `planner-live-check`
- Narrative provider: OpenRouter
- Narrative model: `qwen/qwen3.5-35b-a3b`
- Narrative fallback: `false`
- Planner provider: local deterministic planner
- Planner status: `validated`
- Planner confidence: `0.75`
- Planner latency: `0.041 ms`
- Secret guidance: preserve secret information until character knowledge permits reveal

## Authority-Invariant Verification

- Replay benchmark actions were identical in both memory conditions.
- Planner receives an immutable `LockedActionSnapshot`.
- Planner output has no replacement-action field.
- Renderer receives the original locked action.
- Runner compares the locked action after rendering and raises on mutation.
- Jev remains an evaluator and does not replace the VQC action.

## Tests and Commands

```powershell
python -m pytest -q
```

Result:

```text
30 passed, 1 SQLAlchemy deprecation warning
```

```powershell
python -m compileall -q agents decision environment evaluation explainability llm memory planning quantum_policy ui app.py test.py experiments\run_single_episode.py benchmarks\memory_benchmark.py
```

Result: passed.

```powershell
python tests\verify_system.py
```

Result: passed.

## Known Limitations

- The benchmark uses deterministic fixtures and therefore cannot measure paid
  provider token usage or cost.
- Planner generation is local deterministic planning; hosted planner inference
  is not required.
- Secret handling is represented through validated character-knowledge
  constraints and renderer guidance; a future stage should add a dedicated
  structured secret ledger and prose-level reveal validator.
- One pre-existing SQLAlchemy deprecation warning remains.

## Reproduction

```powershell
python -m pytest -q
python benchmarks\memory_benchmark.py
python tests\verify_system.py
```

For hosted planner mode, set:

```env
NARRATIVE_MEMORY_ENABLED=true
NARRATIVE_PLANNER_ENABLED=true
NARRATIVE_LLM_PROVIDER=openrouter
```

## Recommended Next Stage

Add a structured secret ledger and deterministic reveal gate before rendering,
then compare human-rated continuity quality for memory-off, memory-on, and
planner-on conditions.
