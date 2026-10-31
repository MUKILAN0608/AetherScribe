# Runtime Validation Report

**Target:** `C:\Users\Mukil\quantum_rl`

## Summary

| Step | Status | Exit code | Evidence |
|---|---|---:|---|
| Regression tests | PASS | 0 | 25 pytest tests passed |
| Compilation | PASS | 0 | All application modules compiled |
| Quantum integration | PASS | 0 | `python tests\verify_system.py` passed |
| Hosted provider | PASS | 0 | Jev and Qwen3.5 OpenRouter paths verified |
| Stage 2 memory tests | PASS | 0 | 7 focused memory tests passed |
| Memory-enabled episode | PASS | 0 | Two scenes, story-scoped retrieval, SQLite persistence |
| Streamlit startup | PASS | 0 | `/_stcore/health` returned HTTP 200 on port 8502 |

## Stage 2 Evidence

- Story ID: `stage2-example-story`
- Persisted episode ID: `9`
- Scenes: `2`
- Active memory records: `11`
- Retrieval on unrelated story: `0` records
- Memory updates attempted: `12`
- Same-story retrieval selected records on scene 2.
- Every memory record contains category, confidence, origin, status, and scene provenance.
- Memory-off integration remains compatible with the existing mock renderer.
- The renderer receives the locked action through an immutable mapping and the
  runner verifies the action is unchanged after rendering.

## Measured hosted baseline

- Jev: `meta-llama/llama-3.1-8b-instruct`
- Narrative: `qwen/qwen3.5-35b-a3b`
- Reasoning effort: `none`
- Prior measured three-scene run: 4,481 input tokens, 1,072 output tokens,
  estimated cost `$0.002006`.

## Commands

```powershell
python -m pytest -q
python -m compileall -q agents decision environment evaluation explainability llm memory quantum_policy ui app.py test.py experiments\run_single_episode.py
python tests\verify_system.py
streamlit run app.py --server.headless true --server.port 8502
Invoke-WebRequest http://127.0.0.1:8502/_stcore/health
```

## Known limitations

- Candidate extraction is deliberately deterministic and authoritative-data-only;
  it does not infer facts from prose.
- Memory retrieval is lexical/deterministic; semantic embeddings are disabled.
- A live hosted provider is not required by the default test suite.
- SQLite migration uses an idempotent additive `story_id` column and preserves
  existing episode rows.

## Overall

**PASS** — Persistent story-scoped memory is implemented, bounded, inspectable,
safe-failure compatible, and integrated without changing quantum authority.
