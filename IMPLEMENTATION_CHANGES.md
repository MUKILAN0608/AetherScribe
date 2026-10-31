# AetherScribe Generation Repair

## Implemented

- Local provider selection now probes and reuses an existing OpenAI-compatible Qwen endpoint, discovers the model ID from `/v1/models`, parses usage and finish metadata, and exposes provider failures instead of silently reporting a template as a real model.
- Local Qwen is selected in `.env` and `.env.example`. The UI distinguishes the local model, template test output, and provider error.
- Scene prompts use a compact local-model path carrying the premise, configured characters, latest accepted context, scene event, evidence, consequence, and translated action cues.
- The planner creates a premise-aware corporate-vault investigation arc with scene-specific objectives, opposition, ally contribution, evidence, consequences, and next-step effects.
- The quality gate rejects repeated sentences, prompt fragments, known generic fallbacks, missing planned evidence, invalid lengths, truncated output, invented city/year changes, premise restatement, and repeated scenes.
- Retry instructions are targeted and do not pass raw validation issue codes to the model. Rejected scenes do not update accepted memory/history or count toward completion.
- Reader prose and decision traces are stored separately. Manuscript exports contain reader-facing prose only.
- Completion status requires the requested number of accepted scenes and a passing ending check. UI metrics report requested, planned, accepted, failed, and status values. Unmeasured coherence is shown as unavailable.
- The UI hides decision analysis unless explicitly enabled.

## Verification

- `python -m pytest -q --basetemp .pytest-tmp`: 67 passed.
- `python -m py_compile app.py evaluation/episode_runner.py llm/providers.py llm/story_renderer.py llm/prompt_templates.py continuity/quality.py continuity/repetition.py planning/planner.py tests/verify_system.py`: passed.
- `python test.py`: passed in deterministic smoke mode.
- `python tests/verify_system.py`: passed for all three integration fixture scenes.
- Local endpoint verification: Qwen model `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` was reached at `http://127.0.0.1:8080/v1`; metadata reported the real local provider, tokens, latency, and no fallback.
- Streamlit returned HTTP 200 at `http://localhost:8503`.

## Live 11-Scene Result

A real 11-scene run was attempted with local Qwen, no fallback, planning, memory, evidence validation, and SQLite persistence. The final attempt accepted and persisted 2 of 11 scenes, then stopped as `partial` because the model did not include required access-history evidence within the configured retry limit. The stored partial result is `logs/live-qwen-final13.db`; persisted manuscript story fields do not contain quantum trace markup.

The 0.5B model therefore does not yet satisfy the full 11-scene acceptance condition. The system correctly reports partial output instead of marking it complete. A more capable local model or longer, more reliable inference configuration is still needed for a fully accepted 11-scene manuscript.

## Run

With the local Qwen server available:

```powershell
$env:NARRATIVE_LLM_PROVIDER="local"
$env:LLM_PROVIDER="local"
streamlit run app.py
```

Run tests:

```powershell
$env:NARRATIVE_LLM_PROVIDER="deterministic"
$env:LLM_PROVIDER="deterministic"
python -m pytest -q --basetemp .pytest-tmp
```
