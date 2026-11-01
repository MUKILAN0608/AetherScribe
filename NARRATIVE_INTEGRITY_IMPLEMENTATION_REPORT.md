# Narrative Integrity Implementation and Verification

## Changes completed in this pass

- A failed sequential scene now stops the episode. Dependent scenes are marked blocked and are not rendered from stale canonical state.
- Episode payloads track planned contracts, attempted and accepted scenes, failures, blocked indices, state versions, and completion conditions.
- Scene contracts are derived from the current story plan and coordinator state, then supplied as bounded rendering guidance. Contracts record dependencies, goals, known facts, objects, conflicts, expected consequences, and disclosure constraints.
- The runner builds each VQC candidate set from the current scene plan and cast. The measured posterior is conditioned on valid candidates before sampling; selected-candidate metadata, measured/conditioned probabilities, preconditions, risks, and effects are recorded. A caller-supplied precondition checker can reject candidates before the VQC sample.
- The coordinator serializes commits, rejects duplicate event IDs, checks the expected state version, and advances canonical facts only through accepted commits.
- Story configuration can seed canonical facts, locations, objects, open threads, active conflicts, plus character goals, beliefs, knowledge, relationships, condition, emotional state, and location. Accepted events can add knowledge only to explicitly named characters.
- Generated summaries are no longer copied into canonical facts or character knowledge. They are identified to the model as accepted narrative context, not verified world facts.
- The final-scene resolution gate runs before commit and provides a focused bounded revision instruction.
- Repetition checks compare normalized opening sentences. The runner now stores openings in the same normalized form the checker uses.
- Reader exports sort and deduplicate accepted scenes, exclude rejected drafts, and label partial manuscripts with missing and blocked scene numbers and unmet completion conditions.
- A reusable structural acceptance evaluator and live-provider CLI produce inspectable manuscript and JSON artifacts.
- Scene contracts now include a validated lifecycle with transition history; invalid transitions are rejected. Failed scenes include structured validation records with category, severity, evidence, expected/actual conditions, repair suggestion, and retry permission.
- Provider output ending with an explicit token-limit finish reason receives one bounded continuation request with the original contract and generated prefix. The continuation is joined with exact overlap removal and then goes through the ordinary full-scene validation gate.
- Local retries now prioritize the exact rejected opening, require a different cast member as the next opener, and increase sampling diversity within bounded limits; regression tests verify the directive reaches the compact provider prompt.
- Overlong retry drafts receive a bounded focused compression pass. The draft is kept only when it is complete and below the retry target; the normal quality, continuity, secret, coordinator, and ending checks still run afterward.
- Removed conflicting duplicate local context-size entries; managed llama.cpp now uses one configured 2048-token context matching the active endpoint and current GPU memory constraints. A separate 4096-token server could not allocate Vulkan KV cache while the app's model was active.
- Local llama.cpp startup now waits up to the configured startup timeout and preserves server output for diagnostics. Unavailable providers are reported with the configured model and endpoint.

## Verification

Baseline before this pass: `python -m pytest -q -p no:cacheprovider --basetemp C:\\Users\\Mukil\\AppData\\Local\\Temp\\quantum-rl-baseline` â€” **93 passed**.

After implementation: the final full-suite, compile, and mocked integration verification results are recorded below.

The original application's 1-of-8 screen did not include its run ID, full episode payload, or provider logs, so the exact historical scene failure cannot be recovered from the repository. The original traceback and subsequent error output do establish two independent failure paths: UI code indexed the final scene when `scenes` was empty, and configured generation runs failed from provider truncation or llama-server startup timeout. Current code guards empty scene lists, reports provider failures, and does not accept a rejected scene.

The latest live eight-scene run after sentence-boundary length repair attempted two scenes, accepted one, failed scene 2 after three retries, and blocked scenes 3–8. The failure was `internal_metadata_leak` and `scene_above_maximum_length`. The manuscript contains only the accepted first scene; the 8/8 target remains unmet. A separate 4096-context server could not allocate Vulkan KV cache while the app model was active, so `.env` now has one consistent 2048 setting. Artifacts:

- `acceptance_runs/live-thriller-8-scene.md`
- `acceptance_runs/live-thriller-8-scene-report.json`

The latest three-scene run accepted **3/3** with structural completion and ending gates passing. Inspection found vague, unresolved prose, so this is evidence of pipeline completion only, not literary or semantic correctness. Earlier runs also encountered truncated output and a llama-server startup timeout. Artifacts:

- `acceptance_runs/live-thriller-3-scene.md`
- `acceptance_runs/live-thriller-3-scene-report.json`

The eight-scene result demonstrates that pending scenes are not miscounted as failed and dependent scenes are blocked rather than rendered against stale state. Full eight-scene live completion has not been demonstrated.

Final commands and results after all code changes:

```text
python -m pytest -q -p no:cacheprovider --basetemp C:\Users\Mukil\AppData\Local\Temp\quantum-rl-final8: **126 passed**.
python -m compileall -q agents continuity evaluation experiments llm planning quantum_policy ui tests app.py test.py future_prediction.py temporal_adaptation.py visual_observation.py world_representation.py: **passed**.
python tests\verify_system.py: **passed** for all three mocked integration scenes.
```

Latest live eight-scene command:

```text
python experiments\run_acceptance.py --genre thriller --scenes 8 --premise "Mira investigates a silent archive where Voss has sealed the records and Nia knows how the old index was altered." --output-dir acceptance_runs
```

Latest result after complete-sentence trimming: **requested 8, attempted 2, accepted 1, failed 1, blocked 6, status partial**. The CLI returned exit code 1; structural checks passed, but completion did not.

## Remaining limits

The implementation has meaningful integrity gates, but this report does not claim that every requirement in `instructions.md` is solved:

- The runner exposes lifecycle transitions and rejects invalid transitions. Exact token overlap is removed during continuation, though semantic or paraphrased duplicate paragraphs are not detected reliably.
- Some planner, ending, provider, and coordinator errors still use separate issue formats rather than the shared structured quality-error schema.
- Candidate intents are generated from the scene plan, cast, and accepted predecessor. Built-in preconditions cover named actors and sequential dependencies; richer checks for arbitrary object access, location, injury, and knowledge need story-specific configuration or a domain precondition checker.
- Action realization is directed by prompts and encoded-label leakage is checked, but there is no general semantic proof that arbitrary selected actions occurred in the prose.
- Continuity checks are deterministic structural and lexical checks. They do not reliably infer arbitrary locations, object possession, relationship changes, beliefs, or knowledge from free prose. No semantic reviewer is currently authoritative for those facts.
- Scene-plan events and resolution conditions are generic guidance. The final ending check verifies structure and resolution signals, not whether every character arc or central question is substantively resolved.
- The live run does not establish reliable completion across genres, scene counts, or premises. The manuscript should be reviewed for literary quality and causal fidelity.

Retry prompt prioritization, per-retry sampling diversity, matched-prior opening diagnostics, focused opening repair, bounded continuation, overlength compression, exact duplicate-sentence cleanup, and complete-sentence trimming now have regression coverage. Live results remain unreliable with the configured local model; the latest attempt failed scene 2 on prompt leakage and length validation, and dependent scenes were blocked. A separate 4096-context process could not allocate Vulkan KV cache alongside the running app, so the supported local setting remains 2048. Quality gates remain enabled.


These limits are enforced as honest scope boundaries rather than hidden by inflating completion metrics.








