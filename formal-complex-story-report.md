# AetherScribe Formal Complex Story Run Report

**Run date:** 2026-10-31  
**Story ID:** `formal-complex-example`  
**Episode ID:** `10`  
**Result:** PASS

## Input

**Premise:**  
In 2149, archivist Mira discovers that the Helix Observatory can erase one
historical catastrophe, but each altered memory removes a living person from
the present. Director Voss wants to rewrite the event to consolidate power,
while Nia, a defector carrying the original star-map, knows the cost and hides
the final coordinate.

| Field | Value |
|---|---|
| Genre | Thriller |
| Language | English |
| Protagonist | Mira |
| Antagonist | Voss |
| Ally | Nia |
| Scenes | 2 |
| Random seed | 23 |
| Memory | Enabled |
| Memory budget | 800 tokens |
| Narrative provider | OpenRouter |
| Narrative model | `qwen/qwen3.5-35b-a3b` |
| Jev model | `meta-llama/llama-3.1-8b-instruct` |
| Fallback used | No |

## Timing Measurements

| Component | Measurement |
|---|---:|
| Database/provider setup | 76.4 ms |
| VQC scene 1 | 4.1 ms |
| VQC scene 2 | 3.6 ms |
| Jev scene 1 | 2,310.3 ms |
| Jev scene 2 | 4,519.0 ms |
| Memory retrieval scene 1 | 10.0 ms |
| Memory retrieval scene 2 | 2.6 ms |
| Memory persistence total | 135.9 ms |
| Narrative generation scene 1 | 3,091.5 ms |
| Narrative generation scene 2 | 3,820.4 ms |
| Episode execution | 13,899.3 ms |
| Episode SQLite persistence | 14.3 ms |
| Total wall-clock runtime | 13,990.9 ms |

## Scene Audit

| Scene | Locked action | Memory selected | Coherence | Provider |
|---:|---|---:|---:|---|
| 1 | Mira: Final Stand; Voss: Unleash Full Might; Nia: Last Support | 0 | 0.90 | OpenRouter Qwen3.5 |
| 2 | Mira: Final Stand; Voss: Unleash Full Might; Nia: Last Support | 6 | 0.90 | OpenRouter Qwen3.5 |

## Generated Output

### Scene 1

Mira stood before the Helix console where the amber light of the erased
catastrophe flickered like a dying heartbeat against the sterile air. She felt
the weight of the final stand pressing against her ribs as the system demanded
a living price for every second of rewritten history. Director Voss stepped
from the shadows, his silhouette sharp against the data streams, and unleashed
his full might by overriding the safety protocols with a command that cracked
the air. The floor vibrated as the observatory's defenses collapsed, not to
protect the timeline, but to secure Voss's grip on the future he intended to
sculpt. Nia, trembling with the memory of the star-map hidden in her blood,
offered the last support needed to keep the coordinates from dissolving into
the void. Her fingers hovered over the physical lock, a desperate anchor
against the tide of erasure that Voss now commanded with terrifying ease.

**Quantum trace:** Mira, Voss, and Nia collided at the narrative event horizon
where the decision to erase the past demanded the immediate sacrifice of the
present.

### Scene 2

Mira planted her boots against the slick floor of the Helix chamber, refusing
to yield an inch as Voss's armored form surged forward with crushing,
calculated violence. The Director's hand cracked against her ribs, a sharp,
wet sound that silenced the hum of the collapsing timeline and forced air from
her lungs in a ragged gasp. While Mira struggled to maintain her grip on the
console's edge, Nia's voice cut through the din, whispering the final
coordinate that anchored their reality before the void could swallow them
whole. The star-map flickered violently in Nia's trembling hand, casting jagged
shadows across her face as she fought to keep the erasure at bay with a
strength born of desperate loyalty. Voss roared, unleashing the full might of
his mechanical augmentations to crush the archivist beneath a wall of steel,
yet he faltered when the coordinate locked into place. Mira gritted her teeth
against the pain, channeling every ounce of will into the interface while Nia
held the line, preventing the present from unraveling into a history that
never was.

**Quantum trace:** Voss's unleashed force met Mira's unyielding stance, while
Nia's whispered anchor stabilized the fracturing timeline.

## Persistence and Safety Checks

- Episode saved to SQLite as episode `10`.
- Memory remained scoped to `formal-complex-example`.
- Seven active deduplicated memory records were available after the run.
- Scene 2 retrieved six records from the same story.
- No provider fallback occurred.
- The locked quantum action remained unchanged across both scenes.
- Jev evaluated the action but did not replace it.

## Execution Command

The run used the existing Python pipeline with hosted narrative mode enabled,
memory enabled, seed `23`, and a two-scene complex premise. Credentials were
read from the local environment and were not printed or stored in this report.
