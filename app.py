"""
Narrative trajectory application.
"""
import os
import sys
import uuid

import numpy as np
import streamlit as st

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from architecture_utils import load_env_variables, setup_research_logging
from environment.story_environment import StoryEnvironment
from evaluation.episode_runner import EpisodeRunner
from evaluation.database import AetherDatabase
from decision.providers import DecisionRouter
from explainability.explanation_engine import ExplanationEngine
from llm.story_renderer import StoryRenderer
from quantum_policy.quantum_action_space import QuantumActionSpace
from quantum_policy.quantum_policy import QuantumPolicy
from ui.research_views import (
    render_dashboard_tab,
    render_kpi_strip,
    render_memory_panel,
    render_secret_panel,
    render_story_tab,
)
from ui.seed_analysis import analyze_seed
from ui.styles import RESEARCH_CSS
from planning import NarrativePlanner

config = load_env_variables()
setup_research_logging()

st.set_page_config(
    page_title="Narrative Studio",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(RESEARCH_CSS, unsafe_allow_html=True)

DEFAULT_LR = 0.01
DEFAULT_EPS = 0.01
DEFAULT_TEMP = 0.88



def init_state():
    for key, val in {
        "current_episode": None,
        "trained_weights": None,
        "running": False,
        "char_protagonist": "",
        "char_antagonist": "",
        "char_ally": "",
        "story_id": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val


def run_trajectory(brief, progress_bar=None, status=None):
    env = StoryEnvironment(
        genre=brief["genre"], director_brief=brief, max_steps=brief["scenes"]
    )
    policy = QuantumPolicy(n_qubits=4)
    if st.session_state.trained_weights is not None:
        policy.weights = np.array(st.session_state.trained_weights)

    database = AetherDatabase()
    renderer = StoryRenderer(api_key=config.get("api_key"))
    rstatus = renderer.status()
    if rstatus["state"] == "local_model":
        st.sidebar.success(f"Local model active: {rstatus['model']}")
    elif rstatus["state"] == "template_fallback":
        st.sidebar.warning("Template fallback active: generated output is test-only and cannot complete a manuscript.")
    else:
        st.sidebar.error(
            f"Narrative provider error: {rstatus.get('provider_error') or 'model unavailable'}"
        )
    st.sidebar.caption(f"Decision provider: {'Jev' if config['jev_enabled'] else 'local rules'}")
    if rstatus.get("endpoint"):
        st.sidebar.caption(f"Endpoint: {rstatus['endpoint']}")
    st.sidebar.caption(f"Model: {rstatus['model']} · Latency: {rstatus['last_latency_ms'] or '—'} ms")
    st.sidebar.caption(
        f"Tokens: {rstatus['input_tokens'] or '—'} in / {rstatus['output_tokens'] or '—'} out · "
        f"Cost: ${rstatus['estimated_cost_usd'] or 0:.6f}"
    )

    runner = EpisodeRunner(
        env, policy, QuantumActionSpace(), ExplanationEngine(), renderer,
        decision_provider=DecisionRouter(),
        memory_repository=database.memory if config["memory_enabled"] else None,
        memory_enabled=config["memory_enabled"],
        memory_limit=config["memory_retrieval_limit"],
        memory_budget=config["memory_context_budget"],
        story_id=brief.get("story_id"),
        planner=NarrativePlanner(enabled=config["planner_enabled"]),
        planner_enabled=config["planner_enabled"],
        secret_repository=database.secrets if config["secret_ledger_enabled"] else None,
        secret_enabled=(
            config["secret_ledger_enabled"]
        ),
        reveal_gate_enabled=config["reveal_gate_enabled"],
        scene_history_repository=database.scene_history,
        repetition_retry_limit=int(os.getenv("NARRATIVE_MAX_RETRIES", "3")),
    )

    total = brief["scenes"]

    def on_step(_entry, current, _total):
        if progress_bar is not None:
            progress_bar.progress(
                current / total,
                text=f"Scene {current} of {total}",
            )
        if status is not None:
            status.caption(f"Writing scene {current} of {total}…")

    result = runner.run_episode(
        verbose=False,
        language=brief["language"],
        lr=DEFAULT_LR,
        eps=DEFAULT_EPS,
        temperature=DEFAULT_TEMP,
        mode="Long Story",
        render_depth="fast",
        on_step=on_step,
    )
    episode_id = database.save_episode(
        brief["genre"], brief["initial_story_premise"], result["total_reward"], result["scenes"],
        story_id=result["story_id"],
    )
    st.sidebar.caption(
        f"Memory: {'enabled' if config['memory_enabled'] else 'disabled'} · "
        f"selected {sum(s.get('memory_retrieval', {}).get('selected', 0) for s in result['scenes'])} · "
        f"updated {sum(s.get('memory_updates', 0) for s in result['scenes'])} · "
        f"episode {episode_id}"
    )
    memory_snapshot = database.get_memory_context(
        result["story_id"],
        budget=config["memory_context_budget"],
        limit=config["memory_retrieval_limit"],
    )
    st.sidebar.caption(
        f"Last generation: {renderer.last_latency_ms if renderer.last_latency_ms is not None else '—'} ms · "
        f"provider: {renderer.provider} · fallback: {'yes' if renderer.last_used_fallback else 'no'}"
    )

    if result["scenes"]:
        st.session_state.trained_weights = result["scenes"][-1]["weights"]

    return {
        "id": result["id"],
        "scenes": result["scenes"],
        "total_reward": result["total_reward"],
        "genre": brief["genre"],
        "premise": brief["initial_story_premise"],
        "characters": brief["characters"],
        "language": brief["language"],
        "scenes_count": brief["scenes"],
        "story_id": result["story_id"],
        "episode_id": episode_id,
        "requested_scene_count": result["requested_scene_count"],
        "planned_scene_count": result["planned_scene_count"],
        "story_bible": result["story_bible"],
        "story_plan": result["story_plan"],
        "accepted_scene_count": result["accepted_scene_count"],
        "failed_scene_count": result["failed_scene_count"],
        "story_complete": result["story_complete"],
        "status": result["status"],
        "scene_failures": result["scene_failures"],
        "ending_validation": result["ending_validation"],
        "memory": [
            {
                "id": record.id,
                "category": record.category,
                "content": record.content,
                "confidence": record.confidence,
                "origin": record.origin,
                "scene_index": record.scene_index,
                "status": record.status.value,
            }
            for record in memory_snapshot.records
        ],
    }


def _sync_names_from_seed(seed: str):
    """Refresh sidebar character fields when premise changes."""
    _, _, role_map = analyze_seed(seed, api_key=config.get("api_key"))
    if role_map.get("protagonist"):
        st.session_state.char_protagonist = role_map["protagonist"]
    if role_map.get("antagonist"):
        st.session_state.char_antagonist = role_map["antagonist"]
    if role_map.get("ally"):
        st.session_state.char_ally = role_map["ally"]


def render_sidebar():
    st.sidebar.markdown("## Configuration")

    seed = st.sidebar.text_area(
        "Premise",
        value="",
        placeholder="Describe the central character, situation, and choice or conflict.",
        height=120,
        help="Describe any story premise. Character names and supporting characters are optional.",
    )

    rec_genre, _, role_map = analyze_seed(seed, api_key=config.get("api_key"))

    if st.sidebar.button("Detect names from premise", use_container_width=True):
        _sync_names_from_seed(seed)
        st.sidebar.success("Character names updated.")

    # Auto-fill on first load if empty
    if not st.session_state.char_protagonist and role_map:
        _sync_names_from_seed(seed)

    st.sidebar.markdown("**Characters**")

    n1 = role_map.get("protagonist") or "Character 1"
    n2 = role_map.get("antagonist") or "Character 2"
    n3 = role_map.get("ally") or "Character 3"

    protagonist = st.sidebar.text_input(
        n1,
        value=st.session_state.char_protagonist or role_map.get("protagonist", ""),
        key="input_protagonist",
    )
    antagonist = st.sidebar.text_input(
        n2,
        value=st.session_state.char_antagonist or role_map.get("antagonist", ""),
        key="input_antagonist",
    )
    ally = st.sidebar.text_input(
        n3,
        value=st.session_state.char_ally or role_map.get("ally", ""),
        key="input_ally",
    )

    st.session_state.char_protagonist = protagonist.strip()
    st.session_state.char_antagonist = antagonist.strip()
    st.session_state.char_ally = ally.strip()

    genre_options = ["Thriller", "Mystery", "Fantasy", "Science fiction", "Romance", "Horror", "Adventure", "Drama", "Comedy", "Historical fiction"]
    genre = st.sidebar.selectbox("Genre", genre_options, index=genre_options.index(rec_genre) if rec_genre in genre_options else 0)
    language = st.sidebar.selectbox(
        "Language",
        ["English", "Spanish", "French", "German", "Japanese"],
    )
    scenes = st.sidebar.slider("Number of scenes", min_value=6, max_value=12, value=8)
    tone = st.sidebar.text_input("Tone (optional)", placeholder="e.g. intimate, wry, tense")
    writing_style = st.sidebar.text_input("Writing style (optional)", placeholder="e.g. spare, lyrical, fast-paced")
    scene_word_target = st.sidebar.slider("Approximate words per scene", 60, 120, 80, step=10)
    pov_preference = st.sidebar.selectbox("Point of view", ["Protagonist", "Third person limited", "First person"])
    ending_preference = st.sidebar.selectbox("Ending preference", ["Earned resolution", "Hopeful", "Bittersweet", "Open-ended", "Tragic"])
    world_rules = st.sidebar.text_area("World rules or constraints (optional)", height=70)
    continue_story = st.sidebar.checkbox(
        "Continue story memory",
        value=False,
        help="Reuse memory only for an intentional continuation of the current story.",
    )

    st.sidebar.markdown("---")
    run = st.sidebar.button("Generate", type="primary", use_container_width=True)

    return {
        "seed": seed.strip(),
        "genre": genre,
        "language": language,
        "scenes": scenes,
        "tone": tone.strip(),
        "writing_style": writing_style.strip(),
        "scene_word_target": scene_word_target,
        "pov_preference": pov_preference,
        "ending_preference": ending_preference,
        "world_rules": world_rules.strip(),
        "characters": {
            "protagonist": protagonist.strip(),
            "antagonist": antagonist.strip(),
            "ally": ally.strip(),
        },
        "run": run,
        "continue_story": continue_story,
    }


def main():
    init_state()
    cfg = render_sidebar()

    if cfg["run"] and not cfg["seed"]:
        st.warning("Enter a premise before generating.")
        return

    if cfg["run"] and not st.session_state.running:
        brief = {
            "initial_story_premise": cfg["seed"],
            "genre": cfg["genre"],
            "characters": cfg["characters"],
            "scenes": cfg["scenes"],
            "language": cfg["language"],
            "tone": cfg["tone"],
            "writing_style": cfg["writing_style"],
            "scene_word_target": cfg["scene_word_target"],
            "pov_preference": cfg["pov_preference"],
            "ending_preference": cfg["ending_preference"],
            "world_rules": cfg["world_rules"],
            "story_id": (
                st.session_state.story_id
                if cfg["continue_story"] and st.session_state.story_id
                else f"story-{uuid.uuid4()}"
            ),
        }

        st.session_state.running = True
        progress = st.progress(0, text="Starting…")
        status = st.empty()
        try:
            ep = run_trajectory(brief, progress_bar=progress, status=status)
            if ep:
                st.session_state.current_episode = ep
                st.session_state.story_id = ep["story_id"]
                if ep.get("story_complete"):
                    st.success("Complete.")
                else:
                    st.warning(
                        f"Incomplete: {ep.get('accepted_scene_count', len(ep.get('scenes', [])))} "
                        f"of {ep.get('requested_scene_count', brief['scenes'])} scenes passed validation."
                    )
        except Exception as exc:
            st.error("Generation failed. Check the local provider status and try again.")
            with st.expander("Details"):
                st.code(str(exc))
        finally:
            progress.empty()
            status.empty()
            st.session_state.running = False

    ep = st.session_state.current_episode

    if not ep:
        st.markdown(
            """
<div class="idle-panel">
  <p style="font-size:1.05rem;color:#c9d1d9;margin-bottom:0.5rem;">
    No story yet
  </p>
  <p style="font-size:0.88rem;">
    Enter a premise and character names in the sidebar, then select Generate.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    requested = ep.get("requested_scene_count", ep.get("scenes_count", len(ep["scenes"])))
    accepted = ep.get("accepted_scene_count", len(ep["scenes"]))
    failed = len(ep.get("scene_failures", []))
    st.caption(
        f"{ep['genre']} · requested {requested} · accepted {accepted} · failed {failed} · "
        f"status {ep.get('status', 'unknown')} · {ep['language']}"
    )

    tab_story, tab_dash = st.tabs(["Manuscript", "Dashboard"])

    with tab_story:
        render_kpi_strip(ep)
        render_story_tab(ep)

    with tab_dash:
        render_dashboard_tab(ep)
        render_memory_panel(ep)
        render_secret_panel(ep)

    if st.button("New story"):
        st.session_state.current_episode = None
        st.rerun()


if __name__ == "__main__":
    main()
