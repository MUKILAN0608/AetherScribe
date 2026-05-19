"""
Narrative trajectory application.
"""
import os
import sys

import numpy as np
import streamlit as st

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from architecture_utils import load_env_variables, setup_research_logging
from environment.story_environment import StoryEnvironment
from evaluation.episode_runner import EpisodeRunner
from explainability.explanation_engine import ExplanationEngine
from llm.story_renderer import StoryRenderer
from quantum_policy.quantum_action_space import QuantumActionSpace
from quantum_policy.quantum_policy import QuantumPolicy
from ui.research_views import render_dashboard_tab, render_kpi_strip, render_story_tab
from ui.seed_analysis import analyze_seed
from ui.styles import RESEARCH_CSS

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

    renderer = StoryRenderer(api_key=config.get("api_key"))
    rstatus = renderer.status()
    if not rstatus["has_api_key"]:
        st.sidebar.error("GOOGLE_API_KEY missing from .env")
    elif rstatus["provider"] != "gemini":
        st.sidebar.warning("Narrative engine not connected — check GOOGLE_API_KEY in .env")
    else:
        st.sidebar.success("Narrative engine connected")

    runner = EpisodeRunner(
        env, policy, QuantumActionSpace(), ExplanationEngine(), renderer
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
        render_depth="full",
        on_step=on_step,
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
        value=(
            "Elias Vance investigates a concealed corporate vault while Director Kael "
            "obstructs the inquiry; Lyra supplies intelligence from the periphery."
        ),
        height=120,
        help="Describe the story. Include character names (e.g. Elias Vance, Director Kael).",
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

    genre = st.sidebar.selectbox(
        "Genre",
        ["Thriller", "Horror", "Comedy", "Romance", "Drama", "Fantasy"],
        index=["Thriller", "Horror", "Comedy", "Romance", "Drama", "Fantasy"].index(rec_genre),
    )
    language = st.sidebar.selectbox(
        "Language",
        ["English", "Spanish", "French", "German", "Japanese"],
    )
    scenes = st.sidebar.slider("Number of scenes", min_value=6, max_value=12, value=8)

    st.sidebar.markdown("---")
    run = st.sidebar.button("Generate", type="primary", use_container_width=True)

    return {
        "seed": seed.strip(),
        "genre": genre,
        "language": language,
        "scenes": scenes,
        "characters": {
            "protagonist": protagonist.strip() or n1,
            "antagonist": antagonist.strip() or n2,
            "ally": ally.strip() or n3,
        },
        "run": run,
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
        }

        st.session_state.running = True
        progress = st.progress(0, text="Starting…")
        status = st.empty()
        try:
            ep = run_trajectory(brief, progress_bar=progress, status=status)
            if ep:
                st.session_state.current_episode = ep
                st.success("Complete.")
        except Exception as exc:
            st.error("Generation failed. Check your API key and try again.")
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

    st.caption(f"{ep['genre']} · {len(ep['scenes'])} scenes · {ep['language']}")

    tab_story, tab_dash = st.tabs(["Manuscript", "Dashboard"])

    with tab_story:
        render_kpi_strip(ep)
        render_story_tab(ep)

    with tab_dash:
        render_dashboard_tab(ep)

    if st.button("New story"):
        st.session_state.current_episode = None
        st.rerun()


if __name__ == "__main__":
    main()
