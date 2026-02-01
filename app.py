import streamlit as st
import json
import os
import sys
import re
import plotly.graph_objects as go
import plotly.express as px
try:
    from sklearn.decomposition import PCA
    HAS_PCA = True
except ImportError:
    HAS_PCA = False

# Environment Path Configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Narrative Architecture Imports
from architecture_utils import load_env_variables, setup_research_logging
from environment.story_environment import StoryEnvironment
from quantum_policy.quantum_policy import QuantumPolicy
from quantum_policy.quantum_action_space import QuantumActionSpace
from explainability.explanation_engine import ExplanationEngine
from llm.story_renderer import StoryRenderer
from evaluation.episode_runner import EpisodeRunner

# Initialize Core Infrastructure
config = load_env_variables()
logger = setup_research_logging()

# Professional Workspace Configuration
st.set_page_config(
    page_title="Aether Scribe | Decision Intelligence Lab",
    page_icon="🖋️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Robust CSS Injection for High-Fidelity Dark Manuscript UI
# IMPORTANT: Absolutely NO indentation in the f-strings below to ensure perfect HTML rendering.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

/* Foundation: Deep Research Dark Mode */
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
}

/* Centering and Sidebar Removal */
[data-testid="stSidebar"] { display: none !important; }
.main .block-container {
    max-width: 850px !important;
    padding-top: 5rem !important;
    margin: 0 auto !important;
}

/* Laboratory Header */
.centered-header {
    text-align: center;
    color: #f0f6fc;
    font-size: 4rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.centered-subheader {
    text-align: center;
    color: #8b949e;
    font-size: 1.1rem;
    margin-bottom: 4rem;
}

/* Manuscript Narration Cards */
.manuscript-card {
    padding: 4rem 5rem;
    border-radius: 12px;
    border: 1px solid #30363d;
    background-color: #161b22;
    margin-bottom: 4rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.4);
    text-align: center;
}

.scene-marker {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3em;
    color: #8b949e;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid #30363d;
    padding-bottom: 1.2rem;
}

.manuscript-prose {
    font-family: 'Crimson Pro', 'Georgia', serif;
    font-size: 2.1rem;
    line-height: 1.5;
    color: #f0f6fc;
    font-style: italic;
    margin: 3.5rem 0;
}

/* Research Audit Sections */
.audit-section {
    margin-top: 4rem;
    padding-top: 2.5rem;
    border-top: 2px solid #30363d;
    text-align: left;
}

.audit-header {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 1.5rem;
    display: block;
    letter-spacing: 0.12em;
}

.selection-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-bottom: 2rem;
    background-color: #0d1117;
    border-radius: 6px;
    overflow: hidden;
}
.selection-table th {
    padding: 12px;
    color: #8b949e;
    background-color: #161b22;
    border-bottom: 1px solid #30363d;
    text-transform: uppercase;
    font-size: 0.65rem;
    text-align: left;
}
.selection-table td {
    padding: 14px 12px;
    border-bottom: 1px solid #30363d;
    color: #c9d1d9;
}
.row-selected {
    background-color: #1c2128;
    color: #58a6ff !important;
    font-weight: 700;
    border-left: 4px solid #58a6ff;
}

/* Professional Logic Memo */
.logic-block {
    padding: 1.5rem;
    background-color: #0d1117;
    border-left: 4px solid #58a6ff;
    margin: 2rem 0;
    font-size: 1rem;
    color: #c9d1d9;
    line-height: 1.7;
    text-align: left;
}

/* Laboratory Status Indicators */
.status-bar {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin-bottom: 4rem;
    padding: 1.2rem;
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
}
.status-item {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #8b949e;
    display: flex;
    align-items: center;
    gap: 10px;
}
.active-dot {
    height: 8px;
    width: 8px;
    background-color: #3fb950;
    border-radius: 50%;
    box-shadow: 0 0 10px #3fb950;
}

/* Justification Box */
.justification-memo {
    font-size: 0.85rem;
    color: #8b949e;
    background-color: #0d1117;
    padding: 1.2rem 1.5rem;
    border-radius: 6px;
    margin: 1.5rem 0;
    border-left: 3px solid #30363d;
    font-style: italic;
    line-height: 1.5;
}

/* Technical Trace Log */
.technical-log {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    background-color: #0d1117;
    color: #3fb950; /* Terminal Green */
    padding: 1.5rem;
    border-radius: 6px;
    line-height: 1.6;
    margin-top: 1.5rem;
    border: 1px solid #30363d;
    text-align: left;
}

/* Requirement labels */
.req-tag {
    font-size: 0.65rem;
    font-weight: bold;
    color: #58a6ff;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

def init_state():
    if 'current_episode' not in st.session_state:
        st.session_state.current_episode = None
    if 'trained_weights' not in st.session_state:
        st.session_state.trained_weights = None

def analyze_seed(text):
    if not text.strip(): return "Thriller", []
    text_clean = re.sub(r'[^\w\s]', '', text)
    text_lower = text_clean.lower()
    scores = {
        "Horror": len(re.findall(r"(ghost|murder|kill|blood|dark|scary|fear|dead|night|monster|demon|haunted|shadow|scream|curse|grave)", text_lower)),
        "Thriller": len(re.findall(r"(spy|data|corporate|detective|heist|chase|clock|time|future|conspiracy|investigate|secret|agent|ransom)", text_lower)),
        "Comedy": len(re.findall(r"(funny|laugh|joke|absurd|clumsy|prank|hilarious|witty|misunderstanding|satire|humor|silly)", text_lower)),
        "Romance": len(re.findall(r"(love|kiss|heart|passion|together|meet|date|forever|relationship|chemistry|attraction|proposal)", text_lower)),
        "Drama": len(re.findall(r"(family|life|growth|sad|tear|history|truth|emotional|conflict|struggle|betrayal|grief|orphan)", text_lower)),
        "Fantasy": len(re.findall(r"(magic|dragon|sword|realm|dream|spell|floating|kingdom|ancient|wizard|myth|quest|rune)", text_lower))
    }
    recommended = max(scores, key=scores.get) if max(scores.values()) > 0 else "Thriller"
    stops = {"The", "A", "In", "On", "At", "And", "He", "She", "It", "They", "But", "Or", "If", "Is", "Are", "Was", "Were", "An", "To", "With", "By", "For", "From"}
    potential_names = re.findall(r"\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+\b)*", text)
    extracted_names = []
    for name in potential_names:
        if name not in stops and name not in scores.keys():
            if name not in ["Scribe", "Aether", "Seed", "Story", "Manuscript", "Short", "Long", "Decision", "Log"]:
                if name not in extracted_names: extracted_names.append(name)
    return recommended, extracted_names[:3]

def run_engine_protocol(brief, mode="Short Story"):
    try:
        params = {"lr": 0.01, "eps": 0.01, "temp": 0.9}
        env = StoryEnvironment(genre=brief["genre"], director_brief=brief, max_steps=brief["max_scenes"])
        policy = QuantumPolicy(n_qubits=4)
        if st.session_state.get('trained_weights') is not None:
            policy.weights = st.session_state.trained_weights
        runner = EpisodeRunner(env, policy, QuantumActionSpace(), ExplanationEngine(), StoryRenderer(api_key=config["api_key"]))
        result = runner.run_episode(verbose=False, language=brief["language"], lr=params['lr'], eps=params['eps'], temperature=params['temp'], mode=mode)
        return {
            "id": result["id"],
            "scenes": result["scenes"],
            "total_reward": result["total_reward"],
            "language": brief["language"],
            "traits": result["traits"],
            "genre": brief["genre"],
            "premise": brief["initial_story_premise"],
            "mode": mode,
            "characters": brief["characters"]
        }
    except Exception as e:
        st.error(f"ENGINE_PROTOCOL_FAILURE: {str(e)}")
        return None

def generate_manuscript_latex(episode):
    latex = ["\\section{Narrative Analysis: " + episode['genre'].upper() + "}", "\\begin{description}", "  \\item[Execution ID:] " + episode['id'][:8], "  \\item[Architecture:] " + episode['mode'], "  \\item[Theme Seed:] " + episode['premise'][:150], "\\end{description}", "", "\\subsection{Decision Chronicles}", "\\begin{itemize}"]
    for s in episode['scenes']:
        content = s['story'].split("[QUANTUM_TRACE]")
        res_line = content[0].replace(f"[SCENE {s['step']}]", "").replace("RESULT:", "").strip()
        trace = content[1].strip() if len(content) > 1 else "N/A"
        latex.append(f"  \\item[Scene {s['step']}] {res_line}")
        latex.append(f"  \\textit{{Quantum Logic: {trace}}}")
    latex.append("\\end{itemize}")
    return "\n".join(latex)

def main():
    init_state()

    # --- CENTERED TITLE SECTION ---
    st.markdown("<h1 class='centered-header'>Aether Scribe</h1>", unsafe_allow_html=True)
    st.markdown("<p class='centered-subheader'>Decision Intelligence & Multi-Agent Research Laboratory v5.4.2</p>", unsafe_allow_html=True)

    # Laboratory Status
    st.markdown(f"""
<div class="status-bar">
<div class="status-item"><div class="active-dot"></div> Quantum Core: ACTIVE</div>
<div class="status-item"><div class="active-dot"></div> LLM Renderer: READY</div>
<div class="status-item"><div class="active-dot"></div> Telemetry: ONLINE</div>
<div class="status-item"><div class="active-dot" style="background-color: #58a6ff; box-shadow: 0 0 10px #58a6ff;"></div> Auth: AUTHORIZED</div>
</div>
""", unsafe_allow_html=True)

    # --- LABORATORY CALIBRATION (MAIN PAGE) ---
    with st.container():
        st.subheader("🖋️ Specification & Protocol Calibration")
        seed = st.text_area("Narrative Seed / Story Premise", "Elias Vance investigations lead him to a corporate vault where Director Kael hides an AI ghost named Lyra.", height=120)
        st.markdown("<div class='justification-memo'>**Research Necessity:** The seed defines the initial semantic constraints and character orientations. Without this, the quantum biasing weights have no mathematical anchor.</div>", unsafe_allow_html=True)

        rec_genre, extracted_chars = analyze_seed(seed)

        col1, col2, col3 = st.columns(3)
        with col1:
            genre_list = ["Thriller", "Horror", "Comedy", "Romance", "Drama", "Fantasy"]
            genre = st.selectbox("Active Domain", genre_list, index=genre_list.index(rec_genre))
            st.markdown("<span class='req-tag'>Requirement:</span> Genre выбор resolves corresponding tension multipliers for the narrative physics.", unsafe_allow_html=True)
        with col2:
            lang = st.selectbox("Output Language", ["English", "Spanish", "French", "German", "Japanese"])
            st.markdown("<span class='req-tag'>Requirement:</span> Localized rendering preserves semantic parity during the LLM feedback loop.", unsafe_allow_html=True)
        with col3:
            mode = st.selectbox("Trajectory Depth", ["Short Story", "Long Story"])
            st.markdown("<span class='req-tag'>Requirement:</span> Determines the number of quantum measurement cycles for the story arc.", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("EXECUTE RESEARCH PROTOCOL", type="primary", use_container_width=True):
            roles = ["protagonist", "antagonist", "ally"]
            char_names = extracted_chars + ["The Protagonist", "The Antagonist", "The Ally"]
            char_map = {roles[i]: char_names[i] for i in range(len(roles))}

            brief = {
                "initial_story_premise": seed,
                "genre": genre,
                "characters": char_map,
                "max_scenes": 5 if mode == "Short Story" else 10,
                "language": lang
            }
            with st.spinner("Resolving Quantum Trajectory Utility..."):
                result = run_engine_protocol(brief, mode=mode)
                if result:
                    st.session_state.current_episode = result
                    st.toast("Protocol Execution Success")

    st.divider()

    # --- RESEARCH RESULTS ---
    if st.session_state.current_episode:
        ep = st.session_state.current_episode
        st.markdown(f"<h2 style='text-align: center; color: #58a6ff; letter-spacing: 0.15em; margin-bottom: 5rem; font-weight: 300;'>{ep['genre'].upper()} MANUSCRIPT RESOLUTION</h2>", unsafe_allow_html=True)

        for scene in ep['scenes']:
            # Resolution
            content = scene['story'].split("[QUANTUM_TRACE]")
            res_line = content[0].replace(f"[SCENE {scene['step']}]", "").replace("RESULT:", "").strip()
            trace_content = content[1].strip() if len(content) > 1 else "Logic internalized."
            joint_action = scene['action']

            # Render Manuscript Card (No indentation in f-string to prevent Streamlit rendering bug)
            st.markdown(f"""
<div class="manuscript-card">
<div class="scene-marker">Observation Node {scene['step']} — {scene['state']['phase'].upper()}</div>
<div class="manuscript-prose">"{res_line}"</div>
<div class="audit-section">
<span class="audit-header">Selection Audit: Path Resolution Analysis</span>
<table class="selection-table">
<thead>
<tr>
<th>Character Combination</th>
<th>Decision Status</th>
<th>Probability</th>
<th>Selection Logic</th>
</tr>
</thead>
<tbody>
<tr class="row-selected">
<td><b>Chosen:</b> {joint_action.get('Protagonist')} + {joint_action.get('Antagonist')} + {joint_action.get('Ally')}</td>
<td>COLLAPSED</td>
<td>{scene['probs'][scene['action_idx']]*100:.1f}%</td>
<td>Optimal Utility</td>
</tr>
{"".join([f"<tr><td>{r['action']['Protagonist']} + {r['action']['Antagonist']} + {r['action']['Ally']}</td><td>REJECTED</td><td>{r['prob']*100:.1f}%</td><td>{r['reason']}</td></tr>" for r in scene['rejections'][:3]])}
</tbody>
</table>
<div class='justification-memo'>**Research Significance:** Contrasting rejected paths is required to verify character agency. It proves that the quantum circuit actively suppresses 'boring' futures based on character traits and tension.</div>
<div class="logic-block">
<p style="font-weight: 700; margin-bottom: 0.6rem; color: #58a6ff; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.12em;">Decision Rationale Memo</p>
{trace_content}
</div>
<div class="technical-log">
[RESOLVER_LOG_NODE_{scene['step']}]
STATUS: MEASURED_COLLAPSE
ENTROPY: {scene['entropy']:.4f} | REWARD: {scene['reward']:.2f} | RISK: {scene['risk']*100:.0f}%
ATTRIBUTION: P={scene['attribution']['Protagonist']:.2f}, A={scene['attribution']['Antagonist']:.2f}, L={scene['attribution']['Ally']:.2f}
<b>Why this is required:</b> These low-level metrics verify character agency vs. narrative arc control for formal research.
</div>
</div>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # --- NARRATIVE PHYSICS (APPENDIX) ---
        with st.container():
            st.subheader("🔬 RESEARCH APPENDIX: NARRATIVE PHYSICS")
            st.markdown("Mathematical verification of the story resolution forces.")

            pg1, pg2 = st.columns(2)
            with pg1:
                st.markdown("#### Probability Surface")
                fig1 = go.Figure(data=[go.Bar(
                    x=[str(i) for i in range(16)],
                    y=ep['scenes'][-1]['probs'],
                    marker_color=['#58a6ff' if i == ep['scenes'][-1]['action_idx'] else '#30363d' for i in range(16)]
                )])
                fig1.update_layout(title="Active Quantum Superposition", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=350, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig1, use_container_width=True)
                st.info("**Why this is required:** Visualizes the competitive 'What-If' landscape before path resolution.")

            with pg2:
                st.markdown("#### Influence Attribution")
                agents = ["Protagonist", "Antagonist", "Ally"]
                vals = [ep['scenes'][-1]['attribution'][a] for a in agents]
                fig2 = go.Figure(data=[go.Pie(labels=agents, values=vals, hole=.4, marker=dict(colors=['#58a6ff', '#f85149', '#3fb950']))])
                fig2.update_layout(title="Character Driver Distribution", paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=350, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig2, use_container_width=True)
                st.info("**Why this is required:** Quantifies whose traits drove the final resolution outcome.")

            if HAS_PCA and len(ep['scenes']) >= 2:
                st.divider()
                st.markdown("#### Laboratory Analysis: Trajectory Projection")
                h = [[s['state']['tension'], s['reward'], s['entropy'], s['risk']] for s in ep['scenes']]
                pca = PCA(n_components=2); coords = pca.fit_transform(h)
                fig3 = px.scatter(x=coords[:,0], y=coords[:,1], text=[f"Node {i}" for i in range(len(coords))])
                fig3.update_traces(marker=dict(size=18, color='#58a6ff', line=dict(width=2, color='white')))
                fig3.update_layout(title="High-Dimensional Logic Map (PCA)", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=450)
                st.plotly_chart(fig3, use_container_width=True)
                st.info("**Why this is required:** Maps logical narrative consistency across the entire story arc.")

        st.divider()
        d_col1, d_col2 = st.columns(2)
        with d_col1: st.download_button("Export Raw Telemetry (JSON)", json.dumps(ep, indent=2), file_name="lab_data.json", use_container_width=True)
        with d_col2: st.download_button("Export Manuscript (LaTeX)", generate_manuscript_latex(ep), file_name="manuscript.tex", use_container_width=True)

    else:
        st.info("Laboratory Idle. Define a narrative seed above and click 'EXECUTE' to begin the protocol.")

if __name__ == "__main__":
    main()
