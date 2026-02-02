import streamlit as st
import json
import os
import sys
import re
import numpy as np
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
    st.markdown("<p class='centered-subheader'>Simple & Smart Storytelling Lab</p>", unsafe_allow_html=True)

    # Laboratory Status
    st.markdown(f"""
<div class="status-bar">
<div class="status-item"><div class="active-dot"></div> Story Brain: READY</div>
<div class="status-item"><div class="active-dot"></div> Writer: READY</div>
<div class="status-item"><div class="active-dot"></div> Analysis: ONLINE</div>
</div>
""", unsafe_allow_html=True)

    # --- LABORATORY CALIBRATION (MAIN PAGE) ---
    with st.container():
        st.subheader("🖋️ Set Up Your Story")
        seed = st.text_area("Story Idea (Premise)", "Elias Vance is looking for a secret corporate vault while Director Kael tries to stop him using an AI ghost named Lyra.", height=120)
        st.markdown("<div class='justification-memo'>Give a starting point for your story and name your characters.</div>", unsafe_allow_html=True)

        rec_genre, extracted_chars = analyze_seed(seed)

        col1, col2, col3 = st.columns(3)
        with col1:
            genre_list = ["Thriller", "Horror", "Comedy", "Romance", "Drama", "Fantasy"]
            genre = st.selectbox("Story Type", genre_list, index=genre_list.index(rec_genre))
        with col2:
            lang = st.selectbox("Language", ["English", "Spanish", "French", "German", "Japanese"])
        with col3:
            mode = st.selectbox("Story Length", ["Short Story", "Long Story"])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("WRITE MY STORY", type="primary", use_container_width=True):
            roles = ["protagonist", "antagonist", "ally"]
            # Fill in names from seed, or use defaults
            provided_names = extracted_chars
            while len(provided_names) < 3:
                provided_names.append(f"The {roles[len(provided_names)].capitalize()}")

            char_map = {roles[i]: provided_names[i] for i in range(len(roles))}

            brief = {
                "initial_story_premise": seed,
                "genre": genre,
                "characters": char_map,
                "max_scenes": 5 if mode == "Short Story" else 10,
                "language": lang
            }
            with st.spinner("Building your story..."):
                result = run_engine_protocol(brief, mode=mode)
                if result:
                    st.session_state.current_episode = result
                    st.toast("Story Created Successfully!")

    st.divider()

    # --- RESEARCH RESULTS ---
    if st.session_state.current_episode:
        ep = st.session_state.current_episode
        st.markdown(f"<h2 style='text-align: center; color: #58a6ff; letter-spacing: 0.15em; margin-bottom: 5rem; font-weight: 300;'>YOUR {ep['genre'].upper()} STORY</h2>", unsafe_allow_html=True)

        for scene in ep['scenes']:
            # Resolution
            content = scene['story'].split("[QUANTUM_TRACE]")
            res_line = content[0].replace(f"[SCENE {scene['step']}]", "").replace("RESULT:", "").strip()
            trace_content = content[1].strip() if len(content) > 1 else "Story logic applied."
            joint_action = scene['action']

            # Build Rejection Rows for Research Audit
            rejection_rows = ""
            for i, r in enumerate(scene['rejections'][:3]):
                # Map roles in actions to names
                formatted_actions = ", ".join([f"{name_map.get(c, c)}: {a.replace('_', ' ')}" for c, a in r['action'].items()])
                # Use deep rejection if available
                deep_reason = scene.get('deep_rejections', [])[i] if i < len(scene.get('deep_rejections', [])) else r['reason']
                rejection_rows += f"<tr><td>{formatted_actions}</td><td>{r['prob']*100:.1f}%</td><td>{deep_reason}</td></tr>"

            # Render Manuscript Card
            st.markdown(f"""
<div class="manuscript-card">
<div class="scene-marker">Scene {scene['step'] + 1}</div>
<div class="manuscript-prose">"{res_line}"</div>

<div class="status-bar" style="margin-bottom: 2rem; background-color: #0d1117; padding: 0.8rem; border-radius: 6px;">
<div class="status-item" style="font-size: 0.6rem;"><div class="active-dot" style="background-color: #58a6ff;"></div> Tension: {scene['state']['tension']:.2f}</div>
<div class="status-item" style="font-size: 0.6rem;"><div class="active-dot" style="background-color: #3fb950;"></div> Reward: {scene['reward']:.2f}</div>
<div class="status-item" style="font-size: 0.6rem;"><div class="active-dot" style="background-color: #d29922;"></div> Entropy: {scene['entropy']:.2f}</div>
<div class="status-item" style="font-size: 0.6rem;"><div class="active-dot" style="background-color: #f85149;"></div> Risk: {scene['risk']:.2f}</div>
</div>

<div class="audit-section">
<span class="audit-header">Why this happened</span>
<div class="logic-block">
{scene['rationale']}
</div>

<span class="audit-header">Research Audit: Alternative Path Suppression</span>
<table class="selection-table">
<thead>
<tr>
<th>Alternative Action</th>
<th>Probability</th>
<th>Rejection Reason</th>
</tr>
</thead>
<tbody>
{rejection_rows}
</tbody>
</table>

<div class="technical-log">
<b>Scene Insights:</b><br>
Tension: {scene['state']['tension']*100:.0f}% | Interest Score: {scene['reward']:.2f}<br>
Main Influencer: {name_map.get(max(scene['attribution'], key=scene['attribution'].get))} <br>
Logic Note: {trace_content}
</div>
</div>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # --- NARRATIVE PHYSICS (APPENDIX) ---
        with st.container():
            st.subheader("📊 Research Dashboard: Multi-Agent Narrative Telemetry")
            st.markdown("12-Point Analysis Grid for Decision Intelligence Audit.")

            # Create Name Map for Chart Labels
            name_map = {role.capitalize(): name for role, name in ep['characters'].items()}

            # 4x3 Grid for Graphs
            # Row 1
            r1_c1, r1_c2, r1_c3 = st.columns(3)
            with r1_c1:
                st.markdown("#### Story Tension")
                tension_vals = [s['state']['tension'] for s in ep['scenes']]
                steps = [f"S{s['step']+1}" for s in ep['scenes']]
                fig_tension = go.Figure(data=go.Scatter(x=steps, y=tension_vals, mode='lines+markers', line=dict(color='#58a6ff', width=3)))
                fig_tension.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=250, margin=dict(l=0,r=0,t=30,b=0), yaxis_range=[0,1])
                st.plotly_chart(fig_tension, use_container_width=True)

            with r1_c2:
                st.markdown("#### Final Influence")
                agents_roles = ["Protagonist", "Antagonist", "Ally"]
                agent_names = [name_map.get(a, a) for a in agents_roles]
                vals = [ep['scenes'][-1]['attribution'][a] for a in agents_roles]
                fig2 = go.Figure(data=[go.Pie(labels=agent_names, values=vals, hole=.4, marker=dict(colors=['#58a6ff', '#f85149', '#3fb950']))])
                fig2.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=250, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig2, use_container_width=True)

            with r1_c3:
                st.markdown("#### Decision Confidence")
                gaps = []
                for s in ep['scenes']:
                    sorted_probs = sorted(s['probs'], reverse=True)
                    gaps.append(sorted_probs[0] - sorted_probs[1])
                fig_gap = go.Figure(data=go.Scatter(x=steps, y=gaps, fill='tozeroy', line=dict(color='#bc8cff')))
                fig_gap.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=250, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_gap, use_container_width=True)

            # Row 2
            r2_c1, r2_c2, r2_c3 = st.columns(3)
            with r2_c1:
                st.markdown("#### Action Entropy")
                entropy_vals = [s['entropy'] for s in ep['scenes']]
                fig_entropy = go.Figure(data=go.Scatter(x=steps, y=entropy_vals, mode='lines+markers', line=dict(color='#d29922', width=3)))
                fig_entropy.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=250, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_entropy, use_container_width=True)

            with r2_c2:
                st.markdown("#### Story Quality")
                reward_vals = [s['reward'] for s in ep['scenes']]
                fig_reward = go.Figure(data=go.Bar(x=steps, y=reward_vals, marker_color='#3fb950'))
                fig_reward.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=250, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_reward, use_container_width=True)

            with r2_c3:
                st.markdown("#### Sensitivity Analysis")
                features = ["Aggro", "Strat", "Sneak", "Loyal"]
                sensitivity_vals = ep['scenes'][-1]['sensitivity']
                fig_sens = go.Figure(data=[go.Bar(x=features, y=sensitivity_vals, marker_color='#ff7b72')])
                fig_sens.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=250, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_sens, use_container_width=True)

            # Row 3
            r3_c1, r3_c2, r3_c3 = st.columns(3)
            with r3_c1:
                st.markdown("#### Rel Dynamics")
                trust_vals = [s['state']['relationships']['protagonist_ally']['trust'] for s in ep['scenes']]
                enmity_vals = [s['state']['relationships']['protagonist_antagonist']['enmity'] for s in ep['scenes']]
                p_name = name_map.get("Protagonist", "Hero")
                a_name = name_map.get("Antagonist", "Villain")
                l_name = name_map.get("Ally", "Friend")
                fig_rel = go.Figure()
                fig_rel.add_trace(go.Scatter(x=steps, y=trust_vals, name=f"{p_name}-{l_name}", line=dict(color='#3fb950', width=2)))
                fig_rel.add_trace(go.Scatter(x=steps, y=enmity_vals, name=f"{p_name}-{a_name}", line=dict(color='#f85149', width=2)))
                fig_rel.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=250, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_rel, use_container_width=True)

            with r3_c2:
                st.markdown("#### Control Heatmap")
                attr_data = []
                for s in ep['scenes']:
                    attr_data.append([s['attribution'][c] for c in agents_roles])
                fig_heat = px.imshow(
                    np.array(attr_data).T,
                    x=steps,
                    y=agent_names,
                    color_continuous_scale="Viridis"
                )
                fig_heat.update_layout(coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=250, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_heat, use_container_width=True)

            with r3_c3:
                st.markdown("#### Prob Surface")
                fig_probs = go.Figure(data=[go.Bar(
                    x=[str(i) for i in range(len(ep['scenes'][-1]['probs']))],
                    y=ep['scenes'][-1]['probs'],
                    marker_color=['#58a6ff' if i == ep['scenes'][-1]['action_idx'] else '#30363d' for i in range(len(ep['scenes'][-1]['probs']))]
                )])
                fig_probs.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=250, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_probs, use_container_width=True)

            # Row 4 (Wider visual row)
            r4_c1, r4_c2, r4_c3 = st.columns(3)
            with r4_c1:
                st.markdown("#### Narrative Shape")
                if HAS_PCA and len(ep['scenes']) >= 2:
                    h = [[s['state']['tension'], s['reward'], s['entropy'], s['risk']] for s in ep['scenes']]
                    pca = PCA(n_components=2); coords = pca.fit_transform(h)
                    fig3 = px.scatter(x=coords[:,0], y=coords[:,1], text=steps)
                    fig3.update_traces(marker=dict(size=12, color='#58a6ff'), textposition='top center')
                    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=250, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Insufficient data for PCA.")

            with r4_c2:
                st.markdown("#### Branching Audit")
                sources, targets, values, labels = [], [], [], []
                p_name = name_map.get("Protagonist", "Hero")
                a_name = name_map.get("Antagonist", "Villain")
                for i, scene in enumerate(ep['scenes']):
                    labels.append(f"S{i+1}")
                    curr = len(labels) - 1

                    labels.append(f"{p_name[:5]}...")
                    chosen = len(labels) - 1
                    sources.append(curr); targets.append(chosen); values.append(scene['probs'][scene['action_idx']])

                    if scene['rejections']:
                        labels.append(f"{a_name[:5]}?")
                        rej = len(labels) - 1
                        sources.append(curr); targets.append(rej); values.append(scene['rejections'][0]['prob'])

                fig_sankey = go.Figure(data=[go.Sankey(
                    node = dict(pad = 10, thickness = 10, label = labels, color = "#58a6ff"),
                    link = dict(source = sources, target = targets, value = values, color = "rgba(88, 166, 255, 0.2)")
                )])
                fig_sankey.update_layout(font_size=8, height=250, paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_sankey, use_container_width=True)

            with r4_c3:
                st.markdown("#### Tension vs Utility")
                fig_traj = px.scatter(x=tension_vals, y=reward_vals, trendline="ols")
                fig_traj.update_traces(marker=dict(size=10, color='#58a6ff'))
                fig_traj.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=250, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_traj, use_container_width=True)

        st.divider()
        d_col1, d_col2 = st.columns(2)
        with d_col1: st.download_button("Save Story Data (JSON)", json.dumps(ep, indent=2), file_name="story_data.json", use_container_width=True)
        with d_col2: st.download_button("Download Story (LaTeX)", generate_manuscript_latex(ep), file_name="story.tex", use_container_width=True)

    else:
        st.info("Laboratory Idle. Define a narrative seed above and click 'EXECUTE' to begin the protocol.")

if __name__ == "__main__":
    main()
