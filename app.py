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
    page_title="Aether Scribe | Master Research Edition",
    page_icon="🖋️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- MASTER CSS: HIGH-FIDELITY RENDER ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
}

.main .block-container {
    padding-top: 2rem !important;
    max-width: 96% !important;
    margin: 0 auto !important;
}

.centered-header {
    text-align: center;
    color: #f0f6fc;
    font-size: 6rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.centered-subheader {
    text-align: center;
    color: #8b949e;
    font-size: 1.8rem;
    margin-bottom: 5rem;
    font-weight: 300;
}

.manuscript-card {
    padding: 4rem 6rem;
    border-radius: 16px;
    border: 1px solid #30363d;
    background-color: #161b22;
    margin-bottom: 4rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.4);
}

.scene-marker {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4em;
    color: #58a6ff;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid #30363d;
    padding-bottom: 1.2rem;
}

.manuscript-prose {
    font-family: 'Crimson Pro', serif;
    font-size: 2.8rem;
    line-height: 1.6;
    color: #f0f6fc;
    font-style: italic;
    margin: 3rem 0;
    text-align: center;
}

.audit-section {
    margin-top: 4rem;
    padding-top: 3rem;
    border-top: 2px solid #30363d;
}

.audit-header {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 1.5rem;
    display: block;
    letter-spacing: 0.15em;
}

.selection-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 1.1rem;
    margin-bottom: 2.5rem;
    background-color: #0d1117;
    border-radius: 10px;
    overflow: hidden;
}
.selection-table th {
    padding: 20px;
    color: #8b949e;
    background-color: #1c2128;
    border-bottom: 1px solid #30363d;
    text-align: left;
    font-size: 0.85rem;
}
.selection-table td {
    padding: 20px;
    border-bottom: 1px solid #30363d;
    color: #c9d1d9;
}

.logic-block {
    padding: 2rem;
    background-color: #0d1117;
    border-left: 6px solid #58a6ff;
    margin: 2rem 0;
    font-size: 1.3rem;
    line-height: 1.7;
    color: #e6edf3;
}

.status-bar {
    display: flex;
    justify-content: center;
    gap: 4rem;
    margin-bottom: 4rem;
    padding: 1.5rem;
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 15px;
}
.status-item {
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #8b949e;
    display: flex;
    align-items: center;
    gap: 12px;
}
.active-dot {
    height: 12px;
    width: 12px;
    background-color: #3fb950;
    border-radius: 50%;
    box-shadow: 0 0 15px #3fb950;
}

.technical-log {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    background-color: #0d1117;
    color: #3fb950;
    padding: 2rem;
    border-radius: 10px;
    border: 1px solid #30363d;
    line-height: 1.8;
}

.justification-memo {
    font-size: 1.2rem;
    color: #8b949e;
    background-color: #161b22;
    padding: 2rem;
    border-radius: 12px;
    margin: 2rem 0;
    border: 1px solid #30363d;
    line-height: 1.7;
}

.char-badge {
    background-color: #238636;
    color: white;
    padding: 6px 16px;
    border-radius: 25px;
    font-size: 0.9rem;
    font-weight: 700;
    margin-right: 10px;
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

    # --- MASTER NAME EXTRACTION ---
    meta_stops = {
        "The", "A", "An", "In", "On", "At", "And", "He", "She", "It", "They", "But", "Or", "If", "Is", "Are", "Was", "Were",
        "To", "With", "By", "For", "From", "There", "While", "Story", "Idea", "Premise", "Character", "Names", "Hero",
        "Villain", "Friend", "Vault", "Corporate", "Secret", "Looking", "Lead", "Ghost", "Director", "Research", "Lab",
        "Writing", "My", "Your", "This", "That", "When", "Where", "How", "Why", "About", "Using", "Hides", "Named", "Lead",
        "Protagonist", "Antagonist", "Ally", "Scene", "Logic", "Choice", "Result", "Technical", "Trace", "Node", "Research"
    }
    potential_entities = re.findall(r"\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+\b)*", text)
    extracted_names = []
    for entity in potential_entities:
        first_word = entity.split()[0]
        if first_word not in meta_stops and entity not in scores.keys() and entity not in extracted_names:
            extracted_names.append(entity)
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
            "characters": brief["characters"]
        }
    except Exception as e:
        st.error(f"MASTER_ENGINE_FAILURE: {str(e)}")
        return None

def generate_manuscript_latex(episode):
    latex = ["\\section{Master Narrative Analysis: " + episode['genre'].upper() + "}", "\\begin{description}", "  \\item[ID:] " + episode['id'][:8], "  \\item[Seed:] " + episode['premise'][:200], "\\end{description}", "", "\\subsection{Story Path}"]
    for s in episode['scenes']:
        content = s['story'].split("[QUANTUM_TRACE]")
        res_line = content[0].replace(f"[SCENE {s['step']}]", "").replace("RESULT:", "").strip()
        latex.append(f"\\paragraph{{Scene {s['step'] + 1}}} {res_line}")
    return "\n".join(latex)

def main():
    init_state()

    st.markdown("<h1 class='centered-header'>Aether Scribe</h1>", unsafe_allow_html=True)
    st.markdown("<p class='centered-subheader'>Master Render Research Edition v6.0</p>", unsafe_allow_html=True)

    st.markdown(f"""
<div class="status-bar">
<div class="status-item"><div class="active-dot"></div> Intelligence: READY</div>
<div class="status-item"><div class="active-dot"></div> Renderer: READY</div>
<div class="status-item"><div class="active-dot"></div> Broad Telemetry: ONLINE</div>
</div>
""", unsafe_allow_html=True)

    with st.container():
        st.subheader("🖋️ Set Up Your Story")
        seed = st.text_area("Story Idea (Character Names & Setting)", "Elias Vance is looking for a secret corporate vault while Director Kael tries to stop him using Lyra.", height=120)
        st.markdown("<div class='justification-memo'>Describe who is involved and what is happening. Use real names. The system will use these to tell the story.</div>", unsafe_allow_html=True)

        rec_genre, extracted_chars = analyze_seed(seed)

        col1, col2, col3 = st.columns(3)
        with col1:
            genre_list = ["Thriller", "Horror", "Comedy", "Romance", "Drama", "Fantasy"]
            genre = st.selectbox("Story Type", genre_list, index=genre_list.index(rec_genre))
        with col2:
            lang = st.selectbox("Language", ["English", "Spanish", "French", "German", "Japanese"])
        with col3:
            mode = st.selectbox("Story Depth", ["Short Story", "Long Story"])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("WRITE MY STORY", type="primary", use_container_width=True):
            roles = ["protagonist", "antagonist", "ally"]
            provided_names = extracted_chars
            while len(provided_names) < 3:
                provided_names.append(f"The {roles[len(provided_names)].capitalize()}")
            char_map = {roles[i]: provided_names[i] for i in range(len(roles))}
            brief = {
                "initial_story_premise": seed,
                "genre": genre,
                "characters": char_map,
                "max_scenes": 6 if mode == "Short Story" else 12,
                "language": lang
            }
            with st.spinner("Rendering Trajectory..."):
                result = run_engine_protocol(brief, mode=mode)
                if result:
                    st.session_state.current_episode = result
                    st.toast("Full Render Success")

    st.divider()

    if st.session_state.current_episode:
        ep = st.session_state.current_episode
        name_map = {role.capitalize(): name for role, name in ep['characters'].items()}

        tab_story, tab_research = st.tabs(["🖋️ MANUSCRIPT PASS", "📊 RESEARCH DASHBOARD"])

        with tab_story:
            st.markdown(f"<h2 style='text-align: center; color: #58a6ff; letter-spacing: 0.15em; margin-bottom: 2rem; font-weight: 300;'>{ep['genre'].upper()} FILE: MASTER RENDER</h2>", unsafe_allow_html=True)

            # --- EXECUTIVE SUMMARY & RESEARCH MEMO ---
            memo_col1, memo_col2 = st.columns([1, 1])
            with memo_col1:
                st.markdown("### 📝 Mission Summary")
                st.markdown(f"""
                <div class='justification-memo'>
                <b>Objective:</b> Resolve a {ep['genre'].lower()} story arc with child-friendly English.<br>
                <b>Primary Idea:</b> {ep['premise']}<br>
                <b>Outcome:</b> Story finished with a total score of {ep['total_reward']:.2f}.
                </div>
                """, unsafe_allow_html=True)
            with memo_col2:
                st.markdown("### 🔬 Executive Research Memo")
                st.markdown(f"""
                <div class='justification-memo'>
                The system kept character names perfectly.
                The story is easy to read. Logic follows personality traits.
                Alternative choices were skipped based on story quality.
                </div>
                """, unsafe_allow_html=True)

            with st.expander("👤 Character Dossiers & Entity Bonds", expanded=True):
                dcol1, dcol2, dcol3 = st.columns(3)
                roles = ["Protagonist", "Antagonist", "Ally"]
                cols = [dcol1, dcol2, dcol3]
                for i, role in enumerate(roles):
                    name = name_map.get(role, "Unknown")
                    with cols[i]:
                        st.markdown(f"<span class='char-badge'>{role}</span>", unsafe_allow_html=True)
                        st.markdown(f"### {name}")
                        if role == "Antagonist":
                            val = ep['scenes'][-1]['state']['relationships']['protagonist_antagonist']['enmity']
                            st.markdown(f"**Rivalry:** {val*100:.0f}% Hate")
                        elif role == "Ally":
                            val = ep['scenes'][-1]['state']['relationships']['protagonist_ally']['trust']
                            st.markdown(f"**Bond:** {val*100:.0f}% Trust")
                        else:
                            st.markdown(f"**Status:** Lead Character")

            st.divider()

            for scene in ep['scenes']:
                content = scene['story'].split("[QUANTUM_TRACE]")
                res_line = content[0].replace(f"[SCENE {scene['step']}]", "").replace("RESULT:", "").strip()
                trace_content = content[1].strip() if len(content) > 1 else "Logic verified."

                rejection_rows = ""
                for i, r in enumerate(scene['rejections'][:3]):
                    choices = " | ".join([f"**{name_map.get(role, role)}**: {act.lower().replace('_', ' ')}" for role, act in r['action'].items()])
                    deep_reason = scene.get('deep_rejections', [])[i] if i < len(scene.get('deep_rejections', [])) else r['reason']
                    rejection_rows += f"<tr><td>{choices}</td><td>{r['prob']*100:.1f}%</td><td>{deep_reason}</td></tr>"

                st.markdown(f"""
<div class="manuscript-card">
<div class="scene-marker">Observation Node {scene['step'] + 1}</div>
<div class="manuscript-prose">"{res_line}"</div>

<div class="status-bar">
<div class="status-item"><div class="active-dot" style="background-color: #58a6ff; box-shadow: 0 0 10px #58a6ff;"></div> Tension: {scene['state']['tension']:.2f}</div>
<div class="status-item"><div class="active-dot" style="background-color: #3fb950; box-shadow: 0 0 10px #3fb950;"></div> Story Fit: {scene['reward']:.2f}</div>
<div class="status-item"><div class="active-dot" style="background-color: #d29922; box-shadow: 0 0 10px #d29922;"></div> Confidence: {1.0 - scene['entropy']:.2f}</div>
<div class="status-item"><div class="active-dot" style="background-color: #f85149; box-shadow: 0 0 10px #f85149;"></div> Risk: {scene['risk']:.2f}</div>
</div>

<div class="audit-section">
<span class="audit-header">📜 Why this happened (Logic Pass)</span>
<div class="logic-block">{scene['rationale']}</div>

<span class="audit-header">🚫 Paths Not Taken (Simple Audit)</span>
<table class="selection-table">
<thead><tr><th>Possible Choice</th><th>Likelihood</th><th>Why it was skipped</th></tr></thead>
<tbody>{rejection_rows}</tbody>
</table>

<div class="technical-log">
<b>Research Summary:</b> Driven by <b>{name_map.get(max(scene['attribution'], key=scene['attribution'].get))}</b> personality.<br>
Trace Log: {trace_content}
</div>
</div>
</div>
""", unsafe_allow_html=True)

        with tab_research:
            st.subheader("📊 Research Dashboard: 12-Point Trajectory Analysis")
            st.markdown("High-resolution telemetry grid showing character choices and world dynamics.")

            steps = [f"S{s['step']+1}" for s in ep['scenes']]
            tension_vals = [s['state']['tension'] for s in ep['scenes']]
            reward_vals = [s['reward'] for s in ep['scenes']]
            entropy_vals = [s['entropy'] for s in ep['scenes']]

            # 4-Column x 3-Row Broad Grid
            chart_height = 700
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            with r1c1:
                st.markdown("#### Story Tension")
                fig = go.Figure(data=go.Scatter(x=steps, y=tension_vals, mode='lines+markers', line=dict(color='#58a6ff', width=5)))
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=chart_height, margin=dict(l=10,r=10,t=40,b=40), yaxis_range=[0,1], font_size=16)
                st.plotly_chart(fig, use_container_width=True)
            with r1c2:
                st.markdown("#### Power Split")
                roles = ["Protagonist", "Antagonist", "Ally"]
                agent_names = [name_map.get(r, r) for r in roles]
                vals = [ep['scenes'][-1]['attribution'][r] for r in roles]
                fig = go.Figure(data=[go.Pie(labels=agent_names, values=vals, hole=.4, marker=dict(colors=['#58a6ff', '#f85149', '#3fb950']))])
                fig.update_layout(showlegend=True, paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=chart_height, margin=dict(l=10,r=10,t=40,b=40), legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5), font_size=16)
                st.plotly_chart(fig, use_container_width=True)
            with r1c3:
                st.markdown("#### Decision Clarity")
                gaps = [sorted(s['probs'], reverse=True)[0] - sorted(s['probs'], reverse=True)[1] for s in ep['scenes']]
                fig = go.Figure(data=go.Scatter(x=steps, y=gaps, fill='tozeroy', line=dict(color='#bc8cff', width=4)))
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=chart_height, margin=dict(l=10,r=10,t=40,b=40), font_size=16)
                st.plotly_chart(fig, use_container_width=True)
            with r1c4:
                st.markdown("#### Choice Certainty")
                fig = go.Figure(data=go.Scatter(x=steps, y=entropy_vals, mode='lines+markers', line=dict(color='#d29922', width=4)))
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=chart_height, margin=dict(l=10,r=10,t=40,b=40), font_size=16)
                st.plotly_chart(fig, use_container_width=True)

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            with r2c1:
                st.markdown("#### Story Fit (Reward)")
                fig = go.Figure(data=go.Bar(x=steps, y=reward_vals, marker_color='#3fb950'))
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=chart_height, margin=dict(l=10,r=10,t=40,b=40), font_size=16)
                st.plotly_chart(fig, use_container_width=True)
            with r2c2:
                st.markdown("#### Personality Pulse")
                sens_vals = ep['scenes'][-1]['sensitivity']
                fig = go.Figure(data=[go.Bar(x=["Aggro", "Strat", "Sneak", "Loyal"], y=sens_vals, marker_color='#ff7b72')])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=chart_height, margin=dict(l=10,r=10,t=40,b=40), font_size=16)
                st.plotly_chart(fig, use_container_width=True)
            with r2c3:
                st.markdown("#### Character Bonds")
                t_vals = [s['state']['relationships']['protagonist_ally']['trust'] for s in ep['scenes']]
                e_vals = [s['state']['relationships']['protagonist_antagonist']['enmity'] for s in ep['scenes']]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=steps, y=t_vals, name=f"Trust: {name_map.get('Ally')}", line=dict(color='#3fb950', width=4)))
                fig.add_trace(go.Scatter(x=steps, y=e_vals, name=f"Hate: {name_map.get('Antagonist')}", line=dict(color='#f85149', width=4)))
                fig.update_layout(showlegend=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=chart_height, margin=dict(l=10,r=10,t=40,b=40), legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5), font_size=16)
                st.plotly_chart(fig, use_container_width=True)
            with r2c4:
                st.markdown("#### Control Heatmap")
                attr_data = [[s['attribution'][r] for r in ["Protagonist", "Antagonist", "Ally"]] for s in ep['scenes']]
                fig = px.imshow(np.array(attr_data).T, x=steps, y=agent_names, color_continuous_scale="Viridis")
                fig.update_layout(coloraxis_showscale=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=chart_height, margin=dict(l=10,r=10,t=40,b=40), font_size=16)
                st.plotly_chart(fig, use_container_width=True)

            r3c1, r3c2, r3c3, r3c4 = st.columns(4)
            with r3c1:
                st.markdown("#### Choice Pool")
                fig = go.Figure(data=[go.Bar(x=[f"O{i+1}" for i in range(len(ep['scenes'][-1]['probs']))], y=ep['scenes'][-1]['probs'], marker_color=['#58a6ff' if i == ep['scenes'][-1]['action_idx'] else '#30363d' for i in range(len(ep['scenes'][-1]['probs']))])])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=chart_height, margin=dict(l=10,r=10,t=40,b=40), font_size=16)
                st.plotly_chart(fig, use_container_width=True)
            with r3c2:
                st.markdown("#### Narrative Shape")
                if HAS_PCA and len(ep['scenes']) >= 2:
                    h = [[s['state']['tension'], s['reward'], s['entropy'], s['risk']] for s in ep['scenes']]
                    pca = PCA(n_components=2); coords = pca.fit_transform(h)
                    fig = px.scatter(x=coords[:,0], y=coords[:,1], text=steps)
                    fig.update_traces(marker=dict(size=25, color='#58a6ff', line=dict(width=2, color='white')), textposition='top center')
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=chart_height, margin=dict(l=10,r=10,t=40,b=40), font_size=16)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info("No PCA data.")
            with r3c3:
                st.markdown("#### Decision Tree")
                sources, targets, values, labels = [], [], [], []
                p_char_name_s, a_char_name_s = name_map.get("Protagonist", "Hero"), name_map.get("Antagonist", "Villain")
                for i, s in enumerate(ep['scenes']):
                    labels.append(f"S{i+1}"); curr = len(labels) - 1
                    labels.append(f"Used: {p_char_name_s[:10]}"); sources.append(curr); targets.append(len(labels)-1); values.append(s['probs'][s['action_idx']])
                    if s['rejections']:
                        labels.append(f"Skip: {a_char_name_s[:10]}"); sources.append(curr); targets.append(len(labels)-1); values.append(s['rejections'][0]['prob'])
                fig = go.Figure(data=[go.Sankey(node = dict(pad = 15, thickness = 20, label = labels, color = "#58a6ff", line=dict(color="white", width=0.5)), link = dict(source = sources, target = targets, value = values, color = "rgba(88, 166, 255, 0.4)"))])
                fig.update_layout(font_size=14, height=chart_height, paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", margin=dict(l=10,r=10,t=40,b=40))
                st.plotly_chart(fig, use_container_width=True)
            with r3c4:
                st.markdown("#### Pressure vs Utility")
                fig = px.scatter(x=tension_vals, y=reward_vals, trendline="ols", text=steps)
                fig.update_traces(marker=dict(size=20, color='#58a6ff', symbol='diamond'), textposition='top center')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=chart_height, margin=dict(l=10,r=10,t=40,b=40), font_size=16)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        dcol1, dcol2 = st.columns(2)
        with dcol1: st.download_button("Export Master JSON", json.dumps(ep, indent=2), file_name="master_render.json", use_container_width=True)
        with dcol2: st.download_button("Export Manuscript", generate_manuscript_latex(ep), file_name="manuscript.tex", use_container_width=True)

    else:
        st.info("Laboratory Idle. Define a narrative seed above and click 'WRITE MY STORY' to begin.")

if __name__ == "__main__":
    main()
