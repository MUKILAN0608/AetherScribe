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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Robust CSS Injection for High-Fidelity Dark Manuscript UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

/* Foundation: Deep Research Dark Mode */
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
}

.main .block-container {
    padding-top: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 98% !important;
    margin: 0 auto !important;
}

/* Laboratory Header */
.centered-header {
    text-align: center;
    color: #f0f6fc;
    font-size: 5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.centered-subheader {
    text-align: center;
    color: #8b949e;
    font-size: 1.5rem;
    margin-bottom: 4rem;
}

/* Manuscript Narration Cards */
.manuscript-card {
    padding: 3rem 4rem;
    border-radius: 12px;
    border: 1px solid #30363d;
    background-color: #161b22;
    margin-bottom: 3rem;
    box-shadow: 0 15px 40px rgba(0,0,0,0.3);
}

.scene-marker {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3em;
    color: #8b949e;
    margin-bottom: 2rem;
    border-bottom: 1px solid #30363d;
    padding-bottom: 1rem;
}

.manuscript-prose {
    font-family: 'Crimson Pro', serif;
    font-size: 2rem;
    line-height: 1.5;
    color: #f0f6fc;
    font-style: italic;
    margin: 2rem 0;
    text-align: center;
}

.audit-section {
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 2px solid #30363d;
}

.audit-header {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 1.2rem;
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
}
.selection-table th {
    padding: 10px;
    color: #8b949e;
    background-color: #161b22;
    border-bottom: 1px solid #30363d;
    text-align: left;
    font-size: 0.7rem;
}
.selection-table td {
    padding: 12px 10px;
    border-bottom: 1px solid #30363d;
    color: #c9d1d9;
}

.logic-block {
    padding: 1.2rem;
    background-color: #0d1117;
    border-left: 4px solid #58a6ff;
    margin: 1.5rem 0;
    font-size: 0.95rem;
}

.status-bar {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin-bottom: 3rem;
    padding: 1rem;
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
}
.status-item {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    color: #8b949e;
    display: flex;
    align-items: center;
    gap: 8px;
}
.active-dot {
    height: 8px;
    width: 8px;
    background-color: #3fb950;
    border-radius: 50%;
    box-shadow: 0 0 10px #3fb950;
}

.technical-log {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    background-color: #0d1117;
    color: #3fb950;
    padding: 1.2rem;
    border-radius: 6px;
    border: 1px solid #30363d;
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

    # --- ROBUST NAME EXTRACTION ---
    meta_stops = {
        "The", "A", "An", "In", "On", "At", "And", "He", "She", "It", "They", "But", "Or", "If", "Is", "Are", "Was", "Were",
        "To", "With", "By", "For", "From", "There", "While", "Story", "Idea", "Premise", "Character", "Names", "Hero",
        "Villain", "Friend", "Vault", "Corporate", "Secret", "Looking", "Lead", "Ghost", "Director", "Research", "Lab",
        "Writing", "My", "Your", "This", "That", "When", "Where", "How", "Why", "About", "Using", "Hides", "Named", "Lead",
        "Protagonist", "Antagonist", "Ally", "Scene", "Logic", "Choice", "Result", "Technical", "Trace", "Node"
    }

    potential_entities = re.findall(r"\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+\b)*", text)
    extracted_names = []
    for entity in potential_entities:
        first_word = entity.split()[0]
        if first_word not in meta_stops and entity not in scores.keys():
            if entity not in extracted_names:
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

    st.markdown("<h1 class='centered-header'>Aether Scribe</h1>", unsafe_allow_html=True)
    st.markdown("<p class='centered-subheader'>Smart Storytelling Research Lab</p>", unsafe_allow_html=True)

    st.markdown(f"""
<div class="status-bar">
<div class="status-item"><div class="active-dot"></div> Story Brain: READY</div>
<div class="status-item"><div class="active-dot"></div> Writer: READY</div>
<div class="status-item"><div class="active-dot"></div> Data Charts: ONLINE</div>
</div>
""", unsafe_allow_html=True)

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

    if st.session_state.current_episode:
        ep = st.session_state.current_episode
        name_map = {role.capitalize(): name for role, name in ep['characters'].items()}

        tab_story, tab_research = st.tabs(["🖋️ Manuscript", "🔬 Research Dashboard"])

        with tab_story:
            st.markdown(f"<h2 style='text-align: center; color: #58a6ff; letter-spacing: 0.15em; margin-bottom: 5rem; font-weight: 300;'>YOUR {ep['genre'].upper()} STORY</h2>", unsafe_allow_html=True)

            for scene in ep['scenes']:
                content = scene['story'].split("[QUANTUM_TRACE]")
                res_line = content[0].replace(f"[SCENE {scene['step']}]", "").replace("RESULT:", "").strip()
                trace_content = content[1].strip() if len(content) > 1 else "Story logic applied."

                rejection_rows = ""
                for i, r in enumerate(scene['rejections'][:3]):
                    simple_actions = []
                    for role, act in r['action'].items():
                        name = name_map.get(role, role)
                        clean_act = act.lower().replace('_', ' ')
                        simple_actions.append(f"**{name}**: {clean_act}")
                    formatted_actions = " | ".join(simple_actions)
                    deep_reason = scene.get('deep_rejections', [])[i] if i < len(scene.get('deep_rejections', [])) else r['reason']
                    rejection_rows += f"<tr><td>{formatted_actions}</td><td>{r['prob']*100:.1f}%</td><td>{deep_reason}</td></tr>"

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

<span class="audit-header">Simple Audit: Other Choices Considered</span>
<table class="selection-table">
<thead>
<tr>
<th>What could have happened</th>
<th>Likelihood</th>
<th>Why it was skipped</th>
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

        with tab_research:
            st.subheader("📊 Research Dashboard: 12-Point Trajectory Analysis")
            st.markdown("Detailed data showing character choices and story world dynamics.")

            # Row 1
            r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
            steps = [f"S{s['step']+1}" for s in ep['scenes']]
            tension_vals = [s['state']['tension'] for s in ep['scenes']]
            reward_vals = [s['reward'] for s in ep['scenes']]
            entropy_vals = [s['entropy'] for s in ep['scenes']]

            with r1_c1:
                st.markdown("#### Story Tension")
                fig_tension = go.Figure(data=go.Scatter(x=steps, y=tension_vals, mode='lines+markers', line=dict(color='#58a6ff', width=5)))
                fig_tension.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=500, margin=dict(l=10,r=10,t=40,b=40), yaxis_range=[0,1], font_size=12)
                st.plotly_chart(fig_tension, use_container_width=True)

            with r1_c2:
                st.markdown("#### Character Power")
                agents_roles = ["Protagonist", "Antagonist", "Ally"]
                agent_names = [name_map.get(a, a) for a in agents_roles]
                vals = [ep['scenes'][-1]['attribution'][a] for a in agents_roles]
                fig2 = go.Figure(data=[go.Pie(labels=agent_names, values=vals, hole=.4, marker=dict(colors=['#58a6ff', '#f85149', '#3fb950']))])
                fig2.update_layout(showlegend=True, paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=500, margin=dict(l=10,r=10,t=40,b=40), legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5), font_size=12)
                st.plotly_chart(fig2, use_container_width=True)

            with r1_c3:
                st.markdown("#### Decision Clarity")
                gaps = []
                for s in ep['scenes']:
                    sorted_probs = sorted(s['probs'], reverse=True)
                    gaps.append(sorted_probs[0] - sorted_probs[1])
                fig_gap = go.Figure(data=go.Scatter(x=steps, y=gaps, fill='tozeroy', line=dict(color='#bc8cff', width=4)))
                fig_gap.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=500, margin=dict(l=10,r=10,t=40,b=40), font_size=12)
                st.plotly_chart(fig_gap, use_container_width=True)

            with r1_c4:
                st.markdown("#### Action Certainty")
                fig_entropy = go.Figure(data=go.Scatter(x=steps, y=entropy_vals, mode='lines+markers', line=dict(color='#d29922', width=4)))
                fig_entropy.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=500, margin=dict(l=10,r=10,t=40,b=40), font_size=12)
                st.plotly_chart(fig_entropy, use_container_width=True)

            # Row 2
            r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
            with r2_c1:
                st.markdown("#### Story Fit")
                fig_reward = go.Figure(data=go.Bar(x=steps, y=reward_vals, marker_color='#3fb950'))
                fig_reward.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=500, margin=dict(l=10,r=10,t=40,b=40), font_size=12)
                st.plotly_chart(fig_reward, use_container_width=True)

            with r2_c2:
                st.markdown("#### Personality Pulse")
                features = ["Aggro", "Strat", "Sneak", "Loyal"]
                sensitivity_vals = ep['scenes'][-1]['sensitivity']
                fig_sens = go.Figure(data=[go.Bar(x=features, y=sensitivity_vals, marker_color='#ff7b72')])
                fig_sens.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=500, margin=dict(l=10,r=10,t=40,b=40), font_size=12)
                st.plotly_chart(fig_sens, use_container_width=True)

            with r2_c3:
                st.markdown("#### Character Bonds")
                trust_vals = [s['state']['relationships']['protagonist_ally']['trust'] for s in ep['scenes']]
                enmity_vals = [s['state']['relationships']['protagonist_antagonist']['enmity'] for s in ep['scenes']]
                l_name_rel = name_map.get("Ally", "Friend")
                a_name_rel = name_map.get("Antagonist", "Villain")
                fig_rel = go.Figure()
                fig_rel.add_trace(go.Scatter(x=steps, y=trust_vals, name=f"Trust: {l_name_rel}", line=dict(color='#3fb950', width=4)))
                fig_rel.add_trace(go.Scatter(x=steps, y=enmity_vals, name=f"Hate: {a_name_rel}", line=dict(color='#f85149', width=4)))
                fig_rel.update_layout(showlegend=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=500, margin=dict(l=10,r=10,t=40,b=40), legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5), font_size=12)
                st.plotly_chart(fig_rel, use_container_width=True)

            with r2_c4:
                st.markdown("#### Control Map")
                attr_data = []
                for s in ep['scenes']:
                    attr_data.append([s['attribution'][c] for c in agents_roles])
                fig_heat = px.imshow(np.array(attr_data).T, x=steps, y=agent_names, color_continuous_scale="Viridis")
                fig_heat.update_layout(coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=500, margin=dict(l=10,r=10,t=40,b=40), font_size=12)
                st.plotly_chart(fig_heat, use_container_width=True)

            # Row 3
            r3_c1, r3_c2, r3_c3, r3_c4 = st.columns(4)
            with r3_c1:
                st.markdown("#### Plot Variety")
                fig_probs = go.Figure(data=[go.Bar(x=[f"O{i+1}" for i in range(len(ep['scenes'][-1]['probs']))], y=ep['scenes'][-1]['probs'], marker_color=['#58a6ff' if i == ep['scenes'][-1]['action_idx'] else '#30363d' for i in range(len(ep['scenes'][-1]['probs']))])])
                fig_probs.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=500, margin=dict(l=10,r=10,t=40,b=40), font_size=12)
                st.plotly_chart(fig_probs, use_container_width=True)

            with r3_c2:
                st.markdown("#### Narrative Shape")
                if HAS_PCA and len(ep['scenes']) >= 2:
                    h = [[s['state']['tension'], s['reward'], s['entropy'], s['risk']] for s in ep['scenes']]
                    pca = PCA(n_components=2); coords = pca.fit_transform(h)
                    fig3 = px.scatter(x=coords[:,0], y=coords[:,1], text=steps)
                    fig3.update_traces(marker=dict(size=20, color='#58a6ff', line=dict(width=2, color='white')), textposition='top center')
                    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=500, margin=dict(l=10,r=10,t=40,b=40), font_size=12)
                    st.plotly_chart(fig3, use_container_width=True)
                else: st.info("No data.")

            with r3_c3:
                st.markdown("#### Choice Tree")
                sources, targets, values, labels = [], [], [], []
                p_role_s, a_role_s = "Protagonist", "Antagonist"
                p_char_name_s = name_map.get(p_role_s, "Hero")
                a_char_name_s = name_map.get(a_role_s, "Villain")
                for i, scene in enumerate(ep['scenes']):
                    labels.append(f"S{i+1}")
                    curr = len(labels) - 1
                    labels.append(f"Chosen: {p_char_name_s[:10]}")
                    chosen = len(labels) - 1
                    sources.append(curr); targets.append(chosen); values.append(scene['probs'][scene['action_idx']])
                    if scene['rejections']:
                        labels.append(f"Alt: {a_char_name_s[:10]}")
                        rej = len(labels) - 1
                        sources.append(curr); targets.append(rej); values.append(scene['rejections'][0]['prob'])
                fig_sankey = go.Figure(data=[go.Sankey(node = dict(pad = 15, thickness = 20, label = labels, color = "#58a6ff", line=dict(color="white", width=0.5)), link = dict(source = sources, target = targets, value = values, color = "rgba(88, 166, 255, 0.4)"))])
                fig_sankey.update_layout(font_size=10, height=500, paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", margin=dict(l=10,r=10,t=40,b=40))
                st.plotly_chart(fig_sankey, use_container_width=True)

            with r3_c4:
                st.markdown("#### Pressure Map")
                fig_traj = px.scatter(x=tension_vals, y=reward_vals, trendline="ols", text=steps)
                fig_traj.update_traces(marker=dict(size=18, color='#58a6ff', symbol='diamond'), textposition='top center')
                fig_traj.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", height=500, margin=dict(l=10,r=10,t=40,b=40), font_size=12)
                st.plotly_chart(fig_traj, use_container_width=True)

        st.divider()
        d_col1, d_col2 = st.columns(2)
        with d_col1: st.download_button("Save Story Data (JSON)", json.dumps(ep, indent=2), file_name="story_data.json", use_container_width=True)
        with d_col2: st.download_button("Download Story (LaTeX)", generate_manuscript_latex(ep), file_name="story.tex", use_container_width=True)

    else:
        st.info("Laboratory Idle. Define a narrative seed above and click 'WRITE MY STORY' to begin.")

if __name__ == "__main__":
    main()
