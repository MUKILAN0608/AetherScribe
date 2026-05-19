RESEARCH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

.stApp {
  background: #0c0f14;
  color: #e6edf3;
}
[data-testid="stSidebar"] {
  background: #11161d !important;
  border-right: 1px solid #21262d;
}
.main .block-container {
  padding-top: 1rem !important;
  max-width: 1320px !important;
}

.platform-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
  padding-bottom: 1.25rem;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid #21262d;
}
.platform-header h1 {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 1.65rem;
  font-weight: 700;
  color: #f0f6fc;
  margin: 0;
  letter-spacing: -0.02em;
}
.platform-header .edition {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  color: #58a6ff;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-top: 0.35rem;
}
.platform-header .subtitle {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.88rem;
  color: #8b949e;
  margin: 0.25rem 0 0;
}

.status-pills {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.status-pill {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  padding: 0.35rem 0.65rem;
  border-radius: 4px;
  border: 1px solid #30363d;
  background: #161b22;
  color: #8b949e;
}
.status-pill.active {
  border-color: #238636;
  color: #3fb950;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}
@media (max-width: 960px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
.kpi-card {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 1rem 1.15rem;
}
.kpi-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.35rem;
  font-weight: 500;
  color: #58a6ff;
}
.kpi-label {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.68rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #8b949e;
  margin-top: 0.35rem;
}

.abstract-block {
  font-family: 'Source Serif 4', serif;
  font-size: 1.05rem;
  line-height: 1.75;
  color: #c9d1d9;
  padding: 1.25rem 1.5rem;
  background: #161b22;
  border: 1px solid #30363d;
  border-left: 3px solid #1f6feb;
  border-radius: 6px;
  margin-bottom: 1.5rem;
}
.abstract-label {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #58a6ff;
  margin-bottom: 0.6rem;
  display: block;
}

.manuscript-body {
  font-family: 'Source Serif 4', serif;
  font-size: 1.12rem;
  line-height: 1.9;
  color: #f0f6fc;
  padding: 2rem 2.25rem;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 8px;
  margin-bottom: 1rem;
}
.manuscript-body p {
  margin: 0 0 1.25rem;
  text-align: justify;
  text-indent: 1.75em;
}
.manuscript-body p:first-child { text-indent: 0; }

.scene-block {
  padding: 1.35rem 1.5rem;
  margin-bottom: 1rem;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 6px;
}
.scene-index {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  color: #8b949e;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.75rem;
}
.scene-prose {
  font-family: 'Source Serif 4', serif;
  font-size: 1.05rem;
  line-height: 1.85;
  color: #f0f6fc;
  margin: 0;
}

.entity-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}
@media (max-width: 768px) { .entity-grid { grid-template-columns: 1fr; } }
.entity-card {
  padding: 1rem;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 6px;
  text-align: center;
}
.entity-type {
  font-size: 0.62rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #8b949e;
  font-family: 'IBM Plex Sans', sans-serif;
}
.entity-name {
  font-family: 'Source Serif 4', serif;
  font-size: 1.1rem;
  color: #f0f6fc;
  margin: 0.35rem 0;
  font-weight: 600;
}
.entity-meta {
  font-size: 0.78rem;
  color: #8b949e;
  font-family: 'IBM Plex Sans', sans-serif;
}

.insight-panel {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.88rem;
  line-height: 1.65;
  color: #c9d1d9;
  padding: 0.85rem 1rem;
  background: #0d1117;
  border-radius: 4px;
  border-left: 3px solid #388bfd;
  margin-top: 0.75rem;
}

.decision-card {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1.25rem;
}
.decision-card h4 {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.95rem;
  font-weight: 600;
  color: #58a6ff;
  margin: 0 0 1rem 0;
}
.decision-section-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #8b949e;
  margin: 1rem 0 0.5rem 0;
}
.chosen-action {
  background: #0d1117;
  border-left: 3px solid #3fb950;
  padding: 0.75rem 1rem;
  margin: 0.5rem 0;
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.9rem;
  color: #e6edf3;
}
.alt-action {
  background: #0d1117;
  border-left: 3px solid #484f58;
  padding: 0.65rem 1rem;
  margin: 0.4rem 0;
  font-size: 0.85rem;
  color: #8b949e;
}
.decision-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem 1.5rem;
  margin-top: 0.75rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  color: #8b949e;
}
.dashboard-section {
  margin-bottom: 2rem;
}

.dash-hero {
  background: linear-gradient(135deg, #161b22 0%, #0d1117 50%, #161b22 100%);
  border: 1px solid #30363d;
  border-radius: 12px;
  padding: 1.75rem 2rem;
  margin-bottom: 1.75rem;
  position: relative;
  overflow: hidden;
}
.dash-hero::before {
  content: '';
  position: absolute;
  top: 0; right: 0;
  width: 40%;
  height: 100%;
  background: radial-gradient(ellipse at top right, rgba(88,166,255,0.12), transparent 70%);
  pointer-events: none;
}
.dash-hero h2 {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 1.35rem;
  font-weight: 700;
  color: #f0f6fc;
  margin: 0 0 0.35rem 0;
  letter-spacing: -0.02em;
}
.dash-hero p {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.88rem;
  color: #8b949e;
  margin: 0;
}
.dash-section-title {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #58a6ff;
  margin: 0 0 1rem 0;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #21262d;
}
.chart-panel {
  background: transparent;
  border: none;
  padding: 0;
  margin-bottom: 0.5rem;
  min-height: 0;
}
.chart-panel-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #8b949e;
  margin-bottom: 0.5rem;
}
.chart-empty-title {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.85rem;
  color: #8b949e;
  text-align: center;
  margin: 0.5rem 0;
}
.manuscript-full {
  white-space: pre-line;
  line-height: 2.05;
  font-size: 1.14rem;
}

/* Quantum explainability panel */
.qx-card {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 1.25rem 1.5rem;
  margin-top: 1rem;
}
.qx-header {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 1rem;
  font-weight: 600;
  color: #58a6ff;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #21262d;
}
.qx-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem 1.25rem;
  margin-bottom: 1.25rem;
}
.qx-metric { text-align: center; min-width: 4.5rem; }
.qx-metric-val {
  display: block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.1rem;
  color: #bc8cff;
}
.qx-metric-lbl {
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #8b949e;
}
.qx-section { margin-bottom: 1.15rem; }
.qx-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #8b949e;
  margin-bottom: 0.5rem;
  text-align: center;
}
.qx-legend {
  font-size: 0.82rem;
  color: #8b949e;
  line-height: 1.55;
  margin: 0 0 1rem 0;
  text-align: center;
}
.qx-note {
  font-size: 0.85rem;
  color: #d2a8ff;
  background: #1c1425;
  border: 1px solid #3d2a54;
  border-radius: 6px;
  padding: 0.55rem 0.75rem;
  margin: 0 0 1rem 0;
  line-height: 1.5;
  text-align: center;
}
.qx-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}
.qx-table td, .qx-table th {
  padding: 0.45rem 0.6rem;
  border-bottom: 1px solid #21262d;
  text-align: left;
  vertical-align: top;
}
.qx-table th {
  color: #8b949e;
  font-weight: 600;
  font-size: 0.68rem;
  text-transform: uppercase;
}
.qx-name { color: #58a6ff; font-weight: 600; width: 28%; }
.qx-action { color: #e6edf3; }
.qx-prob { color: #58a6ff; font-family: 'IBM Plex Mono', monospace; width: 4rem; }
.qx-alt-why { color: #8b949e; font-style: italic; font-size: 0.85rem; }
.qx-p { color: #c9d1d9; font-size: 0.88rem; line-height: 1.55; margin: 0.35rem 0; }
.qx-trace { color: #8b949e; font-size: 0.9rem; font-style: italic; line-height: 1.6; margin: 0; }
.qx-attr-row {
  display: grid;
  grid-template-columns: 6rem 1fr 3rem;
  align-items: center;
  gap: 0.5rem;
  margin: 0.35rem 0;
}
.qx-attr-bar {
  height: 6px;
  background: #21262d;
  border-radius: 3px;
  overflow: hidden;
}
.qx-attr-fill { height: 100%; background: linear-gradient(90deg, #388bfd, #bc8cff); }
.qx-attr-pct { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #8b949e; }
.qx-empty { color: #6e7681; text-align: center; }
.qx-alt-card {
  background: #0d1117;
  border: 1px solid #30363d;
  border-left: 3px solid #484f58;
  border-radius: 8px;
  padding: 0.85rem 1rem;
  margin-bottom: 0.65rem;
}
.qx-alt-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}
.qx-alt-num {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  color: #58a6ff;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.qx-alt-prob {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.75rem;
  color: #bc8cff;
}
.qx-alt-actions {
  font-size: 0.88rem;
  color: #e6edf3;
  line-height: 1.5;
  margin-bottom: 0.5rem;
}
.qx-alt-reason {
  font-size: 0.85rem;
  color: #8b949e;
  line-height: 1.55;
}
.qx-alt-reason-text {
  margin: 0.25rem 0 0;
  font-style: italic;
  color: #c9d1d9;
  line-height: 1.6;
}
.qx-alt-reason-lbl {
  display: block;
  font-style: normal;
  font-weight: 600;
  color: #c9d1d9;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.25rem;
}
.kpi-grid-extended {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 0.65rem;
  margin-bottom: 1.5rem;
}
@media (max-width: 1100px) { .kpi-grid-extended { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 600px) { .kpi-grid-extended { grid-template-columns: repeat(2, 1fr); } }
.kpi-card.accent { border-color: #388bfd; }
.kpi-card.quantum { border-color: #bc8cff; }
.kpi-value.quantum { color: #bc8cff; }

.scene-prose-lines {
  font-family: 'Source Serif 4', serif;
  font-size: 1.08rem;
  line-height: 2;
  color: #f0f6fc;
  margin: 0;
  white-space: pre-line;
}
.manuscript-hero {
  font-family: 'Source Serif 4', serif;
  font-size: 1.14rem;
  line-height: 2.05;
  color: #f0f6fc;
  padding: 2rem 2.5rem;
  background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
  border: 1px solid #30363d;
  border-radius: 10px;
  margin-bottom: 2rem;
  white-space: pre-line;
}
.manuscript-hero .scene-divider {
  display: block;
  height: 1px;
  background: #21262d;
  margin: 1.5rem 0;
  border: none;
}
.scene-num {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.6rem;
  color: #58a6ff;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  display: block;
  margin-bottom: 0.75rem;
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
}

.idle-panel {
  text-align: center;
  padding: 4rem 2rem;
  background: #161b22;
  border: 1px dashed #30363d;
  border-radius: 8px;
  color: #8b949e;
  font-family: 'IBM Plex Sans', sans-serif;
}

.stButton > button[kind="primary"] {
  background: #1f6feb !important;
  border: 1px solid #388bfd !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-weight: 600 !important;
  border-radius: 6px !important;
}
.stButton > button[kind="primary"]:hover {
  background: #388bfd !important;
}

#MainMenu, footer { visibility: hidden; }
</style>
"""
