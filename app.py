import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.generate_data import generate_full_dataset
from utils.features import FEATURE_COLS, engineer_features
from models.ml_pipeline import (run_full_pipeline, predict_transactions,
                                  self_learning_retrain, LOW_CONFIDENCE_THRESHOLD,
                                  get_shap_values)
from ledger.immutable_ledger import add_to_ledger, verify_chain, get_ledger_df, clear_ledger

st.set_page_config(page_title="AuditOS · ExpenseGuard", page_icon="⬡",
                   layout="wide", initial_sidebar_state="expanded")

# ── THEME CONSTANTS ────────────────────────────────────────────────────────────
BG      = "#0c0c0f"
SURFACE = "#13131a"
CARD    = "#1a1a24"
BORDER  = "#2a2a38"
ACCENT  = "#e8a020"        # amber/gold
ACCENT2 = "#c45c2a"        # burnt orange
GREEN   = "#22c55e"
RED     = "#ef4444"
PURPLE  = "#a78bfa"
BLUE    = "#60a5fa"
TEXT    = "#f0ede8"
TEXT_MID= "#a09880"
TEXT_DIM= "#4a4840"

PT = dict(layout=dict(
    paper_bgcolor=SURFACE, plot_bgcolor=BG,
    font=dict(family="'Syne', sans-serif", color=TEXT_MID, size=11),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False,
               tickfont=dict(color=TEXT_DIM)),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False,
               tickfont=dict(color=TEXT_DIM)),
    legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                font=dict(color=TEXT_MID)),
    margin=dict(l=16, r=16, t=36, b=16),
))

ACOLORS = {
    "Normal": BLUE, "Duplicate": RED,
    "Policy Violation": ACCENT, "Ghost Vendor": PURPLE,
    "Redundant Spending": GREEN
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'Syne', sans-serif !important;
    box-sizing: border-box;
}}
.stApp, .main {{ background: {BG} !important; color: {TEXT}; }}
.block-container {{ padding: 1.5rem 2rem 2rem !important; max-width: 100% !important; }}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {{
    background: {SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
    width: 240px !important;
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_MID} !important; }}
section[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.78rem !important;
    padding: 6px 10px !important;
    border-radius: 6px !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: {CARD} !important;
    color: {TEXT} !important;
}}
.sidebar-brand {{
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    color: {ACCENT} !important;
    text-transform: uppercase !important;
    padding: 4px 0 16px !important;
}}
.sidebar-stat {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    margin: 4px 0 !important;
    font-size: 0.72rem !important;
}}
.sidebar-stat-label {{
    color: {TEXT_DIM} !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}}
.sidebar-stat-value {{
    color: {TEXT} !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}}

/* ── HEADER ── */
.top-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 20px 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 24px;
}}
.logo-text {{
    font-family: 'Space Mono', monospace !important;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: {TEXT};
}}
.logo-text span {{ color: {ACCENT}; }}
.header-pill {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.68rem;
    color: {TEXT_DIM};
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.header-pill.live {{
    border-color: {GREEN};
    color: {GREEN};
}}

/* ── SECTION LABELS ── */
.sec-label {{
    font-family: 'Space Mono', monospace !important;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {TEXT_DIM};
    margin: 28px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.sec-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {BORDER};
}}

/* ── KPI CARDS ── */
.kpi-row {{ display: flex; gap: 12px; margin-bottom: 20px; }}
.kpi-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 18px 20px;
    flex: 1;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
    animation: fadeSlideUp 0.4s ease both;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-color, {BORDER});
    border-radius: 10px 10px 0 0;
}}
.kpi-card:hover {{ border-color: {ACCENT}; }}
.kpi-value {{
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem;
    font-weight: 700;
    color: {TEXT};
    line-height: 1;
    margin-bottom: 6px;
}}
.kpi-label {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXT_DIM};
}}
.kpi-delta {{
    font-size: 0.72rem;
    margin-top: 4px;
}}
.kpi-delta.up {{ color: {RED}; }}
.kpi-delta.ok {{ color: {GREEN}; }}
.kpi-delta.warn {{ color: {ACCENT}; }}

/* ── ANOMALY CHIPS ── */
.chip {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace !important;
}}
.chip-dup   {{ background: rgba(239,68,68,0.12);  color: {RED};    border: 1px solid rgba(239,68,68,0.25); }}
.chip-pol   {{ background: rgba(232,160,32,0.12); color: {ACCENT}; border: 1px solid rgba(232,160,32,0.25); }}
.chip-gho   {{ background: rgba(167,139,250,0.12);color: {PURPLE}; border: 1px solid rgba(167,139,250,0.25); }}
.chip-red   {{ background: rgba(34,197,94,0.12);  color: {GREEN};  border: 1px solid rgba(34,197,94,0.25); }}

/* ── LEDGER BLOCKS ── */
.ledger-block {{
    background: {BG};
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem;
    color: {TEXT_MID};
    animation: fadeSlideUp 0.3s ease both;
}}

/* ── BUTTONS ── */
.stButton > button {{
    background: transparent !important;
    color: {ACCENT} !important;
    border: 1px solid {ACCENT} !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}}
.stButton > button:hover {{
    background: {ACCENT} !important;
    color: {BG} !important;
}}

/* ── TRAIN BUTTON (PRIMARY) ── */
.stButton:first-of-type > button {{
    background: {ACCENT} !important;
    color: {BG} !important;
    border-color: {ACCENT} !important;
    font-weight: 700 !important;
    width: 100% !important;
}}
.stButton:first-of-type > button:hover {{
    background: #d4900a !important;
}}

/* ── DATAFRAME ── */
.stDataFrame {{
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}}

/* ── INFO / WARNING ── */
.stAlert {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT_MID} !important;
}}

/* ── EXPANDER ── */
.streamlit-expanderHeader {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    color: {TEXT_MID} !important;
}}

/* ── METRIC ── */
[data-testid="metric-container"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 16px !important;
}}
[data-testid="metric-container"] label {{
    font-size: 0.62rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: {TEXT_DIM} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important;
    color: {TEXT} !important;
}}

/* ── ANIMATIONS ── */
@keyframes fadeSlideUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.5; }}
}}
.live-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: {GREEN};
    display: inline-block; margin-right: 6px;
    animation: pulse 2s infinite;
}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {CARD} !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 6px !important;
    color: {TEXT_DIM} !important;
    font-size: 0.72rem !important;
    padding: 6px 16px !important;
}}
.stTabs [aria-selected="true"] {{
    background: {BORDER} !important;
    color: {TEXT} !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def sec(label, icon=""):
    st.markdown(f'<div class="sec-label">{icon}&nbsp;{label}</div>', unsafe_allow_html=True)

def kpi(value, label, delta="", delta_type="ok", accent=ACCENT):
    return f"""<div class="kpi-card" style="--accent-color:{accent}">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {'<div class="kpi-delta ' + delta_type + '">' + delta + '</div>' if delta else ''}
    </div>"""

def plotly_fig(fig, height=300, title=""):
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color=TEXT_MID, size=12)))
    fig.update_layout(**PT["layout"], height=height)
    st.plotly_chart(fig, use_container_width=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key, val in [("df_raw", None), ("df_pred", None), ("metrics_before", None),
                  ("metrics_after", None), ("best_model", None),
                  ("pipeline_run", False), ("low_conf_reviewed", {})]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="sidebar-brand">⬡ &nbsp;AuditOS</div>', unsafe_allow_html=True)

    page = st.radio("", [
        "Overview", "Anomaly Detection", "Self-Learning",
        "Model Performance", "XAI Explainability",
        "Immutable Ledger", "Analytics",
        "Risk Leaderboard", "Impact Calculator"
    ], label_visibility="collapsed")

    st.markdown("---")

    if st.button("⚡  Initialize System"):
        clear_ledger()
        with st.spinner("Generating dataset..."):
            df = generate_full_dataset(n=1500)
            st.session_state.df_raw = df
        with st.spinner("Training models with SMOTE balancing..."):
            models_dict, metrics_before, best = run_full_pipeline(df, FEATURE_COLS)
            st.session_state.metrics_before = metrics_before
            st.session_state.best_model = best
        with st.spinner("Running predictions..."):
            df_pred = predict_transactions(df, FEATURE_COLS)
            st.session_state.df_pred = df_pred
            st.session_state.pipeline_run = True
            flagged = df_pred[(df_pred["prediction"] == 1) & (df_pred["confidence"] >= 0.80)]
            count = 0
            for _, row in flagged.iterrows():
                add_to_ledger(row.to_dict(), row.get("anomaly_type","Anomaly"),
                              row["confidence"], row["model_used"])
                count += 1
        st.success(f"Done — {count} blocks secured")

    st.markdown("---")
    if st.session_state.pipeline_run:
        dp = st.session_state.df_pred
        chain = verify_chain()
        bm = st.session_state.best_model or ""
        auc = st.session_state.metrics_before.get(bm, {}).get("roc_auc", 0)
        nf = int((dp["prediction"] == 1).sum())
        for label, value in [
            ("Records", f"{len(dp):,}"),
            ("Flagged", f"{nf:,}"),
            ("Ledger", f"{'✓' if chain['valid'] else '✗'} {chain['length']} blocks"),
            ("Best AUC", f"{auc:.4f}"),
            ("Model", bm[:16] if bm else "—"),
        ]:
            st.markdown(f"""<div class="sidebar-stat">
                <div class="sidebar-stat-label">{label}</div>
                <div class="sidebar-stat-value">{value}</div>
            </div>""", unsafe_allow_html=True)

# ── TOP HEADER ────────────────────────────────────────────────────────────────
chain_status = verify_chain()
st.markdown(f"""
<div class="top-header">
    <div>
        <div class="logo-text">Expense<span>Guard</span></div>
        <div style="font-size:0.65rem;color:{TEXT_DIM};letter-spacing:0.08em;margin-top:2px;">
            SELF-LEARNING AUDIT SYSTEM &nbsp;·&nbsp; IMMUTABLE LEDGER
        </div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
        <div class="header-pill">SHA-256 Ledger</div>
        <div class="header-pill">SMOTE Balanced</div>
        <div class="header-pill">SHAP Explainable</div>
        <div class="header-pill {'live' if st.session_state.pipeline_run else ''}">
            <span class="live-dot"></span>
            {'LIVE' if st.session_state.pipeline_run else 'IDLE'}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    if not st.session_state.pipeline_run:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;
            padding:32px;text-align:center;margin:40px 0;">
            <div style="font-size:2rem;margin-bottom:12px;">⬡</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.85rem;
                color:{ACCENT};margin-bottom:8px;">SYSTEM IDLE</div>
            <div style="color:{TEXT_DIM};font-size:0.82rem;">
                Click <strong style="color:{TEXT}">Initialize System</strong> in the sidebar to begin
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:8px;">
        """, unsafe_allow_html=True)

        for icon, label, desc in [
            ("◈", "Hybrid ML", "Random Forest · Gradient Boosting · Extra Trees · Isolation Forest · One-Class SVM"),
            ("⟳", "Self-Learning", "Confidence scoring · Human-in-the-loop · Automatic retraining"),
            ("⬡", "SHA-256 Ledger", "Hash-chained blocks · AES encryption · Tamper detection"),
            ("◎", "SHAP XAI", "Global feature importance · Waterfall explanations · Per-transaction reasoning"),
        ]:
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:20px;">
                <div style="font-family:'Space Mono',monospace;font-size:1.2rem;
                    color:{ACCENT};margin-bottom:8px;">{icon}</div>
                <div style="font-size:0.82rem;font-weight:600;color:{TEXT};margin-bottom:6px;">{label}</div>
                <div style="font-size:0.72rem;color:{TEXT_DIM};line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        df_pred = st.session_state.df_pred
        total   = len(df_pred)
        flagged = int((df_pred["prediction"] == 1).sum())
        low_c   = int((df_pred["confidence_level"].astype(str) == "Low").sum())
        chain   = verify_chain()
        bm      = st.session_state.best_model
        auc     = st.session_state.metrics_before[bm]["roc_auc"]
        flagged_df = df_pred[df_pred["prediction"] == 1]
        flag_amt = flagged_df["amount"].sum()
        total_amt = df_pred["amount"].sum()

        sec("Key Metrics", "◈")
        st.markdown(f"""<div class="kpi-row">
            {kpi(f"{total:,}", "Total Transactions", "", "ok", BLUE)}
            {kpi(f"{flagged:,}", "Anomalies Flagged", f"▲ {flagged/total*100:.1f}% of total", "up", RED)}
            {kpi(f"${flag_amt/1e6:.2f}M", "At-Risk Amount", f"{flag_amt/total_amt*100:.1f}% of spend", "up", ACCENT)}
            {kpi(f"{low_c:,}", "Need Review", "Low confidence", "warn", PURPLE)}
            {kpi(f"{chain['length']}", "Ledger Blocks", "SHA-256 secured", "ok", GREEN)}
            {kpi(f"{auc:.4f}", "Best AUC", bm[:14], "ok", ACCENT)}
        </div>""", unsafe_allow_html=True)

        sec("Anomaly Breakdown", "◑")
        col1, col2 = st.columns(2)
        with col1:
            dist = flagged_df["anomaly_type"].value_counts().reset_index()
            dist.columns = ["type","count"]
            fig = px.pie(dist, names="type", values="count", color="type",
                         color_discrete_map=ACOLORS, hole=0.6)
            fig.update_traces(textposition="outside", textinfo="label+percent",
                              textfont_size=10)
            plotly_fig(fig, 300, "Flagged by anomaly type")
        with col2:
            dept_flag = flagged_df.groupby("department").size().reset_index(name="count")
            fig2 = px.bar(dept_flag.sort_values("count", ascending=True),
                          x="count", y="department", orientation="h",
                          color_discrete_sequence=[ACCENT])
            fig2.update_traces(marker_line_width=0)
            plotly_fig(fig2, 300, "Flagged by department")

        sec("Monthly Timeline", "◷")
        daily = df_pred[df_pred["prediction"]==1].copy()
        daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M").astype(str)
        monthly = daily.groupby(["month","anomaly_type"]).size().reset_index(name="count")
        fig3 = px.bar(monthly, x="month", y="count", color="anomaly_type",
                      color_discrete_map=ACOLORS)
        fig3.update_traces(marker_line_width=0)
        plotly_fig(fig3, 260, "")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Detection":
    if not st.session_state.pipeline_run:
        st.warning("Initialize the system first.")
    else:
        df_pred = st.session_state.df_pred
        sec("Filter Transactions", "⊹")

        col1, col2, col3 = st.columns(3)
        with col1:
            atype = st.multiselect("Anomaly Type", df_pred["anomaly_type"].unique().tolist(),
                                   default=["Duplicate","Ghost Vendor","Policy Violation","Redundant Spending"])
        with col2:
            conf_lvl = st.multiselect("Confidence", ["High","Medium","Low"], default=["High","Medium"])
        with col3:
            dept_sel = st.multiselect("Department", sorted(df_pred["department"].unique()),
                                       default=sorted(df_pred["department"].unique()))

        mask = ((df_pred["anomaly_type"].isin(atype)) &
                (df_pred["confidence_level"].astype(str).isin(conf_lvl)) &
                (df_pred["department"].isin(dept_sel)) &
                (df_pred["prediction"] == 1))
        filtered = df_pred[mask]

        st.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:0.7rem;
            color:{ACCENT};margin:8px 0;letter-spacing:0.05em;">
            ◈ {len(filtered):,} transactions matched
            &nbsp;·&nbsp; At-risk: ${filtered['amount'].sum():,.0f}
        </div>""", unsafe_allow_html=True)

        display_cols = ["transaction_id","date","employee_id","department",
                        "vendor","category","amount","anomaly_type","confidence","confidence_level"]
        available = [c for c in display_cols if c in filtered.columns]
        show_df = filtered[available].copy()
        show_df["amount"]     = show_df["amount"].apply(lambda x: f"${x:,.2f}")
        show_df["confidence"] = show_df["confidence"].apply(lambda x: f"{x:.1%}")
        st.dataframe(show_df.head(200), use_container_width=True, height=380)

        col1, col2 = st.columns(2)
        with col1:
            sec("Amount vs Confidence", "◎")
            fig = px.scatter(filtered, x="confidence", y="amount", color="anomaly_type",
                             color_discrete_map=ACOLORS,
                             hover_data=["employee_id","vendor","department"], opacity=0.7)
            fig.add_vline(x=LOW_CONFIDENCE_THRESHOLD, line_dash="dash",
                          line_color=ACCENT, annotation_text="Review threshold",
                          annotation_font_color=ACCENT)
            fig.update_traces(marker_size=6)
            plotly_fig(fig, 320)
        with col2:
            sec("Top Flagged Vendors", "◈")
            top_v = (filtered.groupby("vendor")
                     .agg(count=("transaction_id","count"), total=("amount","sum"))
                     .sort_values("count", ascending=False).head(10).reset_index())
            fig_v = px.bar(top_v, x="count", y="vendor", orientation="h",
                           color_discrete_sequence=[ACCENT2])
            fig_v.update_traces(marker_line_width=0)
            plotly_fig(fig_v, 320)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SELF-LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Self-Learning":
    if not st.session_state.pipeline_run:
        st.warning("Initialize the system first.")
    else:
        df_pred = st.session_state.df_pred
        low_conf = df_pred[df_pred["confidence_level"].astype(str) == "Low"].copy()

        sec("Confidence Distribution", "◑")
        st.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:0.7rem;
            color:{ACCENT};margin-bottom:12px;">
            {len(low_conf):,} transactions below threshold — these drive the self-learning loop
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns([2,1])
        with col1:
            fig = px.histogram(df_pred, x="confidence", color="confidence_level",
                               color_discrete_map={"High":RED,"Medium":ACCENT,"Low":GREEN}, nbins=40)
            fig.add_vline(x=LOW_CONFIDENCE_THRESHOLD, line_dash="dash", line_color=TEXT_MID)
            plotly_fig(fig, 260, "Confidence score distribution")
        with col2:
            breakdown = df_pred["confidence_level"].astype(str).value_counts().reset_index()
            breakdown.columns = ["level","count"]
            fig2 = px.pie(breakdown, names="level", values="count", hole=0.6,
                          color="level",
                          color_discrete_map={"High":RED,"Medium":ACCENT,"Low":GREEN})
            plotly_fig(fig2, 260)

        sec("Review Low-Confidence Cases", "⟳")
        st.markdown(f'<div style="font-size:0.75rem;color:{TEXT_DIM};margin-bottom:12px;">Correct labels below → samples retrain the model automatically</div>', unsafe_allow_html=True)

        review_sample = low_conf.sample(min(15, len(low_conf)), random_state=1)
        for idx, row in review_sample.iterrows():
            with st.expander(f"▸  {row['transaction_id']}  ·  {row.get('vendor','?')}  ·  ${row.get('amount',0):,.2f}  ·  {row.get('department','')}"):
                cols = st.columns([2,1,1])
                cols[0].markdown(f"**Predicted:** `{row.get('anomaly_type','?')}` · **Confidence:** `{row['confidence']:.1%}`")
                label = cols[1].selectbox("Label", ["Anomaly (1)","Normal (0)"],
                                          key=f"lbl_{idx}",
                                          index=0 if row.get("prediction",1)==1 else 1)
                if cols[2].button("Confirm", key=f"conf_{idx}"):
                    st.session_state.low_conf_reviewed[idx] = 1 if "Anomaly" in label else 0

        reviewed = len(st.session_state.low_conf_reviewed)
        st.markdown(f'<div style="font-size:0.75rem;color:{GREEN};margin:8px 0;">{reviewed} samples confirmed for retraining</div>', unsafe_allow_html=True)

        if st.button("⟳  Trigger Retraining") and reviewed > 0:
            reviewed_df = review_sample[review_sample.index.isin(
                st.session_state.low_conf_reviewed.keys())].copy()
            reviewed_df["label"] = reviewed_df.index.map(st.session_state.low_conf_reviewed)
            with st.spinner("Retraining with corrected labels..."):
                results_after, best_after = self_learning_retrain(reviewed_df, FEATURE_COLS)
            if results_after:
                st.session_state.metrics_after = results_after
                st.success(f"Retrained · Best model: {best_after}")
                st.session_state.low_conf_reviewed = {}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    if not st.session_state.pipeline_run or not st.session_state.metrics_before:
        st.warning("Initialize the system first.")
    else:
        metrics_before = st.session_state.metrics_before
        metrics_after  = st.session_state.metrics_after
        metric_names   = ["accuracy","precision","recall","f1","roc_auc"]

        sec("Model Metrics", "◈")
        rows = []
        for mname, m in metrics_before.items():
            row = {"Model": mname}
            for mn in metric_names:
                row[mn.upper()] = f"{m.get(mn,0):.4f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            sec("Radar Comparison", "◎")
            fig = go.Figure()
            model_colors = [ACCENT, RED, GREEN, PURPLE, BLUE]
            for i, (mname, m) in enumerate(metrics_before.items()):
                vals = [m.get(mn,0) for mn in metric_names] + [m.get(metric_names[0],0)]
                cats = [mn.upper() for mn in metric_names] + [metric_names[0].upper()]
                fig.add_trace(go.Scatterpolar(r=vals, theta=cats, name=mname,
                                              fill="toself", opacity=0.5,
                                              line_color=model_colors[i % len(model_colors)]))
            fig.update_layout(
                polar=dict(bgcolor=BG,
                           radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER,
                                          tickfont=dict(color=TEXT_DIM, size=9)),
                           angularaxis=dict(gridcolor=BORDER,
                                           tickfont=dict(color=TEXT_MID, size=10))),
                paper_bgcolor=SURFACE, font=dict(color=TEXT_MID, family="Syne"),
                height=360, legend=dict(bgcolor=SURFACE, bordercolor=BORDER))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sec("F1 Score Comparison", "◑")
            models = list(metrics_before.keys())
            f1s = [metrics_before[m].get("f1",0) for m in models]
            fig2 = go.Figure(go.Bar(
                x=models, y=f1s,
                marker_color=[ACCENT, RED, GREEN, PURPLE, BLUE][:len(models)],
                marker_line_width=0,
                text=[f"{v:.4f}" for v in f1s], textposition="outside",
                textfont=dict(color=TEXT_MID, size=10)
            ))
            fig2.update_layout(**PT["layout"], height=360, yaxis_range=[0, 1.05])
            st.plotly_chart(fig2, use_container_width=True)

        if metrics_after:
            sec("Before vs After Retraining", "⟳")
            sup_models = [m for m in metrics_before if m not in ("Isolation Forest","One-Class SVM")]
            for metric in ["f1", "precision", "recall", "roc_auc"]:
                before_vals = [metrics_before[m].get(metric,0) for m in sup_models]
                after_vals  = [metrics_after.get(m,{}).get(metric,0) for m in sup_models]
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(name="Before", x=sup_models, y=before_vals,
                                      marker_color=BLUE, opacity=0.7, marker_line_width=0))
                fig3.add_trace(go.Bar(name="After", x=sup_models, y=after_vals,
                                      marker_color=GREEN, opacity=0.85, marker_line_width=0))
                fig3.update_layout(**PT["layout"], barmode="group", height=220,
                                   yaxis_range=[0, 1.05],
                                   title=dict(text=metric.upper(),
                                              font=dict(color=TEXT_MID, size=11)))
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Complete Self-Learning retraining to see before/after comparison.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: XAI EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "XAI Explainability":
    if not st.session_state.pipeline_run:
        st.warning("Initialize the system first.")
    else:
        sec("SHAP — Global Feature Importance", "◎")
        st.markdown(f'<div style="font-size:0.75rem;color:{TEXT_DIM};margin-bottom:16px;">SHAP (SHapley Additive exPlanations) — each feature\'s average contribution to the model\'s fraud prediction, based on game theory.</div>', unsafe_allow_html=True)

        df_raw = st.session_state.df_raw
        with st.spinner("Computing SHAP values..."):
            shap_vals, X_data = get_shap_values(df_raw, FEATURE_COLS)

        if shap_vals is not None:
            mean_shap = np.abs(shap_vals).mean(axis=0)
            shap_df = pd.DataFrame({
                "Feature": FEATURE_COLS, "Mean |SHAP|": mean_shap
            }).sort_values("Mean |SHAP|", ascending=True).tail(15)

            fig = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                         color="Mean |SHAP|",
                         color_continuous_scale=[[0, BORDER],[0.5, ACCENT2],[1, ACCENT]])
            fig.update_traces(marker_line_width=0)
            fig.update_layout(**PT["layout"], height=460,
                              coloraxis_showscale=False)
            plotly_fig(fig, 460, "Top 15 Features by Mean |SHAP|")

            sec("Feature Ranking", "◈")
            rank_df = shap_df.sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
            rank_df.index += 1
            rank_df["Mean |SHAP|"] = rank_df["Mean |SHAP|"].apply(lambda x: f"{x:.6f}")
            st.dataframe(rank_df, use_container_width=True)

            sec("SHAP Distribution — Top 6 Features", "◑")
            top6_idx = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1][:6]
            cols = st.columns(3)
            for i, feat_idx in enumerate(top6_idx):
                feat_name = FEATURE_COLS[feat_idx]
                sv = shap_vals[:, feat_idx]
                with cols[i % 3]:
                    fig_h = px.histogram(x=sv, nbins=30, color_discrete_sequence=[ACCENT])
                    fig_h.add_vline(x=0, line_dash="dash", line_color=RED)
                    fig_h.update_layout(**PT["layout"], height=200, showlegend=False,
                                       title=dict(text=feat_name, font=dict(color=TEXT_MID, size=10)))
                    st.plotly_chart(fig_h, use_container_width=True)

            sec("Single Transaction Waterfall", "⟁")
            df_pred = st.session_state.df_pred
            flagged_idx = df_pred[df_pred["prediction"]==1]["confidence"].idxmax()
            txn_pos = df_pred.index.get_loc(flagged_idx)
            sv_single = shap_vals[txn_pos]
            wf_df = pd.DataFrame({
                "Feature": FEATURE_COLS, "SHAP Value": sv_single
            }).sort_values("SHAP Value", key=abs, ascending=True).tail(12)
            colors = [RED if v > 0 else GREEN for v in wf_df["SHAP Value"]]
            fig_wf = go.Figure(go.Bar(x=wf_df["SHAP Value"], y=wf_df["Feature"],
                                      orientation="h", marker_color=colors,
                                      marker_line_width=0))
            fig_wf.add_vline(x=0, line_color=BORDER)
            txn = df_pred.loc[flagged_idx]
            plotly_fig(fig_wf, 380, f"Why was {txn['transaction_id']} flagged?")
            st.markdown(f'<div style="font-size:0.72rem;color:{TEXT_DIM};">Amount: ${txn["amount"]:,.2f} · Dept: {txn["department"]} · Type: {txn["anomaly_type"]} · Confidence: {txn["confidence"]:.1%}</div>', unsafe_allow_html=True)
        else:
            st.error("Could not compute SHAP values. Run: pip install shap")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: IMMUTABLE LEDGER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Immutable Ledger":
    chain_info = verify_chain()
    sec("Chain Status", "⬡")

    sc = GREEN if chain_info["valid"] else RED
    st.markdown(f"""<div class="kpi-row">
        {kpi("VALID" if chain_info["valid"] else "BROKEN", "Chain Integrity", "SHA-256 verified", "ok", sc)}
        {kpi(str(chain_info["length"]), "Total Blocks", "Immutable records", "ok", ACCENT)}
        {kpi("SHA-256", "Hash Algorithm", "Bitcoin-grade security", "ok", PURPLE)}
        {kpi("AES/Fernet", "PII Encryption", "Employee & vendor data", "ok", BLUE)}
    </div>""", unsafe_allow_html=True)

    ledger_df = get_ledger_df()
    if ledger_df.empty:
        st.info("No entries yet. Initialize the system to populate.")
    else:
        sec("Ledger Entries", "◈")
        st.dataframe(ledger_df, use_container_width=True, height=320)

        sec("Hash Chain Visualization", "⟁")
        from ledger.immutable_ledger import load_ledger
        raw_chain = load_ledger()
        for block in raw_chain[:8]:
            st.markdown(f"""<div class="ledger-block">
                <span style="color:{TEXT_DIM}">Block</span>
                <span style="color:{ACCENT};font-weight:700"> #{block['index']}</span>
                &nbsp;·&nbsp;<span style="color:{TEXT}">{block['transaction_id']}</span>
                &nbsp;·&nbsp;<span style="color:{ACCENT}">{block['anomaly_type']}</span>
                &nbsp;·&nbsp;<span style="color:{GREEN}">${float(block['amount']):,.0f}</span>
                &nbsp;·&nbsp;<span style="color:{TEXT_DIM}">{block['timestamp'][:19]}</span><br>
                <span style="color:{TEXT_DIM}">hash&nbsp;&nbsp;</span><span style="color:{ACCENT2}">{block['hash'][:48]}...</span><br>
                <span style="color:{TEXT_DIM}">prev&nbsp;&nbsp;</span><span style="color:{BORDER}">{block['prev_hash'][:48]}...</span>
            </div>""", unsafe_allow_html=True)

        sec("Add Manual Entry", "⊹")
        with st.form("manual_ledger"):
            c1, c2, c3 = st.columns(3)
            txn_id   = c1.text_input("Transaction ID", value="TXN_MANUAL_001")
            amount   = c2.number_input("Amount ($)", value=5000.0)
            atype    = c3.selectbox("Anomaly Type", ["Duplicate","Ghost Vendor","Policy Violation","Redundant Spending"])
            dept     = c1.text_input("Department", value="Finance")
            emp      = c2.text_input("Employee ID", value="EMP0042")
            conf_val = c3.slider("Confidence", 0.0, 1.0, 0.9)
            if st.form_submit_button("⬡  Add to Ledger"):
                block = add_to_ledger({"transaction_id":txn_id,"amount":amount,
                                       "department":dept,"employee_id":emp,
                                       "vendor":"Manual","category":"Other","date":"2024-01-01"},
                                      atype, conf_val, "Manual")
                st.success(f"Block #{block['index']} sealed · Hash: {block['hash'][:32]}...")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":
    if not st.session_state.pipeline_run:
        st.warning("Initialize the system first.")
    else:
        df_pred = st.session_state.df_pred
        flagged = df_pred[df_pred["prediction"] == 1]
        total_amt   = df_pred["amount"].sum()
        flagged_amt = flagged["amount"].sum()

        sec("Financial Impact", "◈")
        st.markdown(f"""<div class="kpi-row">
            {kpi(f"${flagged_amt/1e6:.2f}M", "Total At-Risk", f"{flagged_amt/total_amt*100:.1f}% of spend", "up", RED)}
            {kpi(f"${flagged['amount'].mean():,.0f}", "Avg Flagged Amount", "per transaction", "warn", ACCENT)}
            {kpi(f"${total_amt/1e6:.2f}M", "Total Spend", "all transactions", "ok", BLUE)}
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            sec("Spend by Category", "◑")
            cat_spend = df_pred.groupby("category")["amount"].sum().reset_index().sort_values("amount")
            fig = px.bar(cat_spend, x="amount", y="category", orientation="h",
                         color_discrete_sequence=[ACCENT])
            fig.update_traces(marker_line_width=0)
            plotly_fig(fig, 300)
        with col2:
            sec("Spend by Department", "◑")
            dept_spend = df_pred.groupby("department")["amount"].sum().reset_index().sort_values("amount")
            fig2 = px.bar(dept_spend, x="amount", y="department", orientation="h",
                          color_discrete_sequence=[PURPLE])
            fig2.update_traces(marker_line_width=0)
            plotly_fig(fig2, 300)

        sec("Anomaly Heatmap: Department × Category", "⊹")
        heat = flagged.groupby(["department","category"]).size().reset_index(name="count")
        heat_pivot = heat.pivot(index="department", columns="category", values="count").fillna(0)
        fig3 = px.imshow(heat_pivot,
                         color_continuous_scale=[[0,BG],[0.5,"#3a2800"],[1,ACCENT]],
                         text_auto=True, aspect="auto")
        fig3.update_layout(**PT["layout"], height=340,
                           coloraxis_colorbar=dict(tickfont=dict(color=TEXT_DIM)))
        st.plotly_chart(fig3, use_container_width=True)

        sec("Top High-Risk Employees", "◈")
        emp_risk = (flagged.groupby("employee_id")
                    .agg(flags=("transaction_id","count"), total=("amount","sum"), dept=("department","first"))
                    .sort_values("flags", ascending=False).head(15).reset_index())
        emp_risk["total"] = emp_risk["total"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(emp_risk, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RISK LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Leaderboard":
    if not st.session_state.pipeline_run:
        st.warning("Initialize the system first.")
    else:
        df_pred = st.session_state.df_pred
        flagged = df_pred[df_pred["prediction"] == 1].copy()

        sec("Employee Risk Rankings", "◈")

        emp_stats = (flagged.groupby("employee_id")
            .agg(flags=("transaction_id","count"), total_risk=("amount","sum"),
                 avg_conf=("confidence","mean"), dept=("department","first"),
                 anomaly_types=("anomaly_type", lambda x: " · ".join(x.unique())))
            .reset_index())
        max_risk = emp_stats["total_risk"].max() or 1
        emp_stats["risk_score"] = (
            emp_stats["flags"] / emp_stats["flags"].max() * 40 +
            emp_stats["total_risk"] / max_risk * 40 +
            emp_stats["avg_conf"] * 20
        ).round(1)
        emp_stats = emp_stats.sort_values("risk_score", ascending=False).reset_index(drop=True)

        def risk_badge(s):
            if s >= 70: return "CRITICAL"
            elif s >= 50: return "HIGH"
            elif s >= 30: return "MEDIUM"
            else: return "LOW"
        def risk_color(s):
            if s >= 70: return RED
            elif s >= 50: return ACCENT2
            elif s >= 30: return ACCENT
            else: return GREEN

        # Top 3 cards
        top3 = emp_stats.head(3)
        medals = ["01", "02", "03"]
        cols = st.columns(3)
        for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
            rc = risk_color(row["risk_score"])
            col.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-top:2px solid {rc};
                border-radius:10px;padding:20px;text-align:center;animation:fadeSlideUp 0.4s ease both;">
                <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                    color:{TEXT_DIM};letter-spacing:0.1em;margin-bottom:12px;">RANK {medals[i]}</div>
                <div style="font-family:'Space Mono',monospace;color:{rc};font-size:1rem;
                    font-weight:700;margin-bottom:4px;">{row['employee_id']}</div>
                <div style="color:{TEXT_DIM};font-size:0.7rem;margin-bottom:12px;">{row['dept']}</div>
                <div style="font-family:'Space Mono',monospace;font-size:2rem;
                    font-weight:700;color:{TEXT};margin-bottom:4px;">{row['risk_score']:.0f}</div>
                <div style="font-size:0.6rem;color:{TEXT_DIM};letter-spacing:0.08em;">RISK SCORE / 100</div>
                <div style="margin-top:12px;padding:4px 10px;border-radius:20px;display:inline-block;
                    background:rgba(0,0,0,0.3);border:1px solid {rc};
                    font-size:0.6rem;color:{rc};letter-spacing:0.08em;">{risk_badge(row['risk_score'])}</div>
                <div style="margin-top:8px;font-size:0.7rem;color:{TEXT_DIM};">
                    {row['flags']} flags · ${row['total_risk']:,.0f}
                </div>
            </div>""", unsafe_allow_html=True)

        sec("Full Rankings", "◑")
        top20 = emp_stats.head(20)
        bar_colors = [risk_color(s) for s in top20["risk_score"]]
        fig = go.Figure(go.Bar(
            x=top20["risk_score"], y=top20["employee_id"], orientation="h",
            marker_color=bar_colors, marker_line_width=0,
            text=top20["risk_score"].apply(lambda x: f"{x:.0f}"),
            textposition="outside", textfont=dict(color=TEXT_MID, size=10),
            customdata=top20[["dept","flags","total_risk"]].values,
            hovertemplate="<b>%{y}</b><br>Dept: %{customdata[0]}<br>Flags: %{customdata[1]}<br>At-Risk: $%{customdata[2]:,.0f}<extra></extra>"
        ))
        fig.update_layout(**PT["layout"], height=540, xaxis_range=[0,115],
                          xaxis_title="Risk Score (0-100)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            emp_stats[["employee_id","dept","flags","total_risk","avg_conf","anomaly_types","risk_score"]]
            .rename(columns={"employee_id":"Employee","dept":"Dept","flags":"Flags",
                             "total_risk":"At-Risk ($)","avg_conf":"Avg Conf",
                             "anomaly_types":"Types","risk_score":"Score"})
            .head(30), use_container_width=True, height=380)

        sec("Vendor Risk Rankings", "⊹")
        vendor_stats = (flagged.groupby("vendor")
            .agg(flags=("transaction_id","count"), total_risk=("amount","sum"),
                 depts=("department","nunique"), avg_conf=("confidence","mean"),
                 types=("anomaly_type", lambda x: " · ".join(x.unique())))
            .reset_index())
        max_v = vendor_stats["total_risk"].max() or 1
        vendor_stats["score"] = (
            vendor_stats["flags"] / vendor_stats["flags"].max() * 40 +
            vendor_stats["total_risk"] / max_v * 40 +
            vendor_stats["avg_conf"] * 20
        ).round(1)
        vendor_stats = vendor_stats.sort_values("score", ascending=False).reset_index(drop=True)

        col1, col2 = st.columns(2)
        with col1:
            top_v = vendor_stats.head(10)
            fig_v = go.Figure(go.Bar(
                x=top_v["score"], y=top_v["vendor"], orientation="h",
                marker_color=[risk_color(s) for s in top_v["score"]],
                marker_line_width=0,
                text=top_v["score"].apply(lambda x: f"{x:.0f}"),
                textposition="outside", textfont=dict(color=TEXT_MID, size=10)
            ))
            fig_v.update_layout(**PT["layout"], height=320, xaxis_range=[0,115])
            plotly_fig(fig_v, 320, "Top 10 Highest-Risk Vendors")
        with col2:
            risk_dist = vendor_stats["score"].apply(risk_badge).value_counts().reset_index()
            risk_dist.columns = ["level","count"]
            color_map = {"CRITICAL":RED,"HIGH":ACCENT2,"MEDIUM":ACCENT,"LOW":GREEN}
            fig_d = px.pie(risk_dist, names="level", values="count",
                           color="level", color_discrete_map=color_map, hole=0.6)
            plotly_fig(fig_d, 320, "Vendor Risk Distribution")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: IMPACT CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Impact Calculator":
    if not st.session_state.pipeline_run:
        st.warning("Initialize the system first.")
    else:
        df_pred = st.session_state.df_pred
        flagged = df_pred[df_pred["prediction"] == 1].copy()
        bm = st.session_state.best_model or "Model"
        mb = st.session_state.metrics_before or {}
        precision = mb.get(bm, {}).get("precision", 0.85)
        recall    = mb.get(bm, {}).get("recall", 0.82)
        f1        = mb.get(bm, {}).get("f1", 0.83)
        total_txns   = len(df_pred)
        total_spend  = df_pred["amount"].sum()
        flagged_amt  = flagged["amount"].sum()
        n_flagged    = len(flagged)

        sec("Scenario Parameters", "⊹")
        col1, col2, col3 = st.columns(3)
        with col1:
            recovery_rate = st.slider("Fraud Recovery Rate (%)", 10, 100, 75, 5) / 100
        with col2:
            audit_cost_per_txn = st.slider("Cost per Review ($)", 5, 200, 25, 5)
        with col3:
            industry_fraud_rate = st.slider("Industry Fraud Rate (%)", 1, 15, 5, 1) / 100

        fraud_prevented  = flagged_amt * precision * recovery_rate
        audit_cost       = n_flagged * audit_cost_per_txn
        net_savings      = fraud_prevented - audit_cost
        baseline_loss    = total_spend * industry_fraud_rate
        system_loss      = flagged_amt * (1-precision) + (total_spend - flagged_amt) * (1-recall) * 0.05
        loss_reduction   = baseline_loss - system_loss
        roi              = (net_savings / audit_cost * 100) if audit_cost > 0 else 0
        true_positives   = n_flagged * precision
        false_positives  = n_flagged * (1 - precision)

        sec("Financial Impact", "◈")
        st.markdown(f"""<div class="kpi-row">
            {kpi(f"${fraud_prevented:,.0f}", "Fraud Prevented", f"Recovery {recovery_rate:.0%}", "ok", GREEN)}
            {kpi(f"${audit_cost:,.0f}", "Audit Cost", f"{n_flagged} reviews", "warn", ACCENT)}
            {kpi(f"${net_savings:,.0f}", "Net Saving", "Prevented − Cost", "ok" if net_savings>0 else "up", GREEN if net_savings>0 else RED)}
            {kpi(f"{roi:.0f}%", "ROI", "on audit investment", "ok", PURPLE)}
        </div>""", unsafe_allow_html=True)

        sec("With vs Without ExpenseGuard", "◑")
        fig = go.Figure()
        scenarios = ["Without System", "With ExpenseGuard AI"]
        fraud_loss = [baseline_loss, system_loss]
        a_cost = [total_txns * audit_cost_per_txn * 0.1, audit_cost]
        fig.add_trace(go.Bar(name="Fraud Loss", x=scenarios, y=fraud_loss,
                             marker_color=RED, marker_line_width=0, opacity=0.85))
        fig.add_trace(go.Bar(name="Audit Cost", x=scenarios, y=a_cost,
                             marker_color=ACCENT, marker_line_width=0, opacity=0.85))
        fig.update_layout(**PT["layout"], barmode="stack", height=320,
                          yaxis_title="Total Cost ($)")
        st.plotly_chart(fig, use_container_width=True)

        sec("Detection Gauges", "◎")
        col1, col2, col3 = st.columns(3)
        def gauge(val, title, color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val * 100,
                number={"suffix":"%","font":{"color":color,"size":28,"family":"Space Mono"}},
                title={"text":title,"font":{"color":TEXT_MID,"size":11}},
                gauge={"axis":{"range":[0,100],"tickcolor":BORDER,
                               "tickfont":{"color":TEXT_DIM,"size":8}},
                       "bar":{"color":color},
                       "bgcolor":BG,"bordercolor":BORDER,
                       "steps":[{"range":[0,50],"color":"#1a0808"},
                                {"range":[50,80],"color":"#1a1208"},
                                {"range":[80,100],"color":"#081a0d"}],
                       "threshold":{"line":{"color":"white","width":1.5},
                                    "thickness":0.8,"value":80}}))
            fig.update_layout(paper_bgcolor=SURFACE,plot_bgcolor=SURFACE,
                              font=dict(color=TEXT_MID,family="Syne"),
                              height=200,margin=dict(l=20,r=20,t=40,b=10))
            return fig
        with col1: st.plotly_chart(gauge(precision,"Precision",BLUE), use_container_width=True)
        with col2: st.plotly_chart(gauge(recall,"Recall",GREEN), use_container_width=True)
        with col3: st.plotly_chart(gauge(f1,"F1 Score",ACCENT), use_container_width=True)

        sec("12-Month Projection", "◷")
        months = list(range(1,13))
        cum_save = [net_savings * m for m in months]
        cum_cost = [audit_cost * m for m in months]
        breakeven = next((m for m,s,c in zip(months,cum_save,cum_cost) if s>c), None)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=months, y=cum_save, name="Cumulative Savings",
                                  line=dict(color=GREEN,width=2.5), fill="tozeroy",
                                  fillcolor=f"rgba(34,197,94,0.06)"))
        fig2.add_trace(go.Scatter(x=months, y=cum_cost, name="Audit Cost",
                                  line=dict(color=RED,width=1.5,dash="dash")))
        if breakeven:
            fig2.add_vline(x=breakeven, line_dash="dot", line_color=ACCENT,
                           annotation_text=f"Break-even: Month {breakeven}",
                           annotation_font_color=ACCENT)
        fig2.update_layout(**PT["layout"], height=280,
                           xaxis_title="Month", yaxis_title="Cumulative ($)")
        st.plotly_chart(fig2, use_container_width=True)

        sec("Detailed Breakdown", "◈")
        bd = [
            ("Total Transactions", f"{total_txns:,}", ""),
            ("Total Spend", f"${total_spend:,.0f}", ""),
            ("Flagged", f"{n_flagged:,}", f"{n_flagged/total_txns*100:.1f}% of transactions"),
            ("True Positives", f"{true_positives:,.0f}", f"Precision = {precision:.1%}"),
            ("False Positives", f"{false_positives:,.0f}", "Wasted reviews"),
            ("Fraud Amount Flagged", f"${flagged_amt:,.0f}", ""),
            ("Fraud Prevented", f"${fraud_prevented:,.0f}", f"Recovery {recovery_rate:.0%}"),
            ("Audit Cost", f"${audit_cost:,.0f}", f"${audit_cost_per_txn}/review"),
            ("Net Saving", f"${net_savings:,.0f}", "Prevented − Cost"),
            ("Baseline Loss (no system)", f"${baseline_loss:,.0f}", f"{industry_fraud_rate:.0%} of spend"),
            ("Loss Reduction", f"${loss_reduction:,.0f}", f"{loss_reduction/baseline_loss*100:.1f}% better"),
            ("ROI", f"{roi:.0f}%", "Return on audit investment"),
        ]
        st.dataframe(pd.DataFrame(bd, columns=["Metric","Value","Notes"]),
                     use_container_width=True, height=380)
