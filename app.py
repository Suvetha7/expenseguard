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

st.set_page_config(page_title="ExpenseGuard", page_icon="⬡",
                   layout="wide", initial_sidebar_state="expanded")

# ── PALETTE (matching reference image) ────────────────────────────────────────
BG      = "#0f0f0f"
CARD    = "#1a1a1a"
CARD2   = "#222222"
BORDER  = "#2a2a2a"
LIME    = "#b8f542"   # main accent — lime green
ORANGE  = "#f57c2a"   # secondary accent — orange
WHITE   = "#ffffff"
GRAY    = "#888888"
DARKGRAY= "#333333"
TEXT    = "#f0f0f0"
TEXTDIM = "#555555"

ACOLORS = {
    "Normal": "#888888",
    "Duplicate": "#f57c2a",
    "Policy Violation": "#b8f542",
    "Ghost Vendor": "#ff4444",
    "Redundant Spending": "#4af0c4",
}

PT = dict(layout=dict(
    paper_bgcolor=CARD, plot_bgcolor=BG,
    font=dict(family="'DM Sans', sans-serif", color=GRAY, size=11),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False,
               tickfont=dict(color=DARKGRAY, size=10)),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False,
               tickfont=dict(color=DARKGRAY, size=10)),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                font=dict(color=GRAY)),
    margin=dict(l=12, r=12, t=32, b=12),
))

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif !important;
    box-sizing: border-box;
}}
.stApp, .main {{ background: {BG} !important; color: {TEXT}; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {{
    background: {BG} !important;
    border-right: 1px solid {BORDER} !important;
    width: 260px !important;
    padding: 0 !important;
}}
section[data-testid="stSidebar"] > div {{
    padding: 24px 16px !important;
}}
section[data-testid="stSidebar"] * {{ color: {GRAY} !important; }}
section[data-testid="stSidebar"] .stRadio > div {{
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 10px 14px !important;
    border-radius: 12px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    color: {GRAY} !important;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: {CARD} !important;
    color: {WHITE} !important;
}}
section[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
section[data-testid="stSidebar"] .stRadio label[data-baseweb] {{
    background: {LIME}18 !important;
    color: {LIME} !important;
}}

/* ── TOP NAV BAR ── */
.topnav {{
    background: {BG};
    border-bottom: 1px solid {BORDER};
    padding: 0 32px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}}
.topnav-logo {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.1rem;
    font-weight: 700;
    color: {WHITE};
    display: flex;
    align-items: center;
    gap: 10px;
}}
.topnav-logo-icon {{
    width: 32px; height: 32px;
    background: {LIME};
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
    color: {BG};
    font-weight: 800;
}}
.topnav-pills {{
    display: flex;
    gap: 8px;
    align-items: center;
}}
.topnav-pill {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 50px;
    padding: 6px 14px;
    font-size: 0.72rem;
    font-weight: 500;
    color: {GRAY};
    display: flex;
    align-items: center;
    gap: 6px;
}}
.topnav-pill.active {{
    background: {LIME}20;
    border-color: {LIME}60;
    color: {LIME};
}}
.live-pulse {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {LIME};
    animation: pulse 2s infinite;
}}
@keyframes pulse {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50% {{ opacity:0.4; transform:scale(1.3); }}
}}

/* ── PAGE CONTENT ── */
.page-wrap {{
    padding: 28px 32px;
}}
.page-title {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2rem;
    font-weight: 700;
    color: {WHITE};
    letter-spacing: -0.5px;
    margin-bottom: 4px;
    text-transform: uppercase;
}}
.page-sub {{
    font-size: 0.78rem;
    color: {TEXTDIM};
    margin-bottom: 28px;
    letter-spacing: 0.02em;
}}

/* ── SECTION LABEL ── */
.sec {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {TEXTDIM};
    margin: 24px 0 12px 0;
}}

/* ── KPI CARDS (pill style like reference) ── */
.kpi-wrap {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    height: 100%;
}}
.kpi-wrap::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: var(--kc, {BORDER});
    border-radius: 0 0 20px 20px;
}}
.kpi-label {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {TEXTDIM};
    margin-bottom: 10px;
    font-weight: 600;
}}
.kpi-value {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2rem;
    font-weight: 700;
    color: {WHITE};
    line-height: 1;
    margin-bottom: 8px;
}}
.kpi-delta {{
    font-size: 0.72rem;
    color: {GRAY};
    display: flex;
    align-items: center;
    gap: 4px;
}}
.kpi-delta.up {{ color: {ORANGE}; }}
.kpi-delta.ok {{ color: {LIME}; }}
.kpi-delta.warn {{ color: {ORANGE}; }}
.kpi-icon {{
    position: absolute;
    top: 18px; right: 18px;
    font-size: 0.8rem;
    color: {TEXTDIM};
}}
.kpi-more {{
    position: absolute;
    top: 16px; right: 16px;
    color: {DARKGRAY};
    font-size: 1rem;
    cursor: pointer;
}}

/* ── STAT CARDS (sidebar) ── */
.sb-stat {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 10px 14px;
    margin: 4px 0;
}}
.sb-stat-label {{
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXTDIM};
}}
.sb-stat-value {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem;
    font-weight: 600;
    color: {WHITE};
    margin-top: 2px;
}}

/* ── INIT CARD ── */
.init-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 24px;
    padding: 48px;
    text-align: center;
    margin: 40px 0;
}}
.init-icon {{
    font-size: 3rem;
    margin-bottom: 16px;
}}
.init-title {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.4rem;
    font-weight: 700;
    color: {WHITE};
    margin-bottom: 8px;
}}
.init-sub {{
    font-size: 0.82rem;
    color: {TEXTDIM};
}}

/* ── FEATURE TILES ── */
.feat-tile {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 24px;
    height: 100%;
    transition: border-color 0.2s;
}}
.feat-tile:hover {{ border-color: {LIME}40; }}
.feat-icon {{
    width: 40px; height: 40px;
    border-radius: 10px;
    background: {DARKGRAY};
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    margin-bottom: 14px;
    color: {LIME};
    font-weight: 700;
}}
.feat-name {{
    font-size: 0.88rem;
    font-weight: 600;
    color: {WHITE};
    margin-bottom: 6px;
}}
.feat-desc {{
    font-size: 0.72rem;
    color: {TEXTDIM};
    line-height: 1.6;
}}

/* ── CHIP TAGS ── */
.chip {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
.chip-dup  {{ background: {ORANGE}20; color: {ORANGE}; border: 1px solid {ORANGE}40; }}
.chip-pol  {{ background: {LIME}15;   color: {LIME};   border: 1px solid {LIME}40; }}
.chip-gho  {{ background: #ff444420;  color: #ff4444;  border: 1px solid #ff444440; }}
.chip-red  {{ background: #4af0c420;  color: #4af0c4;  border: 1px solid #4af0c440; }}

/* ── LEDGER BLOCKS ── */
.ledger-block {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-left: 3px solid {LIME};
    border-radius: 12px;
    padding: 14px 18px;
    margin: 6px 0;
    font-size: 0.72rem;
    color: {GRAY};
    font-family: 'Space Grotesk', sans-serif !important;
}}

/* ── BUTTONS ── */
.stButton > button {{
    background: {LIME} !important;
    color: {BG} !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    width: 100% !important;
}}
.stButton > button:hover {{
    background: #d4f040 !important;
    transform: translateY(-1px) !important;
}}

/* ── DATAFRAME ── */
.stDataFrame {{ border: 1px solid {BORDER} !important; border-radius: 16px !important; }}

/* ── EXPANDER ── */
.streamlit-expanderHeader {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    font-size: 0.8rem !important;
    color: {GRAY} !important;
}}

/* ── METRIC ── */
[data-testid="metric-container"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 16px !important;
    padding: 16px !important;
}}
[data-testid="metric-container"] label {{
    font-size: 0.62rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: {TEXTDIM} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.6rem !important;
    color: {WHITE} !important;
}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: {LIME}; }}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {CARD} !important;
    border-radius: 12px !important;
    padding: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px !important;
    color: {GRAY} !important;
    font-size: 0.75rem !important;
}}
.stTabs [aria-selected="true"] {{
    background: {DARKGRAY} !important;
    color: {WHITE} !important;
}}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def kpi(col, value, label, delta="", dtype="ok", color=LIME):
    arrow = "&#9650;" if dtype != "warn" else "&#9679;"
    delta_part = f'<div class="kpi-delta {dtype}">{arrow} {delta}</div>' if delta else '<div></div>'
    col.markdown(
        '<div class="kpi-wrap" style="--kc:' + color + '">'
        '<div class="kpi-label">' + label + '</div>'
        '<div class="kpi-value">' + str(value) + '</div>'
        + delta_part +
        '</div>',
        unsafe_allow_html=True
    )

def sec(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)

def page_header(title, subtitle=""):
    st.markdown(f"""<div class="page-wrap" style="padding-bottom:0">
        <div class="page-title">{title}</div>
        <div class="page-sub">{subtitle}</div>
    </div>""", unsafe_allow_html=True)

def plotly_fig(fig, height=300, title=""):
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color=GRAY, size=12, family="DM Sans")))
    fig.update_layout(**PT["layout"], height=height)
    st.plotly_chart(fig, use_container_width=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key, val in [("df_raw",None),("df_pred",None),("metrics_before",None),
                  ("metrics_after",None),("best_model",None),
                  ("pipeline_run",False),("low_conf_reviewed",{})]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:28px;">
        <div style="width:34px;height:34px;background:{LIME};border-radius:10px;
            display:flex;align-items:center;justify-content:center;
            font-weight:800;font-size:0.85rem;color:{BG};">EG</div>
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:0.88rem;
                font-weight:700;color:{WHITE};">ExpenseGuard</div>
            <div style="font-size:0.6rem;color:{TEXTDIM};letter-spacing:0.08em;">AUDIT SYSTEM</div>
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("", [
        "Overview", "Anomaly Detection", "Self-Learning",
        "Model Performance", "XAI Explainability",
        "Immutable Ledger", "Analytics",
        "Risk Leaderboard", "Impact Calculator"
    ], label_visibility="collapsed")

    st.markdown("<div style='margin:16px 0 8px'>", unsafe_allow_html=True)
    if st.button("⚡  Initialize System"):
        clear_ledger()
        with st.spinner("Generating dataset..."):
            df = generate_full_dataset(n=1500)
            st.session_state.df_raw = df
        with st.spinner("Training models..."):
            models_dict, metrics_before, best = run_full_pipeline(df, FEATURE_COLS)
            st.session_state.metrics_before = metrics_before
            st.session_state.best_model = best
        with st.spinner("Running predictions..."):
            df_pred = predict_transactions(df, FEATURE_COLS)
            st.session_state.df_pred = df_pred
            st.session_state.pipeline_run = True
            flagged = df_pred[(df_pred["prediction"]==1)&(df_pred["confidence"]>=0.80)]
            count = 0
            for _, row in flagged.iterrows():
                add_to_ledger(row.to_dict(), row.get("anomaly_type","Anomaly"),
                              row["confidence"], row["model_used"])
                count += 1
        st.success(f"Done — {count} blocks secured")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.pipeline_run:
        dp = st.session_state.df_pred
        chain = verify_chain()
        bm = st.session_state.best_model or ""
        auc = st.session_state.metrics_before.get(bm,{}).get("roc_auc",0)
        nf = int((dp["prediction"]==1).sum())
        st.markdown("<div style='margin-top:16px'>", unsafe_allow_html=True)
        for lbl, val in [("Records", f"{len(dp):,}"), ("Flagged", f"{nf:,}"),
                          ("Ledger", f"{'✓' if chain['valid'] else '✗'} {chain['length']} blocks"),
                          ("Best AUC", f"{auc:.4f}"), ("Model", bm[:14] if bm else "—")]:
            st.markdown(f"""<div class="sb-stat">
                <div class="sb-stat-label">{lbl}</div>
                <div class="sb-stat-value">{val}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── TOP NAV ───────────────────────────────────────────────────────────────────
is_live = st.session_state.pipeline_run
chain_s = verify_chain()
st.markdown(f"""<div class="topnav">
    <div style="font-size:0.72rem;color:{TEXTDIM};letter-spacing:0.08em;text-transform:uppercase;">
        {page.upper()}
    </div>
    <div class="topnav-pills">
        <div class="topnav-pill">Date: FY 2023-24</div>
        <div class="topnav-pill">SHA-256 · SMOTE · SHAP</div>
        <div class="topnav-pill {'active' if is_live else ''}">
            <div class="live-pulse"></div>
            {'LIVE' if is_live else 'IDLE'}
        </div>
    </div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    page_header("CHECK BOX", "Real-time expense audit intelligence")

    if not st.session_state.pipeline_run:
        st.markdown(f"""<div class="init-card">
            <div class="init-icon">⬡</div>
            <div class="init-title">System Idle</div>
            <div class="init-sub">Click Initialize System in the sidebar to begin analysis</div>
        </div>""", unsafe_allow_html=True)
        cols = st.columns(4)
        for col, (icon, name, desc) in zip(cols, [
            ("RF", "Hybrid ML", "Random Forest · Gradient Boosting · Extra Trees · Isolation Forest · One-Class SVM"),
            ("⟳", "Self-Learning", "Confidence scoring · Human-in-the-loop · Automatic model retraining"),
            ("⬡", "SHA-256 Ledger", "Hash-chained immutable blocks · AES encryption · Tamper detection"),
            ("◎", "SHAP XAI", "Global feature importance · Waterfall explanations · Per-transaction reasoning"),
        ]):
            col.markdown(f"""<div class="feat-tile">
                <div class="feat-icon">{icon}</div>
                <div class="feat-name">{name}</div>
                <div class="feat-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
    else:
        df_pred = st.session_state.df_pred
        total   = len(df_pred)
        flagged_n = int((df_pred["prediction"]==1).sum())
        low_c   = int((df_pred["confidence_level"].astype(str)=="Low").sum())
        chain   = verify_chain()
        bm      = st.session_state.best_model
        auc     = st.session_state.metrics_before[bm]["roc_auc"]
        flagged_df = df_pred[df_pred["prediction"]==1]
        flag_amt   = flagged_df["amount"].sum()
        total_amt  = df_pred["amount"].sum()

        c = st.columns(6)
        kpi(c[0], f"{total:,}", "Total Records", "", "ok", LIME)
        kpi(c[1], f"{flagged_n:,}", "Anomalies", f"{flagged_n/total*100:.1f}% of total", "up", ORANGE)
        kpi(c[2], f"${flag_amt/1e6:.2f}M", "At-Risk", f"{flag_amt/total_amt*100:.1f}% of spend", "up", ORANGE)
        kpi(c[3], f"{low_c:,}", "Need Review", "Low confidence", "warn", ORANGE)
        kpi(c[4], f"{chain['length']}", "Ledger Blocks", "SHA-256 secured", "ok", LIME)
        kpi(c[5], f"{auc:.4f}", "Best AUC", bm[:14], "ok", LIME)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            sec("CUSTOMER")
            dist = flagged_df["anomaly_type"].value_counts().reset_index()
            dist.columns = ["type","count"]
            fig = px.pie(dist, names="type", values="count", color="type",
                         color_discrete_map=ACOLORS, hole=0.65)
            fig.update_traces(textposition="outside", textinfo="label+percent",
                              textfont_size=10, marker_line_width=0)
            plotly_fig(fig, 280)
        with col2:
            sec("PRODUCT")
            dept_flag = flagged_df.groupby("department").size().reset_index(name="count")
            dept_flag = dept_flag.sort_values("count", ascending=True)
            fig2 = go.Figure(go.Bar(
                x=dept_flag["count"], y=dept_flag["department"],
                orientation="h",
                marker_color=[LIME if i % 2 == 0 else ORANGE for i in range(len(dept_flag))],
                marker_line_width=0,
                text=dept_flag["count"], textposition="outside",
                textfont=dict(color=GRAY, size=10)
            ))
            plotly_fig(fig2, 280)

        sec("PROJECTS TIMELINE")
        daily = df_pred[df_pred["prediction"]==1].copy()
        daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M").astype(str)
        monthly = daily.groupby(["month","anomaly_type"]).size().reset_index(name="count")
        fig3 = px.bar(monthly, x="month", y="count", color="anomaly_type",
                      color_discrete_map=ACOLORS)
        fig3.update_traces(marker_line_width=0)
        plotly_fig(fig3, 240)

# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Detection":
    page_header("MONITORING", "Filter and explore flagged transactions")
    if not st.session_state.pipeline_run:
        st.warning("Initialize the system first.")
    else:
        df_pred = st.session_state.df_pred
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
                (df_pred["prediction"]==1))
        filtered = df_pred[mask]

        st.markdown(f"""<div style="font-family:'Space Grotesk',sans-serif;font-size:0.8rem;
            color:{LIME};margin:10px 0;font-weight:600;">
            {len(filtered):,} transactions matched &nbsp;·&nbsp; At-risk: ${filtered['amount'].sum():,.0f}
        </div>""", unsafe_allow_html=True)

        display_cols = ["transaction_id","date","employee_id","department",
                        "vendor","category","amount","anomaly_type","confidence","confidence_level"]
        available = [c for c in display_cols if c in filtered.columns]
        show_df = filtered[available].copy()
        show_df["amount"]     = show_df["amount"].apply(lambda x: f"${x:,.2f}")
        show_df["confidence"] = show_df["confidence"].apply(lambda x: f"{x:.1%}")
        st.dataframe(show_df.head(200), use_container_width=True, height=360)

        c1, c2 = st.columns(2)
        with c1:
            sec("AMOUNT VS CONFIDENCE")
            fig = px.scatter(filtered, x="confidence", y="amount", color="anomaly_type",
                             color_discrete_map=ACOLORS,
                             hover_data=["employee_id","vendor","department"], opacity=0.75)
            fig.add_vline(x=LOW_CONFIDENCE_THRESHOLD, line_dash="dash",
                          line_color=ORANGE, annotation_text="Review threshold",
                          annotation_font_color=ORANGE)
            fig.update_traces(marker_size=7, marker_line_width=0)
            plotly_fig(fig, 300)
        with c2:
            sec("TOP FLAGGED VENDORS")
            top_v = (filtered.groupby("vendor")
                     .agg(count=("transaction_id","count"), total=("amount","sum"))
                     .sort_values("count", ascending=False).head(10).reset_index())
            fig_v = go.Figure(go.Bar(
                x=top_v["count"], y=top_v["vendor"], orientation="h",
                marker_color=ORANGE, marker_line_width=0,
                text=top_v["count"], textposition="outside",
                textfont=dict(color=GRAY, size=10)
            ))
            plotly_fig(fig_v, 300)

# ══════════════════════════════════════════════════════════════════════════════
# SELF-LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Self-Learning":
    page_header("SELF-LEARNING", "Human-in-the-loop confidence review")
    if not st.session_state.pipeline_run:
        st.warning("Initialize the system first.")
    else:
        df_pred = st.session_state.df_pred
        low_conf = df_pred[df_pred["confidence_level"].astype(str)=="Low"].copy()

        c = st.columns(3)
        kpi(c[0], f"{len(low_conf):,}", "Low Confidence", "Need human review", "warn", ORANGE)
        kpi(c[1], f"{len(st.session_state.low_conf_reviewed)}", "Reviewed", "Confirmed labels", "ok", LIME)
        kpi(c[2], f"{LOW_CONFIDENCE_THRESHOLD:.0%}", "Threshold", "Below = review queue", "ok", LIME)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([2,1])
        with c1:
            sec("CONFIDENCE DISTRIBUTION")
            fig = px.histogram(df_pred, x="confidence", color="confidence_level",
                               color_discrete_map={"High":LIME,"Medium":ORANGE,"Low":"#ff4444"}, nbins=40)
            fig.add_vline(x=LOW_CONFIDENCE_THRESHOLD, line_dash="dash", line_color=WHITE, line_width=1)
            plotly_fig(fig, 240)
        with c2:
            sec("BREAKDOWN")
            breakdown = df_pred["confidence_level"].astype(str).value_counts().reset_index()
            breakdown.columns = ["level","count"]
            fig2 = px.pie(breakdown, names="level", values="count", hole=0.65,
                          color="level", color_discrete_map={"High":LIME,"Medium":ORANGE,"Low":"#ff4444"})
            fig2.update_traces(marker_line_width=0)
            plotly_fig(fig2, 240)

        sec("REVIEW LOW-CONFIDENCE CASES")
        review_sample = low_conf.sample(min(15, len(low_conf)), random_state=1)
        for idx, row in review_sample.iterrows():
            with st.expander(f"  {row['transaction_id']}  ·  {row.get('vendor','?')}  ·  ${row.get('amount',0):,.0f}  ·  {row.get('department','')}"):
                cols = st.columns([2,1,1])
                cols[0].markdown(f"**Predicted:** `{row.get('anomaly_type','?')}` &nbsp; **Confidence:** `{row['confidence']:.1%}`")
                label = cols[1].selectbox("Label", ["Anomaly (1)","Normal (0)"],
                                          key=f"lbl_{idx}", index=0 if row.get("prediction",1)==1 else 1)
                if cols[2].button("Confirm", key=f"conf_{idx}"):
                    st.session_state.low_conf_reviewed[idx] = 1 if "Anomaly" in label else 0

        reviewed = len(st.session_state.low_conf_reviewed)
        st.markdown(f'<div style="font-size:0.78rem;color:{LIME};margin:8px 0;">{reviewed} samples confirmed for retraining</div>', unsafe_allow_html=True)
        if st.button("⟳  Trigger Retraining") and reviewed > 0:
            reviewed_df = review_sample[review_sample.index.isin(st.session_state.low_conf_reviewed.keys())].copy()
            reviewed_df["label"] = reviewed_df.index.map(st.session_state.low_conf_reviewed)
            with st.spinner("Retraining..."):
                results_after, best_after = self_learning_retrain(reviewed_df, FEATURE_COLS)
            if results_after:
                st.session_state.metrics_after = results_after
                st.success(f"Retrained · Best: {best_after}")
                st.session_state.low_conf_reviewed = {}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    page_header("MODEL PERFORMANCE", "Metrics, radar and before/after comparison")
    if not st.session_state.pipeline_run or not st.session_state.metrics_before:
        st.warning("Initialize the system first.")
    else:
        metrics_before = st.session_state.metrics_before
        metrics_after  = st.session_state.metrics_after
        metric_names   = ["accuracy","precision","recall","f1","roc_auc"]

        sec("METRICS TABLE")
        rows = []
        for mname, m in metrics_before.items():
            row = {"Model": mname}
            for mn in metric_names:
                row[mn.upper()] = f"{m.get(mn,0):.4f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            sec("RADAR COMPARISON")
            fig = go.Figure()
            model_colors = [LIME, ORANGE, "#4af0c4", "#ff4444", "#a78bfa"]
            for i, (mname, m) in enumerate(metrics_before.items()):
                vals = [m.get(mn,0) for mn in metric_names]+[m.get(metric_names[0],0)]
                cats = [mn.upper() for mn in metric_names]+[metric_names[0].upper()]
                fig.add_trace(go.Scatterpolar(r=vals, theta=cats, name=mname,
                                              fill="toself", opacity=0.45,
                                              line_color=model_colors[i % len(model_colors)]))
            fig.update_layout(
                polar=dict(bgcolor=BG,
                           radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER,
                                          tickfont=dict(color=TEXTDIM, size=9)),
                           angularaxis=dict(gridcolor=BORDER, tickfont=dict(color=GRAY, size=10))),
                paper_bgcolor=CARD, font=dict(color=GRAY, family="DM Sans"),
                height=340, legend=dict(bgcolor=CARD, bordercolor=BORDER))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sec("F1 SCORE")
            models = list(metrics_before.keys())
            f1s = [metrics_before[m].get("f1",0) for m in models]
            fig2 = go.Figure(go.Bar(
                x=models, y=f1s,
                marker_color=[LIME if i%2==0 else ORANGE for i in range(len(models))],
                marker_line_width=0,
                text=[f"{v:.4f}" for v in f1s], textposition="outside",
                textfont=dict(color=GRAY, size=10)
            ))
            fig2.update_layout(**PT["layout"], height=340, yaxis_range=[0,1.1])
            st.plotly_chart(fig2, use_container_width=True)

        if metrics_after:
            sec("BEFORE VS AFTER RETRAINING")
            sup_models = [m for m in metrics_before if m not in ("Isolation Forest","One-Class SVM")]
            for metric in ["f1","precision","recall","roc_auc"]:
                bv = [metrics_before[m].get(metric,0) for m in sup_models]
                av = [metrics_after.get(m,{}).get(metric,0) for m in sup_models]
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(name="Before", x=sup_models, y=bv,
                                      marker_color=GRAY, marker_line_width=0, opacity=0.7))
                fig3.add_trace(go.Bar(name="After", x=sup_models, y=av,
                                      marker_color=LIME, marker_line_width=0))
                fig3.update_layout(**PT["layout"], barmode="group", height=200,
                                   yaxis_range=[0,1.1],
                                   title=dict(text=metric.upper(), font=dict(color=GRAY,size=11)))
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Complete Self-Learning retraining to see before/after comparison.")

# ══════════════════════════════════════════════════════════════════════════════
# XAI EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "XAI Explainability":
    page_header("XAI EXPLAINABILITY", "SHAP — SHapley Additive exPlanations")
    if not st.session_state.pipeline_run:
        st.warning("Initialize the system first.")
    else:
        df_raw = st.session_state.df_raw
        with st.spinner("Computing SHAP values..."):
            shap_vals, X_data = get_shap_values(df_raw, FEATURE_COLS)

        if shap_vals is not None:
            mean_shap = np.abs(shap_vals).mean(axis=0)
            shap_df = pd.DataFrame({"Feature":FEATURE_COLS,"Mean |SHAP|":mean_shap}
                                   ).sort_values("Mean |SHAP|",ascending=True).tail(15)

            sec("TOP 15 FEATURES BY MEAN |SHAP|")
            fig = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                         color="Mean |SHAP|",
                         color_continuous_scale=[[0,DARKGRAY],[0.5,ORANGE],[1,LIME]])
            fig.update_traces(marker_line_width=0)
            fig.update_layout(**PT["layout"], height=460, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            sec("FEATURE RANKING")
            rank_df = shap_df.sort_values("Mean |SHAP|",ascending=False).reset_index(drop=True)
            rank_df.index += 1
            rank_df["Mean |SHAP|"] = rank_df["Mean |SHAP|"].apply(lambda x: f"{x:.6f}")
            st.dataframe(rank_df, use_container_width=True)

            sec("SHAP DISTRIBUTION — TOP 6 FEATURES")
            top6_idx = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1][:6]
            cols = st.columns(3)
            for i, feat_idx in enumerate(top6_idx):
                sv = shap_vals[:, feat_idx]
                with cols[i%3]:
                    fh = px.histogram(x=sv, nbins=30, color_discrete_sequence=[LIME])
                    fh.add_vline(x=0, line_dash="dash", line_color=ORANGE)
                    fh.update_layout(**PT["layout"], height=180, showlegend=False,
                                    title=dict(text=FEATURE_COLS[feat_idx],
                                               font=dict(color=GRAY,size=10)))
                    st.plotly_chart(fh, use_container_width=True)

            sec("SINGLE TRANSACTION WATERFALL")
            df_pred = st.session_state.df_pred
            flagged_idx = df_pred[df_pred["prediction"]==1]["confidence"].idxmax()
            txn_pos = df_pred.index.get_loc(flagged_idx)
            sv_single = shap_vals[txn_pos]
            wf_df = pd.DataFrame({"Feature":FEATURE_COLS,"SHAP Value":sv_single}
                                  ).sort_values("SHAP Value",key=abs,ascending=True).tail(12)
            colors = [ORANGE if v>0 else LIME for v in wf_df["SHAP Value"]]
            fig_wf = go.Figure(go.Bar(x=wf_df["SHAP Value"], y=wf_df["Feature"],
                                      orientation="h", marker_color=colors, marker_line_width=0))
            fig_wf.add_vline(x=0, line_color=BORDER)
            txn = df_pred.loc[flagged_idx]
            plotly_fig(fig_wf, 360, f"Why was {txn['transaction_id']} flagged?")
            st.markdown(f'<div style="font-size:0.72rem;color:{TEXTDIM};">Amount: ${txn["amount"]:,.2f} · Dept: {txn["department"]} · Type: {txn["anomaly_type"]} · Confidence: {txn["confidence"]:.1%}</div>', unsafe_allow_html=True)
        else:
            st.error("Could not compute SHAP values. Run: pip install shap")

# ══════════════════════════════════════════════════════════════════════════════
# IMMUTABLE LEDGER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Immutable Ledger":
    page_header("IMMUTABLE LEDGER", "SHA-256 blockchain-inspired audit trail")
    chain_info = verify_chain()
    sc = LIME if chain_info["valid"] else "#ff4444"

    c = st.columns(4)
    kpi(c[0], "VALID" if chain_info["valid"] else "BROKEN", "Chain Status", "SHA-256 verified", "ok", sc)
    kpi(c[1], str(chain_info["length"]), "Total Blocks", "Immutable records", "ok", LIME)
    kpi(c[2], "SHA-256", "Hash Algorithm", "Bitcoin-grade", "ok", LIME)
    kpi(c[3], "AES/Fernet", "PII Encryption", "Employee & vendor data", "ok", ORANGE)

    ledger_df = get_ledger_df()
    if ledger_df.empty:
        st.info("No entries yet. Initialize the system to populate.")
    else:
        sec("LEDGER ENTRIES")
        st.dataframe(ledger_df, use_container_width=True, height=300)

        sec("HASH CHAIN")
        from ledger.immutable_ledger import load_ledger
        raw_chain = load_ledger()
        for block in raw_chain[:8]:
            st.markdown(f"""<div class="ledger-block">
                <span style="color:{TEXTDIM}">Block</span>
                <span style="color:{LIME};font-weight:700"> #{block['index']}</span>
                &nbsp;·&nbsp;<span style="color:{WHITE}">{block['transaction_id']}</span>
                &nbsp;·&nbsp;<span style="color:{ORANGE}">{block['anomaly_type']}</span>
                &nbsp;·&nbsp;<span style="color:{LIME}">${float(block['amount']):,.0f}</span>
                &nbsp;·&nbsp;<span style="color:{TEXTDIM}">{block['timestamp'][:19]}</span><br>
                <span style="color:{TEXTDIM}">hash&nbsp;&nbsp;</span><span style="color:{ORANGE}">{block['hash'][:48]}...</span><br>
                <span style="color:{TEXTDIM}">prev&nbsp;&nbsp;</span><span style="color:{DARKGRAY}">{block['prev_hash'][:48]}...</span>
            </div>""", unsafe_allow_html=True)

        sec("ADD MANUAL ENTRY")
        with st.form("manual_ledger"):
            c1,c2,c3 = st.columns(3)
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
                st.success(f"Block #{block['index']} sealed · {block['hash'][:32]}...")
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":
    page_header("ANALYTICS", "Spend intelligence and financial impact")
    if not st.session_state.pipeline_run:
        st.warning("Initialize the system first.")
    else:
        df_pred = st.session_state.df_pred
        flagged = df_pred[df_pred["prediction"]==1]
        total_amt   = df_pred["amount"].sum()
        flagged_amt = flagged["amount"].sum()

        c = st.columns(3)
        kpi(c[0], f"${flagged_amt/1e6:.2f}M", "Total At-Risk", f"{flagged_amt/total_amt*100:.1f}% of spend", "up", ORANGE)
        kpi(c[1], f"${flagged['amount'].mean():,.0f}", "Avg Flagged", "per transaction", "warn", ORANGE)
        kpi(c[2], f"${total_amt/1e6:.2f}M", "Total Spend", "all transactions", "ok", LIME)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sec("SPEND BY CATEGORY")
            cat = df_pred.groupby("category")["amount"].sum().reset_index().sort_values("amount")
            fig = go.Figure(go.Bar(x=cat["amount"], y=cat["category"], orientation="h",
                                   marker_color=LIME, marker_line_width=0))
            plotly_fig(fig, 300)
        with c2:
            sec("SPEND BY DEPARTMENT")
            dept = df_pred.groupby("department")["amount"].sum().reset_index().sort_values("amount")
            fig2 = go.Figure(go.Bar(x=dept["amount"], y=dept["department"], orientation="h",
                                    marker_color=ORANGE, marker_line_width=0))
            plotly_fig(fig2, 300)

        sec("ANOMALY HEATMAP: DEPARTMENT × CATEGORY")
        heat = flagged.groupby(["department","category"]).size().reset_index(name="count")
        heat_pivot = heat.pivot(index="department", columns="category", values="count").fillna(0)
        fig3 = px.imshow(heat_pivot,
                         color_continuous_scale=[[0,BG],[0.4,DARKGRAY],[0.7,ORANGE],[1,LIME]],
                         text_auto=True, aspect="auto")
        fig3.update_layout(**PT["layout"], height=320)
        st.plotly_chart(fig3, use_container_width=True)

        sec("TOP HIGH-RISK EMPLOYEES")
        emp_risk = (flagged.groupby("employee_id")
                    .agg(flags=("transaction_id","count"), total=("amount","sum"), dept=("department","first"))
                    .sort_values("flags",ascending=False).head(15).reset_index())
        emp_risk["total"] = emp_risk["total"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(emp_risk, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RISK LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Leaderboard":
    page_header("RISK LEADERBOARD", "Employee and vendor composite risk rankings")
    if not st.session_state.pipeline_run:
        st.warning("Initialize the system first.")
    else:
        df_pred = st.session_state.df_pred
        flagged = df_pred[df_pred["prediction"]==1].copy()

        emp_stats = (flagged.groupby("employee_id")
            .agg(flags=("transaction_id","count"), total_risk=("amount","sum"),
                 avg_conf=("confidence","mean"), dept=("department","first"),
                 types=("anomaly_type", lambda x: " · ".join(x.unique())))
            .reset_index())
        max_risk = emp_stats["total_risk"].max() or 1
        emp_stats["score"] = (
            emp_stats["flags"]/emp_stats["flags"].max()*40 +
            emp_stats["total_risk"]/max_risk*40 +
            emp_stats["avg_conf"]*20
        ).round(1)
        emp_stats = emp_stats.sort_values("score",ascending=False).reset_index(drop=True)

        def rcolor(s):
            return "#ff4444" if s>=70 else ORANGE if s>=50 else LIME if s>=30 else GRAY
        def rbadge(s):
            return "CRITICAL" if s>=70 else "HIGH" if s>=50 else "MEDIUM" if s>=30 else "LOW"

        sec("TOP RANKED EMPLOYEES")
        top3 = emp_stats.head(3)
        cols = st.columns(3)
        for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
            rc = rcolor(row["score"])
            col.markdown(f"""<div class="kpi-wrap" style="--kc:{rc};text-align:center;">
                <div style="font-size:0.6rem;letter-spacing:0.12em;color:{TEXTDIM};margin-bottom:12px;">
                    RANK 0{i+1}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
                    font-weight:700;color:{rc};margin-bottom:4px;">{row['employee_id']}</div>
                <div style="font-size:0.7rem;color:{TEXTDIM};margin-bottom:14px;">{row['dept']}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:2.2rem;
                    font-weight:800;color:{WHITE};">{row['score']:.0f}</div>
                <div style="font-size:0.6rem;color:{TEXTDIM};letter-spacing:0.1em;">RISK SCORE / 100</div>
                <div style="margin-top:12px;display:inline-block;padding:4px 12px;
                    border-radius:50px;border:1px solid {rc}40;background:{rc}15;
                    font-size:0.6rem;color:{rc};letter-spacing:0.08em;">{rbadge(row['score'])}</div>
                <div style="margin-top:8px;font-size:0.7rem;color:{TEXTDIM};">
                    {row['flags']} flags · ${row['total_risk']:,.0f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sec("FULL RANKINGS")
        top20 = emp_stats.head(20)
        fig = go.Figure(go.Bar(
            x=top20["score"], y=top20["employee_id"], orientation="h",
            marker_color=[rcolor(s) for s in top20["score"]], marker_line_width=0,
            text=top20["score"].apply(lambda x: f"{x:.0f}"),
            textposition="outside", textfont=dict(color=GRAY, size=10),
        ))
        fig.update_layout(**PT["layout"], height=520, xaxis_range=[0,115],
                          xaxis_title="Risk Score (0-100)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            emp_stats[["employee_id","dept","flags","total_risk","avg_conf","types","score"]]
            .rename(columns={"employee_id":"Employee","dept":"Dept","flags":"Flags",
                             "total_risk":"At-Risk","avg_conf":"Avg Conf",
                             "types":"Types","score":"Score"})
            .head(30), use_container_width=True)

        sec("VENDOR RISK RANKINGS")
        vendor_stats = (flagged.groupby("vendor")
            .agg(flags=("transaction_id","count"), total_risk=("amount","sum"),
                 depts=("department","nunique"), avg_conf=("confidence","mean"),
                 types=("anomaly_type", lambda x:" · ".join(x.unique())))
            .reset_index())
        mv = vendor_stats["total_risk"].max() or 1
        vendor_stats["score"] = (
            vendor_stats["flags"]/vendor_stats["flags"].max()*40 +
            vendor_stats["total_risk"]/mv*40 +
            vendor_stats["avg_conf"]*20
        ).round(1)
        vendor_stats = vendor_stats.sort_values("score",ascending=False).reset_index(drop=True)

        c1, c2 = st.columns(2)
        with c1:
            tv = vendor_stats.head(10)
            fig_v = go.Figure(go.Bar(
                x=tv["score"], y=tv["vendor"], orientation="h",
                marker_color=[rcolor(s) for s in tv["score"]], marker_line_width=0,
                text=tv["score"].apply(lambda x: f"{x:.0f}"),
                textposition="outside", textfont=dict(color=GRAY, size=10)
            ))
            fig_v.update_layout(**PT["layout"], height=300, xaxis_range=[0,115])
            plotly_fig(fig_v, 300, "Top 10 Vendors")
        with c2:
            rd = vendor_stats["score"].apply(rbadge).value_counts().reset_index()
            rd.columns = ["level","count"]
            cmap = {"CRITICAL":"#ff4444","HIGH":ORANGE,"MEDIUM":LIME,"LOW":GRAY}
            fig_d = px.pie(rd, names="level", values="count",
                           color="level", color_discrete_map=cmap, hole=0.65)
            fig_d.update_traces(marker_line_width=0)
            plotly_fig(fig_d, 300, "Vendor Risk Distribution")

# ══════════════════════════════════════════════════════════════════════════════
# IMPACT CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Impact Calculator":
    page_header("IMPACT CALCULATOR", "ROI and financial impact analysis")
    if not st.session_state.pipeline_run:
        st.warning("Initialize the system first.")
    else:
        df_pred = st.session_state.df_pred
        flagged = df_pred[df_pred["prediction"]==1].copy()
        bm = st.session_state.best_model or "Model"
        mb = st.session_state.metrics_before or {}
        precision = mb.get(bm,{}).get("precision",0.85)
        recall    = mb.get(bm,{}).get("recall",0.82)
        f1        = mb.get(bm,{}).get("f1",0.83)
        total_txns  = len(df_pred)
        total_spend = df_pred["amount"].sum()
        flagged_amt = flagged["amount"].sum()
        n_flagged   = len(flagged)

        sec("SCENARIO PARAMETERS")
        c1,c2,c3 = st.columns(3)
        with c1: recovery_rate = st.slider("Fraud Recovery Rate (%)",10,100,75,5)/100
        with c2: audit_cost_per_txn = st.slider("Cost per Review ($)",5,200,25,5)
        with c3: industry_fraud_rate = st.slider("Industry Fraud Rate (%)",1,15,5,1)/100

        fraud_prevented = flagged_amt*precision*recovery_rate
        audit_cost      = n_flagged*audit_cost_per_txn
        net_savings     = fraud_prevented - audit_cost
        baseline_loss   = total_spend*industry_fraud_rate
        system_loss     = flagged_amt*(1-precision)+(total_spend-flagged_amt)*(1-recall)*0.05
        loss_reduction  = baseline_loss - system_loss
        roi             = (net_savings/audit_cost*100) if audit_cost>0 else 0
        true_pos        = n_flagged*precision
        false_pos       = n_flagged*(1-precision)

        st.markdown("<br>", unsafe_allow_html=True)
        sec("FINANCIAL IMPACT")
        c = st.columns(4)
        kpi(c[0], f"${fraud_prevented:,.0f}", "Fraud Prevented", f"Recovery {recovery_rate:.0%}", "ok", LIME)
        kpi(c[1], f"${audit_cost:,.0f}", "Audit Cost", f"{n_flagged} reviews", "warn", ORANGE)
        kpi(c[2], f"${net_savings:,.0f}", "Net Saving", "Prevented - Cost", "ok" if net_savings>0 else "up", LIME if net_savings>0 else "#ff4444")
        kpi(c[3], f"{roi:.0f}%", "ROI", "on audit investment", "ok", LIME)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sec("WITH VS WITHOUT EXPENSEGUARD")
            fig = go.Figure()
            scenarios = ["Without System","With ExpenseGuard"]
            fig.add_trace(go.Bar(name="Fraud Loss", x=scenarios,
                                 y=[baseline_loss, system_loss],
                                 marker_color=ORANGE, marker_line_width=0))
            fig.add_trace(go.Bar(name="Audit Cost", x=scenarios,
                                 y=[total_txns*audit_cost_per_txn*0.1, audit_cost],
                                 marker_color=LIME, marker_line_width=0))
            fig.update_layout(**PT["layout"], barmode="stack", height=300,
                              yaxis_title="Total Cost ($)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sec("DETECTION GAUGES")
            def gauge(val, title, color):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=val*100,
                    number={"suffix":"%","font":{"color":color,"size":26,"family":"Space Grotesk"}},
                    title={"text":title,"font":{"color":GRAY,"size":11}},
                    gauge={"axis":{"range":[0,100],"tickcolor":BORDER,"tickfont":{"color":TEXTDIM,"size":8}},
                           "bar":{"color":color},"bgcolor":BG,"bordercolor":BORDER,
                           "steps":[{"range":[0,50],"color":"#1a0a0a"},
                                    {"range":[50,80],"color":"#1a1208"},
                                    {"range":[80,100],"color":"#0a1a0a"}],
                           "threshold":{"line":{"color":WHITE,"width":1.5},"thickness":0.8,"value":80}}))
                fig.update_layout(paper_bgcolor=CARD,plot_bgcolor=CARD,
                                  font=dict(color=GRAY,family="DM Sans"),
                                  height=180,margin=dict(l=16,r=16,t=36,b=8))
                return fig
            ga, gb, gc = st.columns(3)
            with ga: st.plotly_chart(gauge(precision,"Precision",LIME), use_container_width=True)
            with gb: st.plotly_chart(gauge(recall,"Recall",ORANGE), use_container_width=True)
            with gc: st.plotly_chart(gauge(f1,"F1 Score",LIME), use_container_width=True)

        sec("12-MONTH PROJECTION")
        months = list(range(1,13))
        cum_save = [net_savings*m for m in months]
        cum_cost = [audit_cost*m for m in months]
        be = next((m for m,s,c in zip(months,cum_save,cum_cost) if s>c), None)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=months, y=cum_save, name="Cumulative Savings",
                                  line=dict(color=LIME,width=2.5), fill="tozeroy",
                                  fillcolor=f"rgba(184,245,66,0.07)"))
        fig2.add_trace(go.Scatter(x=months, y=cum_cost, name="Audit Cost",
                                  line=dict(color=ORANGE,width=1.5,dash="dash")))
        if be:
            fig2.add_vline(x=be, line_dash="dot", line_color=WHITE, line_width=1,
                           annotation_text=f"Break-even M{be}",
                           annotation_font_color=WHITE, annotation_font_size=10)
        fig2.update_layout(**PT["layout"], height=260,
                           xaxis_title="Month", yaxis_title="Cumulative ($)")
        st.plotly_chart(fig2, use_container_width=True)

        sec("DETAILED BREAKDOWN")
        bd = [
            ("Total Transactions",f"{total_txns:,}",""),
            ("Total Spend",f"${total_spend:,.0f}",""),
            ("Flagged",f"{n_flagged:,}",f"{n_flagged/total_txns*100:.1f}% of transactions"),
            ("True Positives",f"{true_pos:,.0f}",f"Precision = {precision:.1%}"),
            ("False Positives",f"{false_pos:,.0f}","Wasted reviews"),
            ("Fraud Amount Flagged",f"${flagged_amt:,.0f}",""),
            ("Fraud Prevented",f"${fraud_prevented:,.0f}",f"Recovery {recovery_rate:.0%}"),
            ("Audit Cost",f"${audit_cost:,.0f}",f"${audit_cost_per_txn}/review"),
            ("Net Saving",f"${net_savings:,.0f}","Prevented - Cost"),
            ("Baseline Loss",f"${baseline_loss:,.0f}",f"{industry_fraud_rate:.0%} of spend"),
            ("Loss Reduction",f"${loss_reduction:,.0f}",f"{loss_reduction/baseline_loss*100:.1f}% better"),
            ("ROI",f"{roi:.0f}%","Return on audit investment"),
        ]
        st.dataframe(pd.DataFrame(bd, columns=["Metric","Value","Notes"]),
                     use_container_width=True, height=370)

st.markdown('</div>', unsafe_allow_html=True)
