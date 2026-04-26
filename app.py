import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.generate_data import generate_full_dataset
from utils.features import FEATURE_COLS
from models.ml_pipeline import (run_full_pipeline, predict_transactions,
                                  self_learning_retrain, LOW_CONFIDENCE_THRESHOLD,
                                  get_shap_values)
from ledger.immutable_ledger import add_to_ledger, verify_chain, get_ledger_df, clear_ledger

st.set_page_config(page_title="ExpenseGuard AI", page_icon="⬡",
                   layout="wide", initial_sidebar_state="collapsed")

# ── PALETTE ───────────────────────────────────────────────────────────────────
BG    = "#0a0a0a"
SURF  = "#111111"
CARD  = "#181818"
CARD2 = "#1f1f1f"
BDR   = "#2a2a2a"
LIME  = "#c8f135"
ORG   = "#f57c2a"
RED   = "#ff4545"
TEAL  = "#4af0c4"
WHITE = "#f5f5f5"
GRAY  = "#888888"
DIM   = "#444444"

AC = {"Normal":GRAY,"Duplicate":ORG,"Policy Violation":LIME,
      "Ghost Vendor":RED,"Redundant Spending":TEAL}

PT = dict(paper_bgcolor=CARD, plot_bgcolor=BG,
          font=dict(family="Inter, sans-serif", color=GRAY, size=11),
          xaxis=dict(gridcolor=BDR, linecolor=BDR, zeroline=False, tickfont=dict(color=DIM,size=10)),
          yaxis=dict(gridcolor=BDR, linecolor=BDR, zeroline=False, tickfont=dict(color=DIM,size=10)),
          legend=dict(bgcolor=CARD, bordercolor=BDR, borderwidth=1, font=dict(color=GRAY,size=10)),
          margin=dict(l=12,r=12,t=36,b=12))

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

*,html,body,[class*="css"]{{font-family:'Inter',sans-serif!important;box-sizing:border-box;}}
.stApp,.main{{background:{BG}!important;color:{WHITE};}}
.block-container{{padding:0!important;max-width:100%!important;}}

/* Hide default sidebar completely */
section[data-testid="stSidebar"]{{display:none!important;width:0!important;}}
[data-testid="collapsedControl"]{{display:none!important;}}
.css-1d391kg, .css-163ttbj, [data-testid="stSidebarNav"]{{display:none!important;}}

/* ── TOP NAV ── */
.topbar{{
    position:sticky;top:0;z-index:999;
    background:{SURF};
    border-bottom:1px solid {BDR};
    display:flex;align-items:center;justify-content:space-between;
    padding:0 32px;height:58px;
}}
.topbar-left{{display:flex;align-items:center;gap:32px;}}
.topbar-logo{{
    font-family:'Space Grotesk',sans-serif!important;
    font-size:1rem;font-weight:700;color:{WHITE};
    display:flex;align-items:center;gap:10px;
    white-space:nowrap;
}}
.logo-badge{{
    width:30px;height:30px;background:{LIME};border-radius:8px;
    display:flex;align-items:center;justify-content:center;
    font-size:0.75rem;font-weight:800;color:{BG};
}}
.topbar-nav{{display:flex;align-items:center;gap:4px;}}
.nav-btn{{
    padding:7px 16px;border-radius:8px;
    font-size:0.75rem;font-weight:500;color:{GRAY};
    cursor:pointer;transition:all 0.18s cubic-bezier(0.4,0,0.2,1);
    border:1px solid transparent;background:transparent;
    white-space:nowrap;letter-spacing:0.01em;
}}
.nav-btn:hover{{background:{CARD};color:{WHITE};border-color:{BDR};}}
.nav-btn.active{{background:{CARD2};color:{WHITE};border-color:{BDR};box-shadow:0 0 0 1px {BDR};}}
.topbar-right{{display:flex;align-items:center;gap:10px;}}
.status-pill{{
    display:flex;align-items:center;gap:6px;
    background:{CARD};border:1px solid {BDR};border-radius:50px;
    padding:5px 12px;font-size:0.7rem;color:{GRAY};
}}
.dot{{width:6px;height:6px;border-radius:50%;}}
.dot-lime{{background:{LIME};animation:blink 2s infinite;}}
.dot-gray{{background:{DIM};}}
@keyframes blink{{0%,100%{{opacity:1;}}50%{{opacity:0.3;}}}}
.train-btn{{
    background:{LIME};color:{BG};border:none;border-radius:8px;
    padding:7px 16px;font-size:0.78rem;font-weight:700;
    cursor:pointer;transition:all 0.15s;white-space:nowrap;
}}
.train-btn:hover{{background:#d4f040;}}

/* ── PAGE WRAP ── */
.pw{{padding:28px 32px;animation:pageIn 0.3s ease both;}}
@keyframes pageIn{{from{{opacity:0;transform:translateY(8px);}}to{{opacity:1;transform:translateY(0);}}}}
.kc{{animation:cardIn 0.35s ease both;}}
@keyframes cardIn{{from{{opacity:0;transform:translateY(12px);}}to{{opacity:1;transform:translateY(0);}}}}

/* ── PAGE TITLE ── */
.ptitle{{
    font-family:'Space Grotesk',sans-serif!important;
    font-size:1.8rem;font-weight:800;color:{WHITE};
    letter-spacing:-0.5px;text-transform:uppercase;
    margin-bottom:4px;
}}
.psub{{font-size:0.75rem;color:{DIM};margin-bottom:24px;letter-spacing:0.03em;}}

/* ── SECTION LABEL ── */
.sl{{
    font-size:0.62rem;font-weight:700;text-transform:uppercase;
    letter-spacing:0.14em;color:{DIM};
    margin:24px 0 12px;display:flex;align-items:center;gap:10px;
}}
.sl::after{{content:'';flex:1;height:1px;background:{BDR};}}

/* ── KPI CARD ── */
.kc{{
    background:{CARD};border:1px solid {BDR};border-radius:16px;
    padding:20px 22px;position:relative;overflow:hidden;height:100%;
    transition:border-color 0.2s;
}}
.kc:hover{{border-color:{DIM};}}
.kc-bar{{
    position:absolute;bottom:0;left:0;right:0;height:2px;
    background:var(--c,{BDR});border-radius:0 0 16px 16px;
}}
.kc-lbl{{font-size:0.62rem;text-transform:uppercase;letter-spacing:0.12em;color:{DIM};margin-bottom:10px;font-weight:600;}}
.kc-val{{font-family:'Space Grotesk',sans-serif!important;font-size:1.9rem;font-weight:700;color:{WHITE};line-height:1;margin-bottom:8px;}}
.kc-d{{font-size:0.7rem;color:{GRAY};}}
.kc-d.up{{color:{ORG};}} .kc-d.ok{{color:{LIME};}} .kc-d.warn{{color:{ORG};}}

/* ── IDLE STATE ── */
.idle{{
    background:{CARD};border:1px solid {BDR};border-radius:20px;
    padding:60px;text-align:center;margin:40px 0;
}}
.idle-icon{{font-size:2.5rem;margin-bottom:16px;}}
.idle-title{{font-family:'Space Grotesk',sans-serif!important;font-size:1.2rem;font-weight:700;color:{WHITE};margin-bottom:8px;}}
.idle-sub{{font-size:0.8rem;color:{DIM};line-height:1.6;}}

/* ── FEATURE TILES ── */
.ft{{background:{CARD};border:1px solid {BDR};border-radius:16px;padding:22px;height:100%;transition:border-color 0.2s;}}
.ft:hover{{border-color:{LIME}50;}}
.ft-icon{{width:36px;height:36px;border-radius:10px;background:{CARD2};display:flex;align-items:center;justify-content:center;font-size:0.85rem;font-weight:700;color:{LIME};margin-bottom:14px;}}
.ft-name{{font-size:0.85rem;font-weight:600;color:{WHITE};margin-bottom:6px;}}
.ft-desc{{font-size:0.72rem;color:{DIM};line-height:1.6;}}

/* ── LEDGER BLOCK ── */
.lb{{background:{BG};border:1px solid {BDR};border-left:2px solid {LIME};
    border-radius:10px;padding:12px 16px;margin:6px 0;
    font-family:'Space Grotesk',sans-serif!important;font-size:0.68rem;color:{GRAY};}}

/* ── RANK CARD ── */
.rc{{background:{CARD};border:1px solid {BDR};border-radius:16px;
    padding:22px;text-align:center;border-top:2px solid var(--rc,{BDR});}}

/* ── SIDEBAR STAT (now used inline) ── */
.ss{{background:{CARD};border:1px solid {BDR};border-radius:10px;padding:10px 14px;display:inline-block;}}
.ss-l{{font-size:0.58rem;text-transform:uppercase;letter-spacing:0.1em;color:{DIM};}}
.ss-v{{font-family:'Space Grotesk',sans-serif!important;font-size:0.88rem;font-weight:600;color:{WHITE};margin-top:2px;}}

/* ── BUTTONS ── */
.stButton>button{{
    background:transparent!important;color:{GRAY}!important;
    border:1px solid {BDR}!important;border-radius:8px!important;
    font-size:0.72rem!important;font-weight:500!important;
    padding:7px 12px!important;width:100%!important;
    font-family:'Inter',sans-serif!important;
    transition:all 0.18s cubic-bezier(0.4,0,0.2,1)!important;
    white-space:nowrap!important;overflow:hidden!important;
    text-overflow:ellipsis!important;
}}
.stButton>button:hover{{
    background:{CARD2}!important;color:{WHITE}!important;
    border-color:{DIM}!important;
    transform:translateY(-1px)!important;
}}
/* Active page button — first button is always Initialize */
div[data-testid="column"]:last-child .stButton>button{{
    background:{LIME}!important;color:{BG}!important;
    border-color:{LIME}!important;font-weight:700!important;
}}
div[data-testid="column"]:last-child .stButton>button:hover{{
    background:#d4f040!important;
}}

/* ── DATAFRAME ── */
.stDataFrame{{border:1px solid {BDR}!important;border-radius:12px!important;overflow:hidden!important;}}
iframe{{border-radius:12px!important;}}

/* ── EXPANDER ── */
.streamlit-expanderHeader{{background:{CARD}!important;border:1px solid {BDR}!important;border-radius:10px!important;font-size:0.78rem!important;color:{GRAY}!important;}}

/* ── METRICS ── */
[data-testid="metric-container"]{{background:{CARD}!important;border:1px solid {BDR}!important;border-radius:12px!important;padding:16px!important;}}
[data-testid="metric-container"] label{{font-size:0.6rem!important;text-transform:uppercase!important;letter-spacing:0.1em!important;color:{DIM}!important;}}
[data-testid="stMetricValue"]{{font-family:'Space Grotesk',sans-serif!important;font-size:1.5rem!important;color:{WHITE}!important;}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{{background:{CARD}!important;border-radius:10px!important;padding:4px!important;}}
.stTabs [data-baseweb="tab"]{{border-radius:8px!important;color:{GRAY}!important;font-size:0.75rem!important;}}
.stTabs [aria-selected="true"]{{background:{CARD2}!important;color:{WHITE}!important;}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{{width:4px;height:4px;}}
::-webkit-scrollbar-track{{background:{BG};}}
::-webkit-scrollbar-thumb{{background:{BDR};border-radius:2px;}}
::-webkit-scrollbar-thumb:hover{{background:{LIME};}}

/* ── SELECTS/INPUTS ── */
[data-baseweb="select"]>div{{background:{CARD}!important;border-color:{BDR}!important;border-radius:10px!important;}}
[data-baseweb="input"]>div{{background:{CARD}!important;border-color:{BDR}!important;border-radius:10px!important;}}
textarea{{background:{CARD}!important;border-color:{BDR}!important;color:{WHITE}!important;border-radius:10px!important;}}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k,v in [("df_raw",None),("df_pred",None),("metrics_before",None),
             ("metrics_after",None),("best_model",None),
             ("pipeline_run",False),("page","Overview"),("low_conf_reviewed",{})]:
    if k not in st.session_state: st.session_state[k] = v

# ── HELPERS ───────────────────────────────────────────────────────────────────
def kpi(col, val, lbl, delta="", dtype="ok", color=LIME):
    col.markdown(
        f'<div class="kc"><div class="kc-bar" style="--c:{color}"></div>'
        f'<div class="kc-lbl">{lbl}</div>'
        f'<div class="kc-val">{val}</div>'
        + (f'<div class="kc-d {dtype}">{"▲" if dtype=="up" else "●" if dtype=="warn" else "▲"} {delta}</div>' if delta else '')
        + '</div>', unsafe_allow_html=True)

def sl(label):
    st.markdown(f'<div class="sl">{label}</div>', unsafe_allow_html=True)

def ph(title, sub=""):
    st.markdown(f'<div class="ptitle">{title}</div><div class="psub">{sub}</div>', unsafe_allow_html=True)

def pf(fig, h=300, title=""):
    if title: fig.update_layout(title=dict(text=title,font=dict(color=GRAY,size=12,family="Inter")))
    fig.update_layout(**PT, height=h)
    st.plotly_chart(fig, use_container_width=True)

# ── TOP NAV BAR ───────────────────────────────────────────────────────────────
PAGES = ["Overview","Anomaly Detection","Self-Learning","Model Performance",
         "XAI Explainability","Immutable Ledger","Analytics","Risk Leaderboard","Impact Calculator"]

is_live = st.session_state.pipeline_run
chain   = verify_chain()

# Build nav HTML
nav_items = ""
for p in PAGES:
    active = "active" if st.session_state.page == p else ""
    nav_items += f'<button class="nav-btn {active}" onclick="void(0)">{p}</button>'

st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">
      <div class="logo-badge">EG</div>
      ExpenseGuard
    </div>
  </div>
  <div class="topbar-right">
    <div class="status-pill">
      <div class="dot {'dot-lime' if is_live else 'dot-gray'}"></div>
      {'LIVE · ' + str(chain['length']) + ' blocks' if is_live else 'IDLE'}
    </div>
    <div class="status-pill">SHA-256 · SMOTE · SHAP</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Actual navigation using Streamlit (below the visual nav)
cols_nav = st.columns(len(PAGES) + 1)
for i, p in enumerate(PAGES):
    if cols_nav[i].button(p, key=f"nav_{p}",
                           type="primary" if st.session_state.page == p else "secondary"):
        st.session_state.page = p
        st.rerun()

# Initialize button in last column
with cols_nav[-1]:
    if st.button("⚡ Initialize" if not is_live else "↺ Retrain"):
        clear_ledger()
        with st.spinner("Generating dataset..."):
            df = generate_full_dataset(n=1500)
            st.session_state.df_raw = df
        with st.spinner("Training models..."):
            _, mb, best = run_full_pipeline(df, FEATURE_COLS)
            st.session_state.metrics_before = mb
            st.session_state.best_model = best
        with st.spinner("Predicting..."):
            dp = predict_transactions(df, FEATURE_COLS)
            st.session_state.df_pred = dp
            st.session_state.pipeline_run = True
            flagged = dp[(dp["prediction"]==1)&(dp["confidence"]>=0.80)]
            for _, row in flagged.iterrows():
                add_to_ledger(row.to_dict(), row.get("anomaly_type","Anomaly"),
                              row["confidence"], row["model_used"])
        st.success("Done!")
        st.rerun()

page = st.session_state.page
st.markdown('<div class="pw">', unsafe_allow_html=True)

# ── STATUS BAR (when live) ────────────────────────────────────────────────────
if is_live:
    dp = st.session_state.df_pred
    bm = st.session_state.best_model or ""
    auc = st.session_state.metrics_before.get(bm,{}).get("roc_auc",0)
    nf = int((dp["prediction"]==1).sum())
    stats = [("Records",f"{len(dp):,}"),("Flagged",f"{nf:,}"),
             ("AUC",f"{auc:.4f}"),("Model",bm[:18]),
             ("Ledger",f"{'✓' if chain['valid'] else '✗'} {chain['length']} blocks")]
    sc = " &nbsp;·&nbsp; ".join([f'<span style="color:{DIM}">{l}</span> <span style="color:{WHITE};font-weight:600">{v}</span>' for l,v in stats])
    st.markdown(f'<div style="font-size:0.72rem;padding:10px 0 20px;border-bottom:1px solid {BDR};margin-bottom:24px;">{sc}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    ph("Overview", "Real-time expense audit intelligence · FY 2023-24")
    if not is_live:
        st.markdown(f"""<div class="idle">
            <div class="idle-icon">⬡</div>
            <div class="idle-title">System Idle</div>
            <div class="idle-sub">Click <strong style="color:{LIME}">⚡ Initialize</strong> in the nav bar above to generate data and train all models</div>
        </div>""", unsafe_allow_html=True)
        c = st.columns(4)
        for col,(ic,nm,ds) in zip(c,[
            ("RF","Hybrid ML","Random Forest · Gradient Boosting · Extra Trees · Isolation Forest · One-Class SVM"),
            ("⟳","Self-Learning","Confidence scoring · Human-in-the-loop · Automatic model retraining on edge cases"),
            ("⬡","SHA-256 Ledger","Hash-chained immutable blocks · AES encryption · Tamper-evident audit trail"),
            ("◎","SHAP XAI","Global feature importance · Waterfall explanations · Per-transaction reasoning"),
        ]):
            col.markdown(f'<div class="ft"><div class="ft-icon">{ic}</div><div class="ft-name">{nm}</div><div class="ft-desc">{ds}</div></div>', unsafe_allow_html=True)
    else:
        total    = len(dp)
        fn       = int((dp["prediction"]==1).sum())
        lc       = int((dp["confidence_level"].astype(str)=="Low").sum())
        fdf      = dp[dp["prediction"]==1]
        fa       = fdf["amount"].sum()
        ta       = dp["amount"].sum()
        auc_val  = st.session_state.metrics_before.get(bm,{}).get("roc_auc",0)

        c = st.columns(6)
        kpi(c[0],f"{total:,}","Total Records","","ok",LIME)
        kpi(c[1],f"{fn:,}","Anomalies",f"{fn/total*100:.1f}% of total","up",ORG)
        kpi(c[2],f"${fa/1e6:.2f}M","At-Risk",f"{fa/ta*100:.1f}% of spend","up",ORG)
        kpi(c[3],f"{lc:,}","Need Review","Low confidence","warn",ORG)
        kpi(c[4],f"{chain['length']}","Ledger Blocks","SHA-256 secured","ok",LIME)
        kpi(c[5],f"{auc_val:.4f}","Best AUC",bm[:14],"ok",LIME)

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            sl("ANOMALY TYPE SPLIT")
            dist = fdf["anomaly_type"].value_counts().reset_index()
            dist.columns = ["type","count"]
            fig = px.pie(dist,names="type",values="count",color="type",color_discrete_map=AC,hole=0.65)
            fig.update_traces(textposition="outside",textinfo="label+percent",textfont_size=10,marker_line_width=0)
            pf(fig,280)
        with c2:
            sl("FLAGS BY DEPARTMENT")
            df2 = fdf.groupby("department").size().reset_index(name="n").sort_values("n")
            fig2 = go.Figure(go.Bar(x=df2["n"],y=df2["department"],orientation="h",
                marker_color=[LIME if i%2==0 else ORG for i in range(len(df2))],
                marker_line_width=0,text=df2["n"],textposition="outside",
                textfont=dict(color=GRAY,size=10)))
            pf(fig2,280)

        sl("MONTHLY ANOMALY TIMELINE")
        tmp = fdf.copy()
        tmp["mo"] = pd.to_datetime(tmp["date"]).dt.to_period("M").astype(str)
        mo = tmp.groupby(["mo","anomaly_type"]).size().reset_index(name="count")
        fig3 = px.bar(mo,x="mo",y="count",color="anomaly_type",color_discrete_map=AC)
        fig3.update_traces(marker_line_width=0)
        pf(fig3,240)

# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Detection":
    ph("Anomaly Detection","Filter and explore all flagged transactions")
    if not is_live: st.warning("Initialize the system first.")
    else:
        c1,c2,c3 = st.columns(3)
        with c1: at = st.multiselect("Type",dp["anomaly_type"].unique().tolist(),
            default=["Duplicate","Ghost Vendor","Policy Violation","Redundant Spending"])
        with c2: cl = st.multiselect("Confidence",["High","Medium","Low"],default=["High","Medium"])
        with c3: ds = st.multiselect("Department",sorted(dp["department"].unique()),
            default=sorted(dp["department"].unique()))

        mask = ((dp["anomaly_type"].isin(at))&(dp["confidence_level"].astype(str).isin(cl))
                &(dp["department"].isin(ds))&(dp["prediction"]==1))
        f = dp[mask]
        st.markdown(f'<div style="font-size:0.78rem;color:{LIME};font-weight:600;margin:8px 0;">'
                    f'{len(f):,} matched &nbsp;·&nbsp; At-risk: ${f["amount"].sum():,.0f}</div>',
                    unsafe_allow_html=True)

        cols = ["transaction_id","date","employee_id","department","vendor",
                "category","amount","anomaly_type","confidence","confidence_level"]
        av = [c for c in cols if c in f.columns]
        sd = f[av].copy()
        sd["amount"] = sd["amount"].apply(lambda x: f"${x:,.2f}")
        sd["confidence"] = sd["confidence"].apply(lambda x: f"{x:.1%}")
        st.dataframe(sd.head(200),use_container_width=True,height=340)

        c1,c2 = st.columns(2)
        with c1:
            sl("AMOUNT VS CONFIDENCE")
            fig = px.scatter(f,x="confidence",y="amount",color="anomaly_type",
                color_discrete_map=AC,hover_data=["employee_id","vendor","department"],opacity=0.75)
            fig.add_vline(x=LOW_CONFIDENCE_THRESHOLD,line_dash="dash",line_color=ORG,
                annotation_text="Review threshold",annotation_font_color=ORG)
            fig.update_traces(marker_size=7,marker_line_width=0)
            pf(fig,290)
        with c2:
            sl("TOP FLAGGED VENDORS")
            tv = (f.groupby("vendor").agg(count=("transaction_id","count"),total=("amount","sum"))
                  .sort_values("count",ascending=False).head(10).reset_index())
            fig2 = go.Figure(go.Bar(x=tv["count"],y=tv["vendor"],orientation="h",
                marker_color=ORG,marker_line_width=0,text=tv["count"],
                textposition="outside",textfont=dict(color=GRAY,size=10)))
            pf(fig2,290)

# ══════════════════════════════════════════════════════════════════════════════
# SELF-LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Self-Learning":
    ph("Self-Learning","Human-in-the-loop confidence review and retraining")
    if not is_live: st.warning("Initialize the system first.")
    else:
        lc = dp[dp["confidence_level"].astype(str)=="Low"].copy()
        c = st.columns(3)
        kpi(c[0],f"{len(lc):,}","Low Confidence","Need human review","warn",ORG)
        kpi(c[1],f"{len(st.session_state.low_conf_reviewed)}","Reviewed","Confirmed labels","ok",LIME)
        kpi(c[2],f"{LOW_CONFIDENCE_THRESHOLD:.0%}","Threshold","Below = review queue","ok",LIME)
        st.markdown("<br>",unsafe_allow_html=True)

        c1,c2 = st.columns([2,1])
        with c1:
            sl("CONFIDENCE DISTRIBUTION")
            fig = px.histogram(dp,x="confidence",color="confidence_level",
                color_discrete_map={"High":LIME,"Medium":ORG,"Low":RED},nbins=40)
            fig.add_vline(x=LOW_CONFIDENCE_THRESHOLD,line_dash="dash",line_color=WHITE,line_width=1)
            pf(fig,240)
        with c2:
            sl("BREAKDOWN")
            bd = dp["confidence_level"].astype(str).value_counts().reset_index()
            bd.columns = ["l","c"]
            fig2 = px.pie(bd,names="l",values="c",hole=0.65,color="l",
                color_discrete_map={"High":LIME,"Medium":ORG,"Low":RED})
            fig2.update_traces(marker_line_width=0)
            pf(fig2,240)

        sl("REVIEW LOW-CONFIDENCE CASES")
        rs = lc.sample(min(15,len(lc)),random_state=1)
        for idx,row in rs.iterrows():
            with st.expander(f"  {row['transaction_id']}  ·  {row.get('vendor','?')}  ·  ${row.get('amount',0):,.0f}  ·  {row.get('department','')}"):
                ca,cb,cc = st.columns([2,1,1])
                ca.markdown(f"**Predicted:** `{row.get('anomaly_type','?')}` · **Confidence:** `{row['confidence']:.1%}`")
                lbl = cb.selectbox("Label",["Anomaly (1)","Normal (0)"],key=f"l_{idx}",
                    index=0 if row.get("prediction",1)==1 else 1)
                if cc.button("Confirm",key=f"c_{idx}"):
                    st.session_state.low_conf_reviewed[idx] = 1 if "Anomaly" in lbl else 0

        rev = len(st.session_state.low_conf_reviewed)
        st.markdown(f'<div style="font-size:0.75rem;color:{LIME};margin:8px 0;">{rev} samples confirmed for retraining</div>',unsafe_allow_html=True)
        if st.button("⟳ Trigger Retraining") and rev>0:
            rd = rs[rs.index.isin(st.session_state.low_conf_reviewed.keys())].copy()
            rd["label"] = rd.index.map(st.session_state.low_conf_reviewed)
            with st.spinner("Retraining..."):
                ra,ba = self_learning_retrain(rd,FEATURE_COLS)
            if ra:
                st.session_state.metrics_after = ra
                st.success(f"Retrained · Best: {ba}")
                st.session_state.low_conf_reviewed = {}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    ph("Model Performance","Metrics, radar chart and retraining comparison")
    if not is_live or not st.session_state.metrics_before:
        st.warning("Initialize the system first.")
    else:
        mb = st.session_state.metrics_before
        ma = st.session_state.metrics_after
        mns = ["accuracy","precision","recall","f1","roc_auc"]

        sl("METRICS TABLE")
        rows = [{"Model":m,**{n.upper():f"{mb[m].get(n,0):.4f}" for n in mns}} for m in mb]
        st.dataframe(pd.DataFrame(rows),use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            sl("RADAR COMPARISON")
            fig = go.Figure()
            cols_r = [LIME,ORG,TEAL,RED,"#a78bfa"]
            for i,(mn,m) in enumerate(mb.items()):
                v = [m.get(n,0) for n in mns]+[m.get(mns[0],0)]
                ct = [n.upper() for n in mns]+[mns[0].upper()]
                fig.add_trace(go.Scatterpolar(r=v,theta=ct,name=mn,fill="toself",
                    opacity=0.4,line_color=cols_r[i%len(cols_r)]))
            fig.update_layout(
                polar=dict(bgcolor=BG,
                    radialaxis=dict(visible=True,range=[0,1],gridcolor=BDR,tickfont=dict(color=DIM,size=8)),
                    angularaxis=dict(gridcolor=BDR,tickfont=dict(color=GRAY,size=10))),
                paper_bgcolor=CARD,font=dict(color=GRAY,family="Inter"),
                height=320,legend=dict(bgcolor=CARD,bordercolor=BDR))
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            sl("F1 SCORE BY MODEL")
            mlist = list(mb.keys())
            f1s = [mb[m].get("f1",0) for m in mlist]
            fig2 = go.Figure(go.Bar(x=mlist,y=f1s,
                marker_color=[LIME if i%2==0 else ORG for i in range(len(mlist))],
                marker_line_width=0,text=[f"{v:.4f}" for v in f1s],
                textposition="outside",textfont=dict(color=GRAY,size=10)))
            fig2.update_layout(**PT,height=320,yaxis_range=[0,1.1])
            st.plotly_chart(fig2,use_container_width=True)

        if ma:
            sl("BEFORE VS AFTER RETRAINING")
            sm = [m for m in mb if m not in ("Isolation Forest","One-Class SVM")]
            for met in ["f1","precision","recall","roc_auc"]:
                bv = [mb[m].get(met,0) for m in sm]
                av = [ma.get(m,{}).get(met,0) for m in sm]
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(name="Before",x=sm,y=bv,marker_color=GRAY,marker_line_width=0,opacity=0.6))
                fig3.add_trace(go.Bar(name="After",x=sm,y=av,marker_color=LIME,marker_line_width=0))
                fig3.update_layout(**PT,barmode="group",height=200,yaxis_range=[0,1.1],
                    title=dict(text=met.upper(),font=dict(color=GRAY,size=11)))
                st.plotly_chart(fig3,use_container_width=True)
        else:
            st.info("Complete Self-Learning retraining to see before/after comparison.")

# ══════════════════════════════════════════════════════════════════════════════
# XAI EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "XAI Explainability":
    ph("XAI Explainability","SHAP — SHapley Additive exPlanations")
    if not is_live: st.warning("Initialize the system first.")
    else:
        with st.spinner("Computing SHAP values..."):
            sv,Xd = get_shap_values(st.session_state.df_raw,FEATURE_COLS)
        if sv is not None:
            ms = np.abs(sv).mean(axis=0)
            sdf = pd.DataFrame({"Feature":FEATURE_COLS,"Mean |SHAP|":ms}).sort_values("Mean |SHAP|",ascending=True).tail(15)
            sl("TOP 15 FEATURES BY MEAN |SHAP|")
            fig = px.bar(sdf,x="Mean |SHAP|",y="Feature",orientation="h",color="Mean |SHAP|",
                color_continuous_scale=[[0,DIM],[0.5,ORG],[1,LIME]])
            fig.update_traces(marker_line_width=0)
            fig.update_layout(**PT,height=440,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)

            sl("SHAP DISTRIBUTION — TOP 6 FEATURES")
            t6 = np.argsort(np.abs(sv).mean(axis=0))[::-1][:6]
            cols = st.columns(3)
            for i,fi in enumerate(t6):
                with cols[i%3]:
                    fh = px.histogram(x=sv[:,fi],nbins=30,color_discrete_sequence=[LIME])
                    fh.add_vline(x=0,line_dash="dash",line_color=ORG)
                    fh.update_layout(**PT,height=180,showlegend=False,
                        title=dict(text=FEATURE_COLS[fi],font=dict(color=GRAY,size=10)))
                    st.plotly_chart(fh,use_container_width=True)

            sl("SINGLE TRANSACTION WATERFALL")
            fi2 = dp[dp["prediction"]==1]["confidence"].idxmax()
            tp = dp.index.get_loc(fi2)
            ws = sv[tp]
            wd = pd.DataFrame({"Feature":FEATURE_COLS,"SHAP":ws}).sort_values("SHAP",key=abs,ascending=True).tail(12)
            wc = [ORG if v>0 else LIME for v in wd["SHAP"]]
            fw = go.Figure(go.Bar(x=wd["SHAP"],y=wd["Feature"],orientation="h",
                marker_color=wc,marker_line_width=0))
            fw.add_vline(x=0,line_color=BDR)
            tx = dp.loc[fi2]
            pf(fw,340,f"Why was {tx['transaction_id']} flagged?")
            st.markdown(f'<div style="font-size:0.7rem;color:{DIM};">Amount: ${tx["amount"]:,.2f} · {tx["department"]} · {tx["anomaly_type"]} · {tx["confidence"]:.1%}</div>',unsafe_allow_html=True)
        else:
            st.error("Run: pip install shap")

# ══════════════════════════════════════════════════════════════════════════════
# IMMUTABLE LEDGER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Immutable Ledger":
    ph("Immutable Ledger","SHA-256 blockchain-inspired tamper-evident audit trail")
    ci = verify_chain()
    sc = LIME if ci["valid"] else RED
    c = st.columns(4)
    kpi(c[0],"VALID" if ci["valid"] else "BROKEN","Chain Status","SHA-256 verified","ok",sc)
    kpi(c[1],str(ci["length"]),"Total Blocks","Immutable records","ok",LIME)
    kpi(c[2],"SHA-256","Hash Algorithm","Bitcoin-grade","ok",LIME)
    kpi(c[3],"AES/Fernet","PII Encryption","Employee & vendor data","ok",ORG)

    ldf = get_ledger_df()
    if ldf.empty: st.info("No entries yet. Initialize to populate.")
    else:
        sl("LEDGER ENTRIES")
        st.dataframe(ldf,use_container_width=True,height=280)
        sl("HASH CHAIN")
        from ledger.immutable_ledger import load_ledger
        for blk in load_ledger()[:8]:
            st.markdown(f"""<div class="lb">
                <span style="color:{DIM}">Block</span> <span style="color:{LIME};font-weight:700">#{blk['index']}</span>
                &nbsp;·&nbsp;<span style="color:{WHITE}">{blk['transaction_id']}</span>
                &nbsp;·&nbsp;<span style="color:{ORG}">{blk['anomaly_type']}</span>
                &nbsp;·&nbsp;<span style="color:{LIME}">${float(blk['amount']):,.0f}</span>
                &nbsp;·&nbsp;<span style="color:{DIM}">{blk['timestamp'][:19]}</span><br>
                <span style="color:{DIM}">hash&nbsp;</span><span style="color:{ORG}">{blk['hash'][:52]}...</span><br>
                <span style="color:{DIM}">prev&nbsp;</span><span style="color:{DIM}">{blk['prev_hash'][:52]}...</span>
            </div>""",unsafe_allow_html=True)
        sl("ADD MANUAL ENTRY")
        with st.form("ml"):
            a,b,c2 = st.columns(3)
            tid = a.text_input("Transaction ID","TXN_MANUAL_001")
            amt = b.number_input("Amount ($)",value=5000.0)
            at2 = c2.selectbox("Type",["Duplicate","Ghost Vendor","Policy Violation","Redundant Spending"])
            d2,e,f2 = st.columns(3)
            dept = d2.text_input("Department","Finance")
            emp  = e.text_input("Employee ID","EMP0042")
            conf = f2.slider("Confidence",0.0,1.0,0.9)
            if st.form_submit_button("⬡ Add to Ledger"):
                blk = add_to_ledger({"transaction_id":tid,"amount":amt,"department":dept,
                    "employee_id":emp,"vendor":"Manual","category":"Other","date":"2024-01-01"},at2,conf,"Manual")
                st.success(f"Block #{blk['index']} sealed · {blk['hash'][:32]}...")
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":
    ph("Analytics","Spend intelligence and financial impact")
    if not is_live: st.warning("Initialize the system first.")
    else:
        fdf = dp[dp["prediction"]==1]
        ta = dp["amount"].sum()
        fa = fdf["amount"].sum()
        c = st.columns(3)
        kpi(c[0],f"${fa/1e6:.2f}M","Total At-Risk",f"{fa/ta*100:.1f}% of spend","up",ORG)
        kpi(c[1],f"${fdf['amount'].mean():,.0f}","Avg Flagged","per transaction","warn",ORG)
        kpi(c[2],f"${ta/1e6:.2f}M","Total Spend","all transactions","ok",LIME)
        st.markdown("<br>",unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            sl("SPEND BY CATEGORY")
            cat = dp.groupby("category")["amount"].sum().reset_index().sort_values("amount")
            fig = go.Figure(go.Bar(x=cat["amount"],y=cat["category"],orientation="h",
                marker_color=LIME,marker_line_width=0))
            pf(fig,280)
        with c2:
            sl("SPEND BY DEPARTMENT")
            dept = dp.groupby("department")["amount"].sum().reset_index().sort_values("amount")
            fig2 = go.Figure(go.Bar(x=dept["amount"],y=dept["department"],orientation="h",
                marker_color=ORG,marker_line_width=0))
            pf(fig2,280)

        sl("ANOMALY HEATMAP: DEPARTMENT × CATEGORY")
        heat = fdf.groupby(["department","category"]).size().reset_index(name="n")
        hp = heat.pivot(index="department",columns="category",values="n").fillna(0)
        fig3 = px.imshow(hp,color_continuous_scale=[[0,BG],[0.4,DIM],[0.7,ORG],[1,LIME]],
            text_auto=True,aspect="auto")
        fig3.update_layout(**PT,height=300)
        st.plotly_chart(fig3,use_container_width=True)

        sl("TOP HIGH-RISK EMPLOYEES")
        er = (fdf.groupby("employee_id").agg(flags=("transaction_id","count"),
            total=("amount","sum"),dept=("department","first"))
            .sort_values("flags",ascending=False).head(15).reset_index())
        er["total"] = er["total"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(er,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RISK LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Leaderboard":
    ph("Risk Leaderboard","Employee and vendor composite risk rankings")
    if not is_live: st.warning("Initialize the system first.")
    else:
        fdf = dp[dp["prediction"]==1].copy()
        es = (fdf.groupby("employee_id").agg(flags=("transaction_id","count"),
            tr=("amount","sum"),ac=("confidence","mean"),dept=("department","first"),
            types=("anomaly_type",lambda x:" · ".join(x.unique()))).reset_index())
        mr = es["tr"].max() or 1
        es["score"] = (es["flags"]/es["flags"].max()*40+es["tr"]/mr*40+es["ac"]*20).round(1)
        es = es.sort_values("score",ascending=False).reset_index(drop=True)

        def rc(s): return RED if s>=70 else ORG if s>=50 else LIME if s>=30 else GRAY
        def rb(s): return "CRITICAL" if s>=70 else "HIGH" if s>=50 else "MEDIUM" if s>=30 else "LOW"

        sl("TOP RANKED EMPLOYEES")
        top3 = es.head(3)
        cols = st.columns(3)
        for i,(col,(_,row)) in enumerate(zip(cols,top3.iterrows())):
            rcc = rc(row["score"])
            col.markdown(f"""<div class="rc" style="--rc:{rcc}">
                <div style="font-size:0.58rem;letter-spacing:0.14em;color:{DIM};margin-bottom:14px;">RANK 0{i+1}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;color:{rcc};margin-bottom:4px;">{row['employee_id']}</div>
                <div style="font-size:0.7rem;color:{DIM};margin-bottom:16px;">{row['dept']}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:2.4rem;font-weight:800;color:{WHITE};">{row['score']:.0f}</div>
                <div style="font-size:0.58rem;color:{DIM};letter-spacing:0.1em;margin-bottom:12px;">RISK SCORE / 100</div>
                <div style="display:inline-block;padding:4px 12px;border-radius:50px;
                    border:1px solid {rcc}50;background:{rcc}15;font-size:0.6rem;color:{rcc};letter-spacing:0.08em;">{rb(row['score'])}</div>
                <div style="margin-top:10px;font-size:0.7rem;color:{DIM};">{row['flags']} flags · ${row['tr']:,.0f}</div>
            </div>""",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        sl("FULL RANKINGS — TOP 20")
        t20 = es.head(20)
        fig = go.Figure(go.Bar(x=t20["score"],y=t20["employee_id"],orientation="h",
            marker_color=[rc(s) for s in t20["score"]],marker_line_width=0,
            text=t20["score"].apply(lambda x: f"{x:.0f}"),textposition="outside",
            textfont=dict(color=GRAY,size=10),
            customdata=t20[["dept","flags","tr"]].values,
            hovertemplate="<b>%{y}</b><br>Dept: %{customdata[0]}<br>Flags: %{customdata[1]}<br>At-Risk: $%{customdata[2]:,.0f}<extra></extra>"))
        fig.update_layout(**PT,height=500,xaxis_range=[0,115],xaxis_title="Risk Score (0-100)")
        st.plotly_chart(fig,use_container_width=True)

        st.dataframe(es[["employee_id","dept","flags","tr","ac","types","score"]]
            .rename(columns={"employee_id":"Employee","dept":"Dept","flags":"Flags",
                "tr":"At-Risk","ac":"Avg Conf","types":"Types","score":"Score"})
            .head(30),use_container_width=True)

        sl("VENDOR RISK RANKINGS")
        vs = (fdf.groupby("vendor").agg(flags=("transaction_id","count"),tr=("amount","sum"),
            depts=("department","nunique"),ac=("confidence","mean"),
            types=("anomaly_type",lambda x:" · ".join(x.unique()))).reset_index())
        mv2 = vs["tr"].max() or 1
        vs["score"] = (vs["flags"]/vs["flags"].max()*40+vs["tr"]/mv2*40+vs["ac"]*20).round(1)
        vs = vs.sort_values("score",ascending=False).reset_index(drop=True)

        c1,c2 = st.columns(2)
        with c1:
            tv = vs.head(10)
            fv = go.Figure(go.Bar(x=tv["score"],y=tv["vendor"],orientation="h",
                marker_color=[rc(s) for s in tv["score"]],marker_line_width=0,
                text=tv["score"].apply(lambda x:f"{x:.0f}"),textposition="outside",
                textfont=dict(color=GRAY,size=10)))
            fv.update_layout(**PT,height=300,xaxis_range=[0,115])
            pf(fv,300,"Top 10 Vendors")
        with c2:
            rd = vs["score"].apply(rb).value_counts().reset_index()
            rd.columns = ["l","c"]
            cm = {"CRITICAL":RED,"HIGH":ORG,"MEDIUM":LIME,"LOW":GRAY}
            fd = px.pie(rd,names="l",values="c",color="l",color_discrete_map=cm,hole=0.65)
            fd.update_traces(marker_line_width=0)
            pf(fd,300,"Vendor Risk Distribution")

# ══════════════════════════════════════════════════════════════════════════════
# IMPACT CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Impact Calculator":
    ph("Impact Calculator","ROI analysis and financial impact modelling")
    if not is_live: st.warning("Initialize the system first.")
    else:
        fdf = dp[dp["prediction"]==1].copy()
        bm2 = st.session_state.best_model or "Model"
        mb2 = st.session_state.metrics_before or {}
        prec = mb2.get(bm2,{}).get("precision",0.85)
        rec  = mb2.get(bm2,{}).get("recall",0.82)
        f1v  = mb2.get(bm2,{}).get("f1",0.83)
        tn = len(dp); ts = dp["amount"].sum()
        fa2 = fdf["amount"].sum(); nf2 = len(fdf)

        sl("SCENARIO PARAMETERS")
        c1,c2,c3 = st.columns(3)
        with c1: rr = st.slider("Fraud Recovery Rate (%)",10,100,75,5)/100
        with c2: ac2 = st.slider("Cost per Review ($)",5,200,25,5)
        with c3: ifr = st.slider("Industry Fraud Rate (%)",1,15,5,1)/100

        fp = fa2*prec*rr; auc3 = nf2*ac2
        ns = fp-auc3; bl = ts*ifr
        sl2 = fa2*(1-prec)+(ts-fa2)*(1-rec)*0.05
        lr = bl-sl2; roi = (ns/auc3*100) if auc3>0 else 0

        st.markdown("<br>",unsafe_allow_html=True)
        sl("FINANCIAL IMPACT")
        c = st.columns(4)
        kpi(c[0],f"${fp:,.0f}","Fraud Prevented",f"Recovery {rr:.0%}","ok",LIME)
        kpi(c[1],f"${auc3:,.0f}","Audit Cost",f"{nf2} reviews","warn",ORG)
        kpi(c[2],f"${ns:,.0f}","Net Saving","Prevented − Cost","ok" if ns>0 else "up",LIME if ns>0 else RED)
        kpi(c[3],f"{roi:.0f}%","ROI","on audit investment","ok",LIME)
        st.markdown("<br>",unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            sl("WITH VS WITHOUT EXPENSEGUARD")
            fig = go.Figure()
            sc2 = ["Without System","With ExpenseGuard"]
            fig.add_trace(go.Bar(name="Fraud Loss",x=sc2,y=[bl,sl2],marker_color=ORG,marker_line_width=0))
            fig.add_trace(go.Bar(name="Audit Cost",x=sc2,y=[tn*ac2*0.1,auc3],marker_color=LIME,marker_line_width=0))
            fig.update_layout(**PT,barmode="stack",height=280,yaxis_title="Total Cost ($)")
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            sl("DETECTION GAUGES")
            def gauge(v,t,color):
                fig = go.Figure(go.Indicator(mode="gauge+number",value=v*100,
                    number={"suffix":"%","font":{"color":color,"size":22,"family":"Space Grotesk"}},
                    title={"text":t,"font":{"color":GRAY,"size":10}},
                    gauge={"axis":{"range":[0,100],"tickcolor":BDR,"tickfont":{"color":DIM,"size":8}},
                        "bar":{"color":color},"bgcolor":BG,"bordercolor":BDR,
                        "steps":[{"range":[0,50],"color":"#1a0808"},{"range":[50,80],"color":"#1a1208"},{"range":[80,100],"color":"#0a1a0a"}],
                        "threshold":{"line":{"color":WHITE,"width":1},"thickness":0.8,"value":80}}))
                fig.update_layout(paper_bgcolor=CARD,plot_bgcolor=CARD,
                    font=dict(color=GRAY,family="Inter"),height=160,margin=dict(l=12,r=12,t=32,b=8))
                return fig
            ga,gb,gc = st.columns(3)
            with ga: st.plotly_chart(gauge(prec,"Precision",LIME),use_container_width=True)
            with gb: st.plotly_chart(gauge(rec,"Recall",ORG),use_container_width=True)
            with gc: st.plotly_chart(gauge(f1v,"F1 Score",LIME),use_container_width=True)

        sl("12-MONTH PROJECTION")
        months = list(range(1,13))
        cs = [ns*m for m in months]; cc2 = [auc3*m for m in months]
        be = next((m for m,s,c2 in zip(months,cs,cc2) if s>c2),None)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=months,y=cs,name="Cumulative Savings",
            line=dict(color=LIME,width=2.5),fill="tozeroy",fillcolor=f"rgba(200,241,53,0.06)"))
        fig2.add_trace(go.Scatter(x=months,y=cc2,name="Audit Cost",
            line=dict(color=ORG,width=1.5,dash="dash")))
        if be:
            fig2.add_vline(x=be,line_dash="dot",line_color=WHITE,line_width=1,
                annotation_text=f"Break-even M{be}",annotation_font_color=WHITE,annotation_font_size=10)
        fig2.update_layout(**PT,height=240,xaxis_title="Month",yaxis_title="Cumulative ($)")
        st.plotly_chart(fig2,use_container_width=True)

        sl("DETAILED BREAKDOWN")
        bd = [("Total Transactions",f"{tn:,}",""),("Total Spend",f"${ts:,.0f}",""),
            ("Flagged",f"{nf2:,}",f"{nf2/tn*100:.1f}% of transactions"),
            ("True Positives",f"{nf2*prec:,.0f}",f"Precision = {prec:.1%}"),
            ("False Positives",f"{nf2*(1-prec):,.0f}","Wasted reviews"),
            ("Fraud Amount Flagged",f"${fa2:,.0f}",""),
            ("Fraud Prevented",f"${fp:,.0f}",f"Recovery {rr:.0%}"),
            ("Audit Cost",f"${auc3:,.0f}",f"${ac2}/review"),
            ("Net Saving",f"${ns:,.0f}","Prevented − Cost"),
            ("Baseline Loss",f"${bl:,.0f}",f"{ifr:.0%} of spend"),
            ("Loss Reduction",f"${lr:,.0f}",f"{lr/bl*100:.1f}% better"),
            ("ROI",f"{roi:.0f}%","Return on audit investment")]
        st.dataframe(pd.DataFrame(bd,columns=["Metric","Value","Notes"]),use_container_width=True,height=360)

st.markdown('</div>',unsafe_allow_html=True)
