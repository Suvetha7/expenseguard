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

st.set_page_config(page_title="ExpenseGuard AI", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main, .stApp { background: #0a0e1a; color: #e2e8f0; }
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155; border-radius: 12px;
    padding: 24px 32px; margin-bottom: 24px;
}
.header-title { font-size: 2rem; font-weight: 600; color: #38bdf8;
    font-family: 'IBM Plex Mono', monospace; }
.header-sub { color: #94a3b8; font-size: 0.9rem; margin-top: 4px; }
.section-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 1rem;
    color: #38bdf8; border-bottom: 1px solid #1e293b;
    padding-bottom: 8px; margin: 20px 0 16px 0;
    text-transform: uppercase; letter-spacing: 1px;
}
.ledger-block {
    background: #0a0e1a; border: 1px solid #1e3a5f;
    border-left: 3px solid #38bdf8; border-radius: 6px;
    padding: 12px 16px; margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; color: #94a3b8;
}
.shap-info { background: #0f172a; border: 1px solid #1e3a5f;
    border-radius: 8px; padding: 16px; margin: 8px 0; }
section[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
.stButton > button {
    background: #1e3a5f; color: #38bdf8; border: 1px solid #38bdf8;
    border-radius: 6px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem; padding: 8px 20px; transition: all 0.2s;
}
.stButton > button:hover { background: #38bdf8; color: #0a0e1a; }
</style>""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(layout=dict(
    paper_bgcolor="#0f172a", plot_bgcolor="#0a0e1a",
    font=dict(family="IBM Plex Sans", color="#94a3b8"),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    legend=dict(bgcolor="#0f172a", bordercolor="#334155"),
    margin=dict(l=20, r=20, t=40, b=20),
))
ANOMALY_COLORS = {
    "Normal": "#38bdf8", "Duplicate": "#f87171",
    "Policy Violation": "#fbbf24", "Ghost Vendor": "#c4b5fd",
    "Redundant Spending": "#34d399"
}

# ── Session State ─────────────────────────────────────────────────────────────
for key, val in [("df_raw", None), ("df_pred", None), ("metrics_before", None),
                  ("metrics_after", None), ("best_model", None),
                  ("pipeline_run", False), ("low_conf_reviewed", {})]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ ExpenseGuard AI")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Overview", "🔍 Anomaly Detection", "🧠 Self-Learning",
        "📊 Model Performance", "🔬 XAI Explainability",
        "🔐 Immutable Ledger", "📈 Analytics",
        "🏆 Risk Leaderboard", "💰 Impact Calculator"
    ])
    st.markdown("---")

    if st.button("⚡ Generate & Train (Full Run)"):
        clear_ledger()
        with st.spinner("Generating dataset..."):
            df = generate_full_dataset(n=1500)
            st.session_state.df_raw = df
        with st.spinner("Training all models with SMOTE balancing..."):
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
        st.success(f"✅ Done! {count} anomalies secured in ledger.")

    st.markdown("---")
    if st.session_state.pipeline_run:
        df_pred = st.session_state.df_pred
        st.markdown(f"**Dataset:** {len(df_pred):,} records")
        n_flag = int((df_pred["prediction"] == 1).sum())
        st.markdown(f"**Flagged:** {n_flag:,} anomalies")
        chain = verify_chain()
        status = "✅ Valid" if chain["valid"] else "❌ Broken"
        st.markdown(f"**Ledger:** {status} ({chain['length']} blocks)")
        if st.session_state.best_model:
            bm = st.session_state.best_model
            auc = st.session_state.metrics_before[bm]["roc_auc"]
            st.markdown(f"**Best Model:** {bm}")
            st.markdown(f"**AUC:** {auc:.4f}")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <div class="header-title">🛡️ EXPENSEGUARD AI</div>
  <div class="header-sub">Self-Learning Expense Auditing System with Immutable Ledger</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    if not st.session_state.pipeline_run:
        st.info("👈 Click **Generate & Train (Full Run)** in the sidebar to initialize the system.")
        st.markdown("""
        ### System Architecture
        | Component | Technology |
        |---|---|
        | Supervised Detection | Random Forest, XGBoost, LightGBM |
        | Unsupervised Detection | Isolation Forest, One-Class SVM |
        | Class Imbalance | SMOTE Oversampling |
        | Self-Learning | Confidence-based human-in-the-loop retraining |
        | Explainability | SHAP (SHapley Additive exPlanations) |
        | Data Security | SHA-256 Blockchain-style ledger + Model Persistence |
        | Dashboard | Streamlit + Plotly |
        """)
    else:
        df_pred = st.session_state.df_pred
        total   = len(df_pred)
        flagged = int((df_pred["prediction"] == 1).sum())
        low_c   = int((df_pred["confidence_level"].astype(str) == "Low").sum())
        chain   = verify_chain()
        bm      = st.session_state.best_model
        auc     = st.session_state.metrics_before[bm]["roc_auc"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Flagged Anomalies", f"{flagged:,}", f"{flagged/total*100:.1f}% of total")
        c3.metric("Low Confidence", f"{low_c:,}")
        c4.metric("Ledger Blocks", chain["length"])
        c5.metric("Best AUC", f"{auc:.4f}")

        st.markdown('<div class="section-header">Anomaly Type Distribution</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        flagged_df = df_pred[df_pred["prediction"] == 1]
        with col1:
            dist = flagged_df["anomaly_type"].value_counts().reset_index()
            dist.columns = ["type","count"]
            fig = px.pie(dist, names="type", values="count", color="type",
                         color_discrete_map=ANOMALY_COLORS, hole=0.5)
            fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=300, title="Flagged by anomaly type")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            dept_flag = flagged_df.groupby("department").size().reset_index(name="count")
            fig2 = px.bar(dept_flag.sort_values("count", ascending=True),
                          x="count", y="department", orientation="h",
                          color_discrete_sequence=["#f87171"])
            fig2.update_layout(**PLOTLY_TEMPLATE["layout"], height=300, title="Flagged by department")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Timeline of Flagged Transactions</div>', unsafe_allow_html=True)
        daily = df_pred[df_pred["prediction"]==1].copy()
        daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M").astype(str)
        monthly = daily.groupby(["month","anomaly_type"]).size().reset_index(name="count")
        fig3 = px.bar(monthly, x="month", y="count", color="anomaly_type",
                      color_discrete_map=ANOMALY_COLORS)
        fig3.update_layout(**PLOTLY_TEMPLATE["layout"], height=280, title="Monthly flagged anomalies")
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Anomaly Detection":
    if not st.session_state.pipeline_run:
        st.warning("Run the full pipeline first.")
    else:
        df_pred = st.session_state.df_pred
        st.markdown('<div class="section-header">Filter & Explore Anomalies</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            atype = st.multiselect("Anomaly Type", df_pred["anomaly_type"].unique().tolist(),
                                   default=["Duplicate","Ghost Vendor","Policy Violation","Redundant Spending"])
        with col2:
            conf_lvl = st.multiselect("Confidence Level", ["High","Medium","Low"], default=["High","Medium"])
        with col3:
            dept_sel = st.multiselect("Department", sorted(df_pred["department"].unique()),
                                       default=sorted(df_pred["department"].unique()))

        mask = ((df_pred["anomaly_type"].isin(atype)) &
                (df_pred["confidence_level"].astype(str).isin(conf_lvl)) &
                (df_pred["department"].isin(dept_sel)) &
                (df_pred["prediction"] == 1))
        filtered = df_pred[mask]
        st.markdown(f"**{len(filtered):,} transactions** match filters")

        display_cols = ["transaction_id","date","employee_id","department",
                        "vendor","category","amount","anomaly_type","confidence","confidence_level"]
        available = [c for c in display_cols if c in filtered.columns]
        show_df = filtered[available].copy()
        show_df["amount"]     = show_df["amount"].apply(lambda x: f"${x:,.2f}")
        show_df["confidence"] = show_df["confidence"].apply(lambda x: f"{x:.1%}")
        st.dataframe(show_df.head(200), use_container_width=True, height=400)

        st.markdown('<div class="section-header">Amount vs Confidence Scatter</div>', unsafe_allow_html=True)
        fig = px.scatter(filtered, x="confidence", y="amount", color="anomaly_type",
                         color_discrete_map=ANOMALY_COLORS,
                         hover_data=["employee_id","vendor","department"],
                         opacity=0.75)
        fig.add_vline(x=LOW_CONFIDENCE_THRESHOLD, line_dash="dash", line_color="#fbbf24",
                      annotation_text="Low confidence boundary")
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Top Flagged Vendors</div>', unsafe_allow_html=True)
        top_v = (filtered.groupby("vendor")
                 .agg(count=("transaction_id","count"), total=("amount","sum"))
                 .sort_values("count", ascending=False).head(10).reset_index())
        top_v["total"] = top_v["total"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top_v, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SELF-LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Self-Learning":
    if not st.session_state.pipeline_run:
        st.warning("Run the full pipeline first.")
    else:
        df_pred = st.session_state.df_pred
        low_conf = df_pred[df_pred["confidence_level"].astype(str) == "Low"].copy()

        st.markdown('<div class="section-header">Confidence Distribution</div>', unsafe_allow_html=True)
        st.markdown(f"**{len(low_conf):,} transactions** below confidence threshold — these drive the self-learning loop.")

        col1, col2 = st.columns([2,1])
        with col1:
            fig = px.histogram(df_pred, x="confidence", color="confidence_level",
                               color_discrete_map={"High":"#f87171","Medium":"#fbbf24","Low":"#34d399"},
                               nbins=40)
            fig.add_vline(x=LOW_CONFIDENCE_THRESHOLD, line_dash="dash", line_color="white")
            fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=280,
                              title="Confidence score distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            breakdown = df_pred["confidence_level"].astype(str).value_counts().reset_index()
            breakdown.columns = ["level","count"]
            fig2 = px.pie(breakdown, names="level", values="count", hole=0.5,
                          color="level",
                          color_discrete_map={"High":"#f87171","Medium":"#fbbf24","Low":"#34d399"})
            fig2.update_layout(**PLOTLY_TEMPLATE["layout"], height=280)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Review Low-Confidence Cases</div>', unsafe_allow_html=True)
        st.info("Correct labels below → these samples retrain the model (human-in-the-loop).")
        review_sample = low_conf.sample(min(15, len(low_conf)), random_state=1)
        for idx, row in review_sample.iterrows():
            with st.expander(f"TXN `{row['transaction_id']}` — {row.get('vendor','?')} | ${row.get('amount',0):,.2f} | {row.get('department','')}"):
                cols = st.columns([2,1,1])
                cols[0].markdown(f"**Predicted:** `{row.get('anomaly_type','?')}` | **Confidence:** `{row['confidence']:.1%}`")
                label = cols[1].selectbox("Correct label", ["Anomaly (1)","Normal (0)"],
                                          key=f"lbl_{idx}",
                                          index=0 if row.get("prediction",1)==1 else 1)
                if cols[2].button("✓ Confirm", key=f"conf_{idx}"):
                    st.session_state.low_conf_reviewed[idx] = 1 if "Anomaly" in label else 0

        reviewed = len(st.session_state.low_conf_reviewed)
        st.markdown(f"**{reviewed} samples** confirmed for retraining.")

        if st.button("🔄 Trigger Retraining") and reviewed > 0:
            reviewed_df = review_sample[review_sample.index.isin(
                st.session_state.low_conf_reviewed.keys())].copy()
            reviewed_df["label"] = reviewed_df.index.map(st.session_state.low_conf_reviewed)
            with st.spinner("Retraining with corrected labels..."):
                results_after, best_after = self_learning_retrain(reviewed_df, FEATURE_COLS)
            if results_after:
                st.session_state.metrics_after = results_after
                st.success(f"✅ Retrained! Best: **{best_after}**")
                st.session_state.low_conf_reviewed = {}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    if not st.session_state.pipeline_run or not st.session_state.metrics_before:
        st.warning("Run the full pipeline first.")
    else:
        metrics_before = st.session_state.metrics_before
        metrics_after  = st.session_state.metrics_after
        metric_names   = ["accuracy","precision","recall","f1","roc_auc"]

        st.markdown('<div class="section-header">Model Metrics</div>', unsafe_allow_html=True)
        rows = []
        for mname, m in metrics_before.items():
            row = {"Model": mname}
            for mn in metric_names:
                row[mn.upper()] = f"{m.get(mn,0):.4f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown('<div class="section-header">Radar Chart — Model Comparison</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for mname, m in metrics_before.items():
            vals = [m.get(mn,0) for mn in metric_names] + [m.get(metric_names[0],0)]
            cats = [mn.upper() for mn in metric_names] + [metric_names[0].upper()]
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats, name=mname, fill="toself", opacity=0.6))
        fig.update_layout(polar=dict(bgcolor="#0a0e1a",
                                     radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e293b"),
                                     angularaxis=dict(gridcolor="#1e293b")),
                          paper_bgcolor="#0f172a", font=dict(color="#94a3b8"),
                          height=380, legend=dict(bgcolor="#0f172a"))
        st.plotly_chart(fig, use_container_width=True)

        if metrics_after:
            st.markdown('<div class="section-header">Before vs After Retraining</div>', unsafe_allow_html=True)
            sup_models = [m for m in metrics_before if m not in ("Isolation Forest","One-Class SVM")]
            f1_before = [metrics_before[m].get("f1",0) for m in sup_models]
            f1_after  = [metrics_after.get(m,{}).get("f1",0) for m in sup_models]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Before", x=sup_models, y=f1_before,
                                   marker_color="#38bdf8", opacity=0.8))
            fig2.add_trace(go.Bar(name="After Retraining", x=sup_models, y=f1_after,
                                   marker_color="#34d399", opacity=0.8))
            fig2.update_layout(**PLOTLY_TEMPLATE["layout"], barmode="group",
                               height=320, yaxis_range=[0,1],
                               title="F1 Score: Before vs After Self-Learning")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Complete Self-Learning retraining to see before/after comparison.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: XAI EXPLAINABILITY (SHAP)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 XAI Explainability":
    if not st.session_state.pipeline_run:
        st.warning("Run the full pipeline first.")
    else:
        st.markdown('<div class="section-header">SHAP — Feature Importance (Global)</div>', unsafe_allow_html=True)
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** measures each feature's contribution to the model's predictions.
        Based on game theory — each feature is assigned a fair share of the prediction outcome.
        """)

        df_raw = st.session_state.df_raw
        with st.spinner("Computing SHAP values..."):
            shap_vals, X_data = get_shap_values(df_raw, FEATURE_COLS)

        if shap_vals is not None:
            # Global mean absolute SHAP
            mean_shap = np.abs(shap_vals).mean(axis=0)
            shap_df = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Mean |SHAP|": mean_shap
            }).sort_values("Mean |SHAP|", ascending=True).tail(15)

            fig = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                         color="Mean |SHAP|", color_continuous_scale="Blues")
            fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=450,
                              title="Top 15 Features by Mean |SHAP| Value",
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance table
            st.markdown('<div class="section-header">Feature Ranking Table</div>', unsafe_allow_html=True)
            rank_df = shap_df.sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
            rank_df.index += 1
            rank_df["Mean |SHAP|"] = rank_df["Mean |SHAP|"].apply(lambda x: f"{x:.6f}")
            st.dataframe(rank_df, use_container_width=True)

            # SHAP distribution per feature (top 6)
            st.markdown('<div class="section-header">SHAP Value Distribution — Top 6 Features</div>', unsafe_allow_html=True)
            top6_idx = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1][:6]

            cols = st.columns(3)
            for i, feat_idx in enumerate(top6_idx):
                feat_name = FEATURE_COLS[feat_idx]
                sv = shap_vals[:, feat_idx]
                with cols[i % 3]:
                    fig_hist = px.histogram(x=sv, nbins=30,
                                            color_discrete_sequence=["#38bdf8"])
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="#f87171")
                    fig_hist.update_layout(**PLOTLY_TEMPLATE["layout"], height=200,
                                          title=feat_name,
                                          xaxis_title="SHAP Value",
                                          yaxis_title="Count",
                                          showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)

            # Waterfall for single transaction
            st.markdown('<div class="section-header">Single Transaction Explanation</div>', unsafe_allow_html=True)
            st.markdown("SHAP waterfall for the highest-confidence flagged transaction:")

            df_pred = st.session_state.df_pred
            flagged_idx = df_pred[df_pred["prediction"]==1]["confidence"].idxmax()
            txn_pos = df_pred.index.get_loc(flagged_idx)
            sv_single = shap_vals[txn_pos]

            wf_df = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "SHAP Value": sv_single
            }).sort_values("SHAP Value", key=abs, ascending=True).tail(12)

            colors = ["#f87171" if v > 0 else "#34d399" for v in wf_df["SHAP Value"]]
            fig_wf = go.Figure(go.Bar(
                x=wf_df["SHAP Value"], y=wf_df["Feature"],
                orientation="h", marker_color=colors
            ))
            fig_wf.add_vline(x=0, line_color="#64748b")
            fig_wf.update_layout(**PLOTLY_TEMPLATE["layout"], height=380,
                                  title=f"Why was {df_pred.loc[flagged_idx,'transaction_id']} flagged?",
                                  xaxis_title="SHAP Value (red = pushes toward fraud)")
            st.plotly_chart(fig_wf, use_container_width=True)

            txn = df_pred.loc[flagged_idx]
            st.markdown(f"""
            **Transaction Details:** `{txn['transaction_id']}` | 
            Amount: **${txn['amount']:,.2f}** | 
            Dept: **{txn['department']}** | 
            Type: **{txn['anomaly_type']}** | 
            Confidence: **{txn['confidence']:.1%}**
            """)
        else:
            st.error("Could not compute SHAP values. Make sure `shap` is installed: `pip install shap`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: IMMUTABLE LEDGER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔐 Immutable Ledger":
    st.markdown('<div class="section-header">Blockchain-Inspired Immutable Ledger</div>', unsafe_allow_html=True)
    chain_info = verify_chain()

    col1, col2, col3 = st.columns(3)
    col1.metric("Chain Integrity", "✅ VALID" if chain_info["valid"] else "❌ BROKEN")
    col2.metric("Total Blocks", chain_info["length"])
    col3.metric("Hash Algorithm", "SHA-256")

    ledger_df = get_ledger_df()
    if ledger_df.empty:
        st.info("No entries yet. Run the full pipeline to populate.")
    else:
        st.markdown('<div class="section-header">Ledger Entries</div>', unsafe_allow_html=True)
        st.dataframe(ledger_df, use_container_width=True, height=350)

        st.markdown('<div class="section-header">Block Chain Visualization</div>', unsafe_allow_html=True)
        for _, row in ledger_df.head(6).iterrows():
            st.markdown(f"""<div class="ledger-block">
                <span style="color:#64748b">Block #{int(row['index'])}</span> &nbsp;|&nbsp;
                <span style="color:#e2e8f0">{row['transaction_id']}</span> &nbsp;|&nbsp;
                <span style="color:#fbbf24">{row['anomaly_type']}</span> &nbsp;|&nbsp;
                ${float(row['amount']):,.2f}<br>
                <span style="color:#475569">Hash:</span> <span style="color:#38bdf8">{row['hash']}</span><br>
                <span style="color:#475569">Prev:</span> <span style="color:#64748b">{row['prev_hash']}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Add Manual Entry</div>', unsafe_allow_html=True)
        with st.form("manual_ledger"):
            c1, c2, c3 = st.columns(3)
            txn_id   = c1.text_input("Transaction ID", value="TXN_MANUAL_001")
            amount   = c2.number_input("Amount ($)", value=5000.0)
            atype    = c3.selectbox("Anomaly Type", ["Duplicate","Ghost Vendor","Policy Violation","Redundant Spending"])
            dept     = c1.text_input("Department", value="Finance")
            emp      = c2.text_input("Employee ID", value="EMP0042")
            conf_val = c3.slider("Confidence", 0.0, 1.0, 0.9)
            if st.form_submit_button("🔐 Add to Ledger"):
                block = add_to_ledger({"transaction_id":txn_id,"amount":amount,
                                       "department":dept,"employee_id":emp,
                                       "vendor":"Manual","category":"Other","date":"2024-01-01"},
                                      atype, conf_val, "Manual")
                st.success(f"Block #{block['index']} added. Hash: `{block['hash'][:32]}...`")
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    if not st.session_state.pipeline_run:
        st.warning("Run the full pipeline first.")
    else:
        df_pred = st.session_state.df_pred
        flagged = df_pred[df_pred["prediction"] == 1]

        st.markdown('<div class="section-header">Spending Analytics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            cat_spend = df_pred.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
            fig = px.bar(cat_spend, x="category", y="amount", color_discrete_sequence=["#38bdf8"])
            fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=280, title="Total spend by category")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            dept_spend = df_pred.groupby("department")["amount"].sum().reset_index().sort_values("amount", ascending=False)
            fig2 = px.bar(dept_spend, x="amount", y="department", orientation="h",
                          color_discrete_sequence=["#c4b5fd"])
            fig2.update_layout(**PLOTLY_TEMPLATE["layout"], height=280, title="Total spend by department")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Financial Impact of Fraud</div>', unsafe_allow_html=True)
        total_amt   = df_pred["amount"].sum()
        flagged_amt = flagged["amount"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total At-Risk Amount", f"${flagged_amt/1e6:.2f}M")
        c2.metric("% of Total Spend",     f"{flagged_amt/total_amt*100:.1f}%")
        c3.metric("Avg Flagged Amount",   f"${flagged['amount'].mean():,.0f}")

        st.markdown('<div class="section-header">Anomaly Heatmap: Department × Category</div>', unsafe_allow_html=True)
        heat = flagged.groupby(["department","category"]).size().reset_index(name="count")
        heat_pivot = heat.pivot(index="department", columns="category", values="count").fillna(0)
        fig3 = px.imshow(heat_pivot, color_continuous_scale="Blues", text_auto=True, aspect="auto")
        fig3.update_layout(**PLOTLY_TEMPLATE["layout"], height=350)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="section-header">Top High-Risk Employees</div>', unsafe_allow_html=True)
        emp_risk = (flagged.groupby("employee_id")
                    .agg(flags=("transaction_id","count"), total=("amount","sum"), dept=("department","first"))
                    .sort_values("flags", ascending=False).head(15).reset_index())
        emp_risk["total"] = emp_risk["total"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(emp_risk, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RISK LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Risk Leaderboard":
    if not st.session_state.pipeline_run:
        st.warning("Run the full pipeline first.")
    else:
        df_pred = st.session_state.df_pred
        flagged = df_pred[df_pred["prediction"] == 1].copy()

        st.markdown('<div class="section-header">Employee Risk Leaderboard</div>', unsafe_allow_html=True)
        st.markdown("Composite risk score = (flag count × 0.4) + (total at-risk amount normalised × 0.4) + (avg confidence × 0.2)")

        emp_stats = (flagged.groupby("employee_id")
            .agg(
                flags=("transaction_id", "count"),
                total_risk=("amount", "sum"),
                avg_conf=("confidence", "mean"),
                dept=("department", "first"),
                anomaly_types=("anomaly_type", lambda x: ", ".join(x.unique()))
            ).reset_index())

        max_risk = emp_stats["total_risk"].max() or 1
        emp_stats["risk_score"] = (
            emp_stats["flags"] / emp_stats["flags"].max() * 40 +
            emp_stats["total_risk"] / max_risk * 40 +
            emp_stats["avg_conf"] * 20
        ).round(1)
        emp_stats = emp_stats.sort_values("risk_score", ascending=False).reset_index(drop=True)
        emp_stats.index += 1

        def risk_badge(score):
            if score >= 70: return "🔴 CRITICAL"
            elif score >= 50: return "🟠 HIGH"
            elif score >= 30: return "🟡 MEDIUM"
            else: return "🟢 LOW"

        emp_stats["Risk Level"] = emp_stats["risk_score"].apply(risk_badge)
        emp_stats["total_risk_fmt"] = emp_stats["total_risk"].apply(lambda x: f"${x:,.0f}")
        emp_stats["avg_conf_fmt"] = emp_stats["avg_conf"].apply(lambda x: f"{x:.1%}")

        # Top 3 highlight cards
        top3 = emp_stats.head(3)
        medals = ["🥇", "🥈", "🥉"]
        cols = st.columns(3)
        for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
            col.markdown(f"""
            <div style="background:#0f172a;border:1px solid #1e293b;border-top:3px solid #f87171;
                border-radius:8px;padding:16px;text-align:center;">
                <div style="font-size:1.8rem">{medals[i]}</div>
                <div style="font-family:'IBM Plex Mono',monospace;color:#f87171;font-size:1.1rem;
                    font-weight:600;margin:6px 0">{row['employee_id']}</div>
                <div style="color:#94a3b8;font-size:0.78rem">{row['dept']}</div>
                <div style="color:#fbbf24;font-size:1.4rem;font-weight:700;margin:8px 0">
                    {row['risk_score']:.0f}<span style="font-size:0.7rem;color:#64748b"> /100</span></div>
                <div style="color:#94a3b8;font-size:0.75rem">{row['flags']} flags · {row['total_risk_fmt']}</div>
                <div style="margin-top:8px;font-size:0.72rem;color:#475569">{row['anomaly_types'][:40]}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Full Employee Rankings</div>', unsafe_allow_html=True)

        # Horizontal bar chart
        top20 = emp_stats.head(20)
        colors = ["#f87171" if s >= 70 else "#fb923c" if s >= 50 else "#fbbf24" if s >= 30 else "#34d399"
                  for s in top20["risk_score"]]
        fig = go.Figure(go.Bar(
            x=top20["risk_score"],
            y=top20["employee_id"],
            orientation="h",
            marker_color=colors,
            text=top20["risk_score"].apply(lambda x: f"{x:.0f}"),
            textposition="outside",
            customdata=top20[["dept","flags","total_risk_fmt","Risk Level"]].values,
            hovertemplate="<b>%{y}</b><br>Dept: %{customdata[0]}<br>Flags: %{customdata[1]}<br>At-Risk: %{customdata[2]}<br>Level: %{customdata[3]}<extra></extra>"
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=520,
                          xaxis_title="Risk Score (0-100)",
                          xaxis_range=[0, 110],
                          title="Top 20 Highest-Risk Employees")
        st.plotly_chart(fig, use_container_width=True)

        # Full table
        display_emp = emp_stats[["employee_id","dept","flags","total_risk_fmt","avg_conf_fmt","anomaly_types","risk_score","Risk Level"]].copy()
        display_emp.columns = ["Employee ID","Department","Flags","Total At-Risk","Avg Confidence","Anomaly Types","Risk Score","Risk Level"]
        st.dataframe(display_emp.head(30), use_container_width=True, height=400)

        # Vendor Risk Leaderboard
        st.markdown('<div class="section-header">Vendor Risk Leaderboard</div>', unsafe_allow_html=True)
        vendor_stats = (flagged.groupby("vendor")
            .agg(
                flags=("transaction_id", "count"),
                total_risk=("amount", "sum"),
                depts=("department", "nunique"),
                avg_conf=("confidence", "mean"),
                anomaly_types=("anomaly_type", lambda x: ", ".join(x.unique()))
            ).reset_index())

        max_v_risk = vendor_stats["total_risk"].max() or 1
        vendor_stats["vendor_risk_score"] = (
            vendor_stats["flags"] / vendor_stats["flags"].max() * 40 +
            vendor_stats["total_risk"] / max_v_risk * 40 +
            vendor_stats["avg_conf"] * 20
        ).round(1)
        vendor_stats = vendor_stats.sort_values("vendor_risk_score", ascending=False).reset_index(drop=True)
        vendor_stats["Risk Level"] = vendor_stats["vendor_risk_score"].apply(risk_badge)
        vendor_stats["total_risk"] = vendor_stats["total_risk"].apply(lambda x: f"${x:,.0f}")
        vendor_stats["avg_conf"] = vendor_stats["avg_conf"].apply(lambda x: f"{x:.1%}")
        vendor_stats.index += 1

        col1, col2 = st.columns(2)
        with col1:
            top_v = vendor_stats.head(10)
            v_colors = ["#f87171" if s >= 70 else "#fb923c" if s >= 50 else "#fbbf24"
                        for s in top_v["vendor_risk_score"]]
            fig_v = go.Figure(go.Bar(
                x=top_v["vendor_risk_score"], y=top_v["vendor"],
                orientation="h", marker_color=v_colors,
                text=top_v["vendor_risk_score"].apply(lambda x: f"{x:.0f}"),
                textposition="outside"
            ))
            fig_v.update_layout(**PLOTLY_TEMPLATE["layout"], height=320,
                                xaxis_range=[0, 115], title="Top 10 Highest-Risk Vendors")
            st.plotly_chart(fig_v, use_container_width=True)

        with col2:
            risk_dist = vendor_stats["Risk Level"].value_counts().reset_index()
            risk_dist.columns = ["level", "count"]
            risk_colors = {"🔴 CRITICAL":"#f87171","🟠 HIGH":"#fb923c",
                           "🟡 MEDIUM":"#fbbf24","🟢 LOW":"#34d399"}
            fig_d = px.pie(risk_dist, names="level", values="count",
                           color="level", color_discrete_map=risk_colors, hole=0.55)
            fig_d.update_layout(**PLOTLY_TEMPLATE["layout"], height=320,
                                title="Vendor Risk Distribution")
            st.plotly_chart(fig_d, use_container_width=True)

        st.dataframe(vendor_stats[["vendor","flags","total_risk","depts","avg_conf","anomaly_types","vendor_risk_score","Risk Level"]].head(20),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FINANCIAL IMPACT CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Impact Calculator":
    if not st.session_state.pipeline_run:
        st.warning("Run the full pipeline first.")
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

        st.markdown('<div class="section-header">Assumptions & Parameters</div>', unsafe_allow_html=True)
        st.markdown("Adjust the sliders to model different scenarios.")

        col1, col2, col3 = st.columns(3)
        with col1:
            recovery_rate = st.slider("Fraud Recovery Rate (%)",
                help="What % of flagged fraud amount is actually recovered/prevented",
                min_value=10, max_value=100, value=75, step=5) / 100
        with col2:
            audit_cost_per_txn = st.slider("Cost to Audit 1 Transaction ($)",
                help="Labour cost to manually review one flagged transaction",
                min_value=5, max_value=200, value=25, step=5)
        with col3:
            industry_fraud_rate = st.slider("Industry Avg Fraud Rate (%)",
                help="Typical % of spend lost to fraud without any system",
                min_value=1, max_value=15, value=5, step=1) / 100

        st.markdown('<div class="section-header">Financial Impact Summary</div>', unsafe_allow_html=True)

        # Core calculations
        true_positives   = n_flagged * precision
        false_positives  = n_flagged * (1 - precision)
        fraud_prevented  = flagged_amt * precision * recovery_rate
        audit_cost       = n_flagged * audit_cost_per_txn
        net_savings      = fraud_prevented - audit_cost
        baseline_loss    = total_spend * industry_fraud_rate
        system_loss      = flagged_amt * (1 - precision) + (total_spend - flagged_amt) * (1 - recall) * 0.05
        loss_reduction   = baseline_loss - system_loss
        roi              = (net_savings / audit_cost * 100) if audit_cost > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fraud Prevented", f"${fraud_prevented:,.0f}",
                  delta=f"{fraud_prevented/total_spend*100:.1f}% of spend")
        c2.metric("Audit Cost", f"${audit_cost:,.0f}",
                  delta=f"{n_flagged} reviews")
        c3.metric("Net Savings", f"${net_savings:,.0f}",
                  delta="Prevented - Cost")
        c4.metric("Return on Investment", f"{roi:.0f}%",
                  delta="vs manual auditing")

        # Visual comparison: With vs Without system
        st.markdown('<div class="section-header">With vs Without ExpenseGuard AI</div>', unsafe_allow_html=True)

        comparison_data = {
            "Scenario": ["Without System\n(Manual Audit)", "With ExpenseGuard AI"],
            "Fraud Loss ($)": [baseline_loss, system_loss],
            "Audit Cost ($)": [total_txns * audit_cost_per_txn * 0.1, audit_cost],
            "Total Cost ($)": [baseline_loss + total_txns * audit_cost_per_txn * 0.1,
                               system_loss + audit_cost],
        }
        comp_df = pd.DataFrame(comparison_data)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Fraud Loss", x=comp_df["Scenario"],
                             y=comp_df["Fraud Loss ($)"], marker_color="#f87171"))
        fig.add_trace(go.Bar(name="Audit Cost", x=comp_df["Scenario"],
                             y=comp_df["Audit Cost ($)"], marker_color="#fbbf24"))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], barmode="stack", height=360,
                          yaxis_title="Total Cost ($)",
                          title="Cost Comparison: Manual vs AI-Assisted Auditing")
        st.plotly_chart(fig, use_container_width=True)

        # Gauge: Detection Coverage
        st.markdown('<div class="section-header">Detection Coverage Gauges</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        def make_gauge(val, title, color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=val * 100,
                delta={"reference": 70, "valueformat": ".1f"},
                number={"suffix": "%", "font": {"color": color, "size": 28}},
                title={"text": title, "font": {"color": "#94a3b8", "size": 12}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#334155",
                             "tickfont": {"color": "#64748b"}},
                    "bar": {"color": color},
                    "bgcolor": "#0a0e1a",
                    "bordercolor": "#1e293b",
                    "steps": [
                        {"range": [0, 50], "color": "#1a0a0a"},
                        {"range": [50, 75], "color": "#1a1a0a"},
                        {"range": [75, 100], "color": "#0a1a0a"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2},
                                  "thickness": 0.8, "value": 80}
                }
            ))
            fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                              font=dict(color="#94a3b8"), height=220,
                              margin=dict(l=20, r=20, t=40, b=10))
            return fig

        with col1:
            st.plotly_chart(make_gauge(precision, "Precision<br>(Fraud Correctly Flagged)", "#38bdf8"),
                           use_container_width=True)
        with col2:
            st.plotly_chart(make_gauge(recall, "Recall<br>(Fraud Not Missed)", "#34d399"),
                           use_container_width=True)
        with col3:
            st.plotly_chart(make_gauge(f1, "F1 Score<br>(Overall Balance)", "#fbbf24"),
                           use_container_width=True)

        # Savings breakdown table
        st.markdown('<div class="section-header">Detailed Breakdown</div>', unsafe_allow_html=True)
        breakdown_data = [
            ("Total Transactions Analysed", f"{total_txns:,}", ""),
            ("Total Spend", f"${total_spend:,.0f}", ""),
            ("Transactions Flagged", f"{n_flagged:,}", f"{n_flagged/total_txns*100:.1f}% of total"),
            ("True Positives (Real Fraud)", f"{true_positives:,.0f}", f"Precision = {precision:.1%}"),
            ("False Positives (False Alarms)", f"{false_positives:,.0f}", f"Wasted reviews"),
            ("Fraud Amount Flagged", f"${flagged_amt:,.0f}", f"{flagged_amt/total_spend*100:.1f}% of spend"),
            ("Fraud Prevented (after recovery)", f"${fraud_prevented:,.0f}", f"Recovery rate = {recovery_rate:.0%}"),
            ("Total Audit Cost", f"${audit_cost:,.0f}", f"${audit_cost_per_txn}/review × {n_flagged} reviews"),
            ("Net Financial Saving", f"${net_savings:,.0f}", "Prevented - Audit Cost"),
            ("Without System — Baseline Loss", f"${baseline_loss:,.0f}", f"Industry avg {industry_fraud_rate:.0%} of spend"),
            ("Loss Reduction vs Baseline", f"${loss_reduction:,.0f}", f"{loss_reduction/baseline_loss*100:.1f}% improvement"),
            ("ROI on Audit Investment", f"{roi:.0f}%", "Every $1 spent saves ${roi/100:.1f}"),
        ]
        bd_df = pd.DataFrame(breakdown_data, columns=["Metric", "Value", "Notes"])
        st.dataframe(bd_df, use_container_width=True, height=380)

        # Scenario comparison
        st.markdown('<div class="section-header">Scenario Analysis — Scale Over Time</div>', unsafe_allow_html=True)
        months = list(range(1, 13))
        cumulative_savings = [net_savings * m for m in months]
        cumulative_costs   = [audit_cost * m for m in months]
        breakeven = next((m for m, s, c in zip(months, cumulative_savings, cumulative_costs) if s > c), None)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=months, y=cumulative_savings, name="Cumulative Savings",
                                  line=dict(color="#34d399", width=2.5), fill="tozeroy",
                                  fillcolor="rgba(52,211,153,0.06)"))
        fig2.add_trace(go.Scatter(x=months, y=cumulative_costs, name="Cumulative Audit Cost",
                                  line=dict(color="#f87171", width=2, dash="dash")))
        if breakeven:
            fig2.add_vline(x=breakeven, line_dash="dot", line_color="#fbbf24",
                           annotation_text=f"Break-even: Month {breakeven}",
                           annotation_font_color="#fbbf24")
        fig2.update_layout(**PLOTLY_TEMPLATE["layout"], height=300,
                           xaxis_title="Month", yaxis_title="Cumulative Amount ($)",
                           title="12-Month Projected Savings vs Cost")
        st.plotly_chart(fig2, use_container_width=True)
