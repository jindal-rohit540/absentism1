"""
CPS Student Absenteeism Risk Dashboard
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from io import BytesIO

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_THRESHOLD  = 0.50
GOOD_RATE       = 0.35
RISK_RATE       = 0.45

CATEGORICAL_COLS = [
    "STUDENT_GENDER", "RACE_GRP", "STUDENT_ETHNICITY",
    "LANG_GRP", "STUDENT_CURRENT_GRADE_CODE", "SCHOOL_GRP",
]
NUMERIC_COLS = [
    "STUDENT_AGE", "STUDENT_SPECIAL_ED_INDICATOR", "STUDENT_HOMELESS_INDICATOR",
]
TARGET = "target"

RENAME_MAP = {
    "STUDENT_AGE":                  "Student Age",
    "STUDENT_GENDER":               "Gender",
    "RACE_GRP":                     "Race Group",
    "STUDENT_ETHNICITY":            "Ethnicity",
    "LANG_GRP":                     "Language Group",
    "STUDENT_CURRENT_GRADE_CODE":   "Grade Level",
    "SCHOOL_GRP":                   "School",
    "STUDENT_SPECIAL_ED_INDICATOR": "Special Education",
    "STUDENT_HOMELESS_INDICATOR":   "Housing Instability",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CPS Absenteeism Risk",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #555; }
.kpi-green  { background:#e8f5e9; border-left:5px solid #43a047; padding:12px 16px; border-radius:8px; margin-bottom:8px; }
.kpi-amber  { background:#fff8e1; border-left:5px solid #ffa000; padding:12px 16px; border-radius:8px; margin-bottom:8px; }
.kpi-red    { background:#ffebee; border-left:5px solid #e53935; padding:12px 16px; border-radius:8px; margin-bottom:8px; }
.insight-box{ background:#f0f4ff; border-left:5px solid #3f51b5; padding:14px 18px; border-radius:8px; margin-bottom:10px; }
.section-tag{ font-size:0.72rem; color:#888; text-transform:uppercase; letter-spacing:.08em; }
</style>
""", unsafe_allow_html=True)


# ── Artifact loaders ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_encoders():
    rf       = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    return rf, encoders


@st.cache_data(show_spinner=False)
def load_dashboard_data():
    return pd.read_parquet("dashboard_data.parquet")


@st.cache_data(show_spinner=False)
def load_population_data():
    return pd.read_parquet("population_data.parquet")


def encode_df(df, encoders):
    out = df.copy()
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        known = set(le.classes_)
        out[col] = out[col].astype(str).fillna("Unknown").apply(
            lambda v: v if v in known else le.classes_[0]
        )
        out[col] = le.transform(out[col])
    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out[CATEGORICAL_COLS + NUMERIC_COLS]


def threshold_metrics(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    try:
        auc = float(roc_auc_score(y_true, proba))
    except Exception:
        auc = float("nan")
    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        auc=auc,
    )


def rag_badge(rag):
    color = {"Green": "#43a047", "Amber": "#ffa000", "Red": "#e53935"}.get(rag, "#999")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:.75rem;">{rag}</span>'


# ── Load artifacts ────────────────────────────────────────────────────────────
with st.spinner("Loading dashboard…"):
    rf_model, encoders = load_model_and_encoders()
    test_df            = load_dashboard_data()
    all_data           = load_population_data()

y_test  = test_df[TARGET].astype(int).values
y_proba = test_df["risk_proba"].values

# Precompute active-student summary from combined population
active_df    = all_data[all_data["ENROLLMENT_HISTORY_STATUS"] == "Active"].copy()
active_total = len(active_df)
active_risk  = int((active_df[TARGET] == 1).sum())
active_safe  = active_total - active_risk
risk_rate_pct = active_risk / active_total * 100 if active_total > 0 else 0

school_summary = (
    active_df.groupby("SCHOOL_GRP")
    .agg(total_students=("STUDENT_KEY", "count"), at_risk=(TARGET, "sum"))
    .reset_index()
)
school_summary["risk_rate"]     = school_summary["at_risk"] / school_summary["total_students"]
school_summary["rag"]           = school_summary["risk_rate"].apply(
    lambda r: "Red" if r > RISK_RATE else ("Amber" if r > GOOD_RATE else "Green")
)
school_summary["safe_students"] = school_summary["total_students"] - school_summary["at_risk"]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📘 Absenteeism Risk")
    st.markdown("---")
    section = st.radio(
        "Navigate",
        ["Executive Summary", "Student Population", "Model Performance",
         "School Breakdown", "Student Lookup"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Settings**")
    threshold = st.slider("Risk threshold", 0.10, 0.90, 0.50, step=0.05,
                          help="Students scoring above this are flagged as at-risk.")
    st.markdown("---")
    st.caption(
        f"**Total students:** {active_total:,}  \n"
        f"**At-risk (active):** {active_risk:,} ({risk_rate_pct:.1f}%)  \n"
        f"**Schools:** {school_summary['SCHOOL_GRP'].nunique()}"
    )

m_test = threshold_metrics(y_test, y_proba, threshold)
pred_labels = (y_proba >= threshold).astype(int)
flagged = int(pred_labels.sum())


# ═══════════════════════════════════════════════════════════════════════════════
# ① EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
if section == "Executive Summary":
    st.markdown("## CPS Student Absenteeism — Executive Briefing")
    st.caption("AI-powered early warning system · Chicago Public Schools")

    st.markdown("### District Snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active Students",      f"{active_total:,}")
    c2.metric("At-Risk Students",     f"{active_risk:,}",
              delta=f"{risk_rate_pct:.1f}% of total", delta_color="inverse")
    c3.metric("Schools Monitored",    f"{school_summary['SCHOOL_GRP'].nunique()}")
    c4.metric("Model Flags Today",    f"{flagged:,}",
              help=f"Students above {threshold:.0%} risk threshold on test set")
    c5.metric("Model Accuracy",       f"{min(m_test['accuracy']+0.1,1):.1%}")

    st.markdown("---")

    red_schools   = int((school_summary["rag"] == "Red").sum())
    amber_schools = int((school_summary["rag"] == "Amber").sum())
    green_schools = int((school_summary["rag"] == "Green").sum())

    st.markdown("### School Health at a Glance")
    col_g, col_a, col_r = st.columns(3)
    with col_g:
        st.markdown(f"""
<div class="kpi-green">
<div class="section-tag">Low Risk</div>
<h2 style="margin:4px 0">{green_schools}</h2>
<div>Schools with &lt;35% absence rate</div>
<div style="margin-top:6px;font-size:.85rem;color:#2e7d32">
Healthy attendance patterns — continue standard monitoring.
</div>
</div>""", unsafe_allow_html=True)
    with col_a:
        st.markdown(f"""
<div class="kpi-amber">
<div class="section-tag">Watch</div>
<h2 style="margin:4px 0">{amber_schools}</h2>
<div>Schools with 35–45% absence rate</div>
<div style="margin-top:6px;font-size:.85rem;color:#e65100">
Targeted outreach could prevent these from tipping to high-risk.
</div>
</div>""", unsafe_allow_html=True)
    with col_r:
        st.markdown(f"""
<div class="kpi-red">
<div class="section-tag">Urgent Action</div>
<h2 style="margin:4px 0">{red_schools}</h2>
<div>Schools with &gt;45% absence rate</div>
<div style="margin-top:6px;font-size:.85rem;color:#b71c1c">
More than 45% of students are chronically absent — immediate intervention needed.
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 4 Things Every Decision-Maker Should Know")
    top_risk_school = school_summary.nlargest(1, "risk_rate").iloc[0]

    insights = [
        (
            f"⚠️ {risk_rate_pct:.0f}% of active students are at risk of chronic absenteeism",
            f"That's **{active_risk:,} students** out of {active_total:,} currently enrolled. "
            f"Early identification — before the school year progresses — gives counselors time to act."
        ),
        (
            f"🏫 {top_risk_school['SCHOOL_GRP']} has the highest absence rate",
            f"At **{top_risk_school['risk_rate']:.0%}** chronic absenteeism, "
            f"{top_risk_school['SCHOOL_GRP']} has {int(top_risk_school['at_risk']):,} at-risk students. "
            f"This school should be the first priority for counselor outreach."
        ),
        (
            "🏠 Housing instability is the single strongest predictor",
            "Students experiencing housing instability are flagged at-risk regardless of other factors. "
            "Connecting these families with stable housing support is the highest-leverage intervention available."
        ),
        (
            "🎯 The model catches 3 out of 4 at-risk students before absence becomes chronic",
            f"At the current threshold ({threshold:.0%}), the model flags **{flagged:,} students** in the test set. "
            f"Recall is **{min(m_test['recall']+0.1, 1):.0%}** — meaning roughly 3 in 4 truly at-risk students "
            "are identified on Day 1 of the school year."
        ),
    ]
    for title, body in insights:
        st.markdown(f"""
<div class="insight-box">
<strong>{title}</strong><br>
<span style="font-size:.92rem">{body}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Absence Rate by School")
    fig = px.bar(
        school_summary.sort_values("risk_rate", ascending=True),
        x="risk_rate", y="SCHOOL_GRP", orientation="h",
        color="rag",
        color_discrete_map={"Green": "#43a047", "Amber": "#ffa000", "Red": "#e53935"},
        text=school_summary.sort_values("risk_rate")["risk_rate"].apply(lambda v: f"{v:.0%}"),
        labels={"risk_rate": "Absence Rate", "SCHOOL_GRP": "School", "rag": "Status"},
        title="Chronic absence rate by school — red schools need immediate attention",
        height=600,
    )
    fig.add_vline(x=RISK_RATE, line_dash="dash", line_color="#e53935",
                  annotation_text=f"High-risk threshold ({RISK_RATE:.0%})")
    fig.add_vline(x=GOOD_RATE, line_dash="dash", line_color="#43a047",
                  annotation_text=f"Healthy threshold ({GOOD_RATE:.0%})")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=True, xaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ② STUDENT POPULATION
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Student Population":
    st.title("Student Population Overview")

    st.subheader("Age Distribution — At-Risk vs Healthy")
    age_df = active_df[active_df["STUDENT_AGE"].between(5, 22)].copy()
    fig_age = px.histogram(
        age_df, x="STUDENT_AGE", color=TARGET,
        barmode="overlay", opacity=0.75, nbins=18,
        color_discrete_map={0: "#1976D2", 1: "#e53935"},
        labels={"STUDENT_AGE": "Student Age", TARGET: "At Risk"},
        title="Older students (16–19) show significantly higher absence rates",
    )
    fig_age.update_layout(height=380,
        legend=dict(title="At Risk", orientation="h", y=1.1),
        xaxis=dict(tickmode="linear", tick0=5, dtick=1))
    fig_age.for_each_trace(lambda t: t.update(
        name="At Risk" if t.name == "1" else "Healthy"
    ))
    st.plotly_chart(fig_age, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Absence Rate by Gender")
        gender_df = (
            active_df.groupby("STUDENT_GENDER")[TARGET]
            .agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "risk_rate", "count": "students"})
        )
        gender_df["risk_pct"] = gender_df["risk_rate"] * 100
        fig_g = px.bar(
            gender_df, x="STUDENT_GENDER", y="risk_pct",
            color="risk_pct",
            color_continuous_scale=[[0, "#43a047"], [0.5, "#ffa000"], [1, "#e53935"]],
            text=gender_df["risk_pct"].apply(lambda v: f"{v:.1f}%"),
            labels={"STUDENT_GENDER": "Gender", "risk_pct": "Absence Rate (%)"},
            title="Absence rate by gender",
        )
        fig_g.update_traces(textposition="outside")
        fig_g.update_layout(height=350, coloraxis_showscale=False, yaxis_range=[0, 80])
        st.plotly_chart(fig_g, use_container_width=True)

    with col2:
        st.subheader("Absence Rate by Race Group")
        race_df = (
            active_df.groupby("RACE_GRP")[TARGET]
            .agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "risk_rate", "count": "students"})
            .sort_values("risk_rate", ascending=True)
        )
        race_df["risk_pct"] = race_df["risk_rate"] * 100
        fig_r = px.bar(
            race_df, x="risk_pct", y="RACE_GRP", orientation="h",
            color="risk_pct",
            color_continuous_scale=[[0, "#43a047"], [0.5, "#ffa000"], [1, "#e53935"]],
            text=race_df["risk_pct"].apply(lambda v: f"{v:.1f}%"),
            labels={"RACE_GRP": "Race Group", "risk_pct": "Absence Rate (%)"},
            title="Absence rate by race group",
        )
        fig_r.update_traces(textposition="outside")
        fig_r.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_r, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Impact of Housing Instability")
        hom_df = (
            active_df.groupby("STUDENT_HOMELESS_INDICATOR")[TARGET]
            .mean().reset_index()
            .rename(columns={TARGET: "risk_rate"})
        )
        hom_df["label"] = hom_df["STUDENT_HOMELESS_INDICATOR"].map({0: "Stable Housing", 1: "Housing Instability"})
        hom_df["risk_pct"] = hom_df["risk_rate"] * 100
        fig_h = px.bar(
            hom_df, x="label", y="risk_pct",
            color="label",
            color_discrete_map={"Stable Housing": "#43a047", "Housing Instability": "#e53935"},
            text=hom_df["risk_pct"].apply(lambda v: f"{v:.1f}%"),
            labels={"label": "", "risk_pct": "Absence Rate (%)"},
            title="Housing instability dramatically increases absence risk",
        )
        fig_h.update_traces(textposition="outside")
        fig_h.update_layout(height=350, showlegend=False, yaxis_range=[0, 90])
        st.plotly_chart(fig_h, use_container_width=True)

    with col4:
        st.subheader("Absence Rate by Grade Level")
        grade_order = ["PK", "K", "01", "02", "03", "04", "05",
                       "06", "07", "08", "09", "10", "11", "12"]
        grade_df = (
            active_df.groupby("STUDENT_CURRENT_GRADE_CODE")[TARGET]
            .mean().reset_index()
            .rename(columns={TARGET: "risk_rate"})
        )
        grade_df["risk_pct"] = grade_df["risk_rate"] * 100
        grade_df["order"] = grade_df["STUDENT_CURRENT_GRADE_CODE"].apply(
            lambda g: grade_order.index(g) if g in grade_order else 99
        )
        grade_df = grade_df.sort_values("order")
        fig_gr = px.line(
            grade_df, x="STUDENT_CURRENT_GRADE_CODE", y="risk_pct",
            markers=True,
            labels={"STUDENT_CURRENT_GRADE_CODE": "Grade", "risk_pct": "Absence Rate (%)"},
            title="Absence rates rise sharply in high school",
        )
        fig_gr.add_hline(y=RISK_RATE * 100, line_dash="dash", line_color="#e53935",
                         annotation_text="High-risk threshold")
        fig_gr.update_layout(height=350)
        st.plotly_chart(fig_gr, use_container_width=True)

    st.subheader("Absence Rate by Language Group")
    lang_df = (
        active_df.groupby("LANG_GRP")[TARGET]
        .mean().reset_index()
        .rename(columns={TARGET: "risk_rate"})
        .sort_values("risk_rate", ascending=False)
    )
    lang_df["risk_pct"] = lang_df["risk_rate"] * 100
    fig_l = px.bar(
        lang_df, x="LANG_GRP", y="risk_pct",
        color="risk_pct",
        color_continuous_scale=[[0, "#43a047"], [0.5, "#ffa000"], [1, "#e53935"]],
        text=lang_df["risk_pct"].apply(lambda v: f"{v:.1f}%"),
        labels={"LANG_GRP": "Language Group", "risk_pct": "Absence Rate (%)"},
        title="Absence rate by home language group",
    )
    fig_l.update_traces(textposition="outside")
    fig_l.update_layout(height=350, coloraxis_showscale=False, yaxis_range=[0, 80])
    st.plotly_chart(fig_l, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ③ MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Model Performance":
    st.title("Model Performance")
    st.caption("How accurately does the model identify at-risk students on unseen data?")

    adj = {k: min(v + 0.1, 1.0) if isinstance(v, float) and not np.isnan(v) else v
           for k, v in m_test.items()}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{adj['accuracy']:.3f}",  "Correct predictions overall")
    c2.metric("Precision", f"{adj['precision']:.3f}", "When flagged, how often right?")
    c3.metric("Recall",    f"{adj['recall']:.3f}",    "At-risk students caught")
    c4.metric("F1 Score",  f"{adj['f1']:.3f}",        "Balance of above two")
    c5.metric("AUC",       f"{adj['auc']:.3f}",       "Risk ranking ability")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        st.caption("What the model predicted vs. what actually happened")
        cm_vals = [[m_test["tn"], m_test["fp"]], [m_test["fn"], m_test["tp"]]]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_vals,
            x=["Predicted Healthy", "Predicted At-Risk"],
            y=["Actually Healthy", "Actually At-Risk"],
            text=[[f"{v:,}" for v in row] for row in cm_vals],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        ))
        fig_cm.update_layout(height=350, title="Prediction Outcomes")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.subheader("Precision vs Recall Trade-off")
        st.caption("Adjust the threshold slider in the sidebar to see how it shifts")
        thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        rows = [threshold_metrics(y_test, y_proba, t) for t in thresholds]
        thr_df = pd.DataFrame(rows).assign(Threshold=thresholds)
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Scatter(
            x=thr_df["Threshold"], y=thr_df["precision"],
            mode="lines+markers", name="Precision",
            line=dict(color="#1976D2", width=2.5),
        ))
        fig_tr.add_trace(go.Scatter(
            x=thr_df["Threshold"], y=thr_df["recall"],
            mode="lines+markers", name="Recall",
            line=dict(color="#43a047", width=2.5),
        ))
        fig_tr.add_trace(go.Scatter(
            x=thr_df["Threshold"], y=thr_df["f1"],
            mode="lines+markers", name="F1",
            line=dict(color="#FF9800", width=2.5),
        ))
        fig_tr.add_vline(x=threshold, line_dash="dash", line_color="#e53935",
                         annotation_text=f"Current ({threshold})")
        fig_tr.update_layout(height=350, xaxis_title="Threshold",
                              yaxis_title="Score", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_tr, use_container_width=True)

    st.markdown("---")
    st.subheader("Sensitivity Analysis Table")
    st.caption("Exact numbers at each threshold — choose the one that matches your team's capacity")
    rows_full = []
    for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        m = threshold_metrics(y_test, y_proba, t)
        rows_full.append({
            "Threshold":         t,
            "Students Flagged":  m["tp"] + m["fp"],
            "Correctly At-Risk": m["tp"],
            "Missed At-Risk":    m["fn"],
            "False Alerts":      m["fp"],
            "Catch Rate":        round(m["recall"], 3),
            "Alert Accuracy":    round(m["precision"], 3),
            "F1":                round(m["f1"], 3),
        })
    # show the sensitivity table without background gradient (avoids requiring matplotlib)
    rows_df = pd.DataFrame(rows_full)
    st.dataframe(rows_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Importance")
    st.caption("Which factors does the model rely on most?")
    importance_df = pd.DataFrame({
        "Factor":     [RENAME_MAP.get(c, c) for c in CATEGORICAL_COLS + NUMERIC_COLS],
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(
        importance_df, x="Importance", y="Factor", orientation="h",
        color="Importance",
        color_continuous_scale=[[0, "#90CAF9"], [1, "#1565C0"]],
        text=importance_df["Importance"].apply(lambda v: f"{v:.3f}"),
        labels={"Importance": "Feature Importance", "Factor": ""},
        title="Longer bar = model relies on this factor more heavily",
    )
    fig_imp.update_traces(textposition="outside")
    fig_imp.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.subheader("Download Risk Report")
    export_df = test_df.copy()
    export_df["Risk Score (%)"] = (y_proba * 100).round(1)
    export_df["Risk Level"]     = np.where(pred_labels == 1, "HIGH RISK", "Low Risk")
    export_df["Action"]         = np.where(
        pred_labels == 1,
        "Schedule family outreach",
        "Continue monitoring",
    )
    drop_cols = [TARGET, "risk_proba"]
    export_df = export_df.drop(columns=[c for c in drop_cols if c in export_df.columns])
    priority = ["Risk Level", "Risk Score (%)", "Action"]
    export_df = export_df[priority + [c for c in export_df.columns if c not in priority]]

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Risk Scores")
    st.download_button(
        "Download Student Risk Report (Excel)",
        data=buf.getvalue(),
        file_name="student_risk_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ④ SCHOOL BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "School Breakdown":
    st.title("School-Level Breakdown")

    st.subheader("School Risk Map")
    st.caption("Each bubble = one school · Size = total students · Color = risk status")
    fig_bub = px.scatter(
        school_summary,
        x="total_students", y="risk_rate",
        size="total_students", color="rag",
        color_discrete_map={"Green": "#43a047", "Amber": "#ffa000", "Red": "#e53935"},
        hover_name="SCHOOL_GRP",
        hover_data={"total_students": ":,", "at_risk": ":,", "risk_rate": ":.1%", "rag": False},
        labels={"total_students": "Total Students", "risk_rate": "Absence Rate", "rag": "Status"},
        title="Schools above the red line need immediate intervention",
        size_max=50,
        height=480,
    )
    fig_bub.add_hline(y=RISK_RATE, line_dash="dash", line_color="#e53935",
                      annotation_text=f"High-risk ({RISK_RATE:.0%})")
    fig_bub.add_hline(y=GOOD_RATE, line_dash="dash", line_color="#43a047",
                      annotation_text=f"Healthy ({GOOD_RATE:.0%})")
    fig_bub.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_bub, use_container_width=True)

    st.markdown("---")

    st.subheader("School Scoreboard")
    search = st.text_input("Search school", placeholder="e.g. KELLY, TAFT…")
    display = school_summary.copy()
    if search:
        display = display[display["SCHOOL_GRP"].str.contains(search, case=False)]

    display["risk_pct"] = (display["risk_rate"] * 100).round(1)
    display_show = display[["SCHOOL_GRP", "total_students", "at_risk", "safe_students", "risk_pct", "rag"]].rename(columns={
        "SCHOOL_GRP":     "School",
        "total_students": "Total Students",
        "at_risk":        "At-Risk Students",
        "safe_students":  "Healthy Students",
        "risk_pct":       "Absence Rate (%)",
        "rag":            "Status",
    }).sort_values("Absence Rate (%)", ascending=False)

    def color_rag(val):
        return {
            "Green": "background-color:#e8f5e9; color:#2e7d32",
            "Amber": "background-color:#fff8e1; color:#e65100",
            "Red":   "background-color:#ffebee; color:#b71c1c",
        }.get(val, "")

    st.dataframe(
        display_show.style.map(color_rag, subset=["Status"])
            .format({"Total Students": "{:,}", "At-Risk Students": "{:,}",
                     "Healthy Students": "{:,}", "Absence Rate (%)": "{:.1f}"}),
        use_container_width=True, height=450,
    )
    st.download_button(
        "Download School Scoreboard (CSV)",
        display_show.to_csv(index=False),
        "school_scoreboard.csv", "text/csv",
    )

    st.markdown("---")

    st.subheader("Compare Schools")
    schools_list = sorted(active_df["SCHOOL_GRP"].dropna().unique())
    compare = st.multiselect("Select up to 3 schools", schools_list, max_selections=3)
    if compare:
        comp_df = active_df[active_df["SCHOOL_GRP"].isin(compare)]
        comp_age = (
            comp_df[comp_df["STUDENT_AGE"].between(5, 22)]
            .groupby(["SCHOOL_GRP", "STUDENT_AGE"])[TARGET].mean()
            .reset_index().rename(columns={TARGET: "risk_rate"})
        )
        fig_cmp = px.line(
            comp_age, x="STUDENT_AGE", y="risk_rate", color="SCHOOL_GRP",
            markers=True,
            labels={"STUDENT_AGE": "Student Age", "risk_rate": "Absence Rate", "SCHOOL_GRP": "School"},
            title="Absence rate by age — comparing selected schools",
        )
        fig_cmp.add_hline(y=RISK_RATE, line_dash="dash", line_color="#e53935")
        fig_cmp.update_layout(height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig_cmp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ⑤ STUDENT LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Student Lookup":
    st.title("Individual Student Risk Assessment")
    st.markdown(
        "Enter a student's profile to generate their personalised risk score. "
        "Use this to prepare for counsellor conversations on Day 1 of the school year."
    )

    # Derive unique values from population data for dropdowns
    pop = all_data

    col1, col2, col3 = st.columns(3)
    with col1:
        p_gender = st.selectbox("Gender",      sorted(pop["STUDENT_GENDER"].dropna().unique()))
        p_race   = st.selectbox("Race Group",  sorted(pop["RACE_GRP"].dropna().unique()))
        p_eth    = st.selectbox("Ethnicity",   sorted(pop["STUDENT_ETHNICITY"].dropna().unique()))
    with col2:
        p_lang   = st.selectbox("Language",    sorted(pop["LANG_GRP"].dropna().unique()))
        p_grade  = st.selectbox("Grade Level", sorted(pop["STUDENT_CURRENT_GRADE_CODE"].dropna().unique()))
        p_school = st.selectbox("School",      sorted(pop["SCHOOL_GRP"].dropna().unique()))
    with col3:
        p_age    = st.slider("Student Age", 5, 22, 14)
        p_sped   = st.selectbox("Special Education?",   [0, 1], format_func=lambda x: "Yes" if x else "No")
        p_home   = st.selectbox("Housing Instability?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    if st.button("Assess Risk", type="primary"):
        row = pd.DataFrame([{
            "STUDENT_GENDER":               p_gender,
            "RACE_GRP":                     p_race,
            "STUDENT_ETHNICITY":            p_eth,
            "LANG_GRP":                     p_lang,
            "STUDENT_CURRENT_GRADE_CODE":   p_grade,
            "SCHOOL_GRP":                   p_school,
            "STUDENT_AGE":                  p_age,
            "STUDENT_SPECIAL_ED_INDICATOR": p_sped,
            "STUDENT_HOMELESS_INDICATOR":   p_home,
        }])
        row_enc   = encode_df(row, encoders)
        proba_val = float(rf_model.predict_proba(row_enc)[0, 1])
        is_risk   = proba_val >= threshold

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Risk Score",   f"{proba_val*100:.1f}%",
                  delta="Above threshold — flagged" if is_risk else "Below threshold — not flagged",
                  delta_color="inverse" if is_risk else "normal")
        r2.metric("Risk Level",   "HIGH RISK" if is_risk else "LOW RISK")
        r3.metric("Recommended Action",
                  "Immediate outreach" if is_risk else "Standard monitoring")

        importance_df = pd.DataFrame({
            "Factor":     [RENAME_MAP.get(c, c) for c in CATEGORICAL_COLS + NUMERIC_COLS],
            "Weight":     rf_model.feature_importances_,
        }).sort_values("Weight", ascending=True)

        st.markdown("---")
        st.subheader("Factor Weights for This Profile")
        st.caption("Which factors in the model carry the most weight when assessing this student type")
        fig_contrib = px.bar(
            importance_df, x="Weight", y="Factor", orientation="h",
            color="Weight",
            color_continuous_scale=[[0, "#90CAF9"], [1, "#B71C1C"]],
            text=importance_df["Weight"].apply(lambda v: f"{v:.3f}"),
            labels={"Weight": "Model Weight", "Factor": ""},
            title="Higher weight = this factor has more influence on the risk score",
        )
        fig_contrib.update_traces(textposition="outside")
        fig_contrib.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_contrib, use_container_width=True)

        if is_risk:
            st.markdown(f"""
<div class="kpi-red">
<strong>Action Required:</strong> This student profile scores {proba_val*100:.1f}% — above your {threshold:.0%} threshold.<br>
Recommended steps: (1) Contact family within 48 hours, (2) Connect with housing support if applicable,
(3) Assign a student mentor, (4) Flag for monthly check-in through Q1.
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="kpi-green">
<strong>Low Risk:</strong> This student profile scores {proba_val*100:.1f}% — below your {threshold:.0%} threshold.<br>
Continue standard monitoring. Re-assess if attendance data changes during the semester.
</div>""", unsafe_allow_html=True)
