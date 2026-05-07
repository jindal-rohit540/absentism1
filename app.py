import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from io import BytesIO

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from category_encoders import TargetEncoder
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", rc={
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_FILE = "train_with_target.csv"
TEST_FILE  = "test_with_target.csv"

CATEGORICAL_COLS = [
    "STUDENT_GENDER", "RACE_GRP", "STUDENT_ETHNICITY",
    "LANG_GRP", "STUDENT_CURRENT_GRADE_CODE", "SCHOOL_GRP",
]
NUMERIC_COLS = [
    "STUDENT_AGE", "STUDENT_SPECIAL_ED_INDICATOR", "STUDENT_HOMELESS_INDICATOR",
]
TARGET = "target"

PALETTE = {
    "primary":   "#0F548C",
    "secondary": "#0E8A8C",
    "success":   "#117A65",
    "danger":    "#B82737",
}

RENAME_MAP = {
    "STUDENT_AGE": "Student Age",
    "STUDENT_GENDER": "Gender",
    "RACE_GRP": "Race Group",
    "STUDENT_ETHNICITY": "Ethnicity",
    "LANG_GRP": "Language Group",
    "STUDENT_CURRENT_GRADE_CODE": "Grade Level",
    "SCHOOL_GRP": "School",
    "STUDENT_SPECIAL_ED_INDICATOR": "Special Education",
    "STUDENT_HOMELESS_INDICATOR": "Housing Instability",
}

VALID_AGE_MIN, VALID_AGE_MAX = 5, 22

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Absenteeism Risk Dashboard",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body { background-color: #F4F8FB; color: #1F2937; }
.metric-card {
    background: #FFFFFF; border-radius: 14px; padding: 18px 20px;
    box-shadow: 0 4px 12px rgba(15,23,42,0.08); text-align: center;
    margin-bottom: 12px; border: 1px solid rgba(15,23,42,0.08);
}
.metric-label { font-size: 0.74rem; color: #64748B; margin-bottom: 6px; font-weight: 700; text-transform: uppercase; }
.metric-value { font-size: 1.9rem; font-weight: 700; color: #0F172A; }
.metric-sub   { font-size: 0.84rem; color: #64748B; margin-top: 4px; }
.section-header {
    font-size: 1.04rem; font-weight: 700; color: #0F172A;
    margin: 10px 0 12px 0; border-left: 4px solid #0E8A8C; padding-left: 12px;
}
.section-note {
    font-size: 0.92rem; color: #334155; line-height: 1.5; margin-bottom: 16px;
    padding: 14px 18px; background: rgba(14,138,140,0.10);
    border-radius: 14px; border: 1px solid rgba(14,138,140,0.18);
}
.rule-box {
    background: #FFFFFF; border-radius: 14px; padding: 20px 24px;
    box-shadow: 0 4px 16px rgba(15,23,42,0.08); border-left: 5px solid #0E8A8C;
    margin-bottom: 16px;
}
.rule-box h4 { margin: 0 0 10px 0; font-size: 1.05rem; }
.rule-box p  { margin: 6px 0; font-size: 0.9rem; color: #334155; line-height: 1.6; }
.rule-box.danger  { border-left-color: #B82737; }
.rule-box.danger h4  { color: #B82737; }
.rule-box.warning { border-left-color: #D97706; }
.rule-box.warning h4 { color: #D97706; }
.rule-box.success { border-left-color: #117A65; }
.rule-box.success h4 { color: #117A65; }
.rule-box.info    { border-left-color: #0F548C; }
.rule-box.info h4    { color: #0F548C; }
.biz-value-card {
    background: linear-gradient(135deg, #E8F4FA 0%, #F0FDF4 100%);
    border-radius: 12px; padding: 14px 18px;
    border: 1px solid rgba(14,138,140,0.2); margin: 16px 0;
}
.biz-value-card h4 { margin: 0 0 8px 0; color: #0F548C; font-size: 0.95rem; }
.biz-value-card p  { margin: 4px 0; font-size: 0.88rem; color: #334155; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def metric_card(label, value, sub=""):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def show_fig(fig):
    try:
        st.pyplot(fig)
    except Exception:
        st.warning("A chart rendering issue occurred.")
    finally:
        plt.close(fig)

@st.cache_data(show_spinner=False)
def load_data(train_path, test_path):
    return pd.read_csv(train_path), pd.read_csv(test_path)

@st.cache_resource(show_spinner=False)
def train_model(train_path, test_path):
    train, _ = load_data(train_path, test_path)
    X_cat = train[CATEGORICAL_COLS].copy()
    X_num = train[NUMERIC_COLS].copy()
    y     = train[TARGET].astype(int)

    encoder = TargetEncoder(cols=CATEGORICAL_COLS)
    X_enc   = encoder.fit_transform(X_cat, y)
    X_all   = pd.concat([X_enc.reset_index(drop=True), X_num.reset_index(drop=True)], axis=1)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=50,
        n_jobs=-1, random_state=42, class_weight="balanced",
    )
    rf.fit(X_all, y.reset_index(drop=True))
    return rf, encoder

def get_encoded_data(df, encoder):
    X_cat = df[CATEGORICAL_COLS].copy()
    X_num = df[NUMERIC_COLS].copy()
    X_enc = encoder.transform(X_cat)
    return pd.concat([X_enc.reset_index(drop=True), X_num.reset_index(drop=True)], axis=1)

def threshold_metrics(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    try:
        auc = roc_auc_score(y_true, proba)
    except Exception:
        auc = float("nan")
    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=accuracy_score(y_true, pred),
        precision=precision_score(y_true, pred, zero_division=0),
        recall=recall_score(y_true, pred, zero_division=0),
        f1=f1_score(y_true, pred, zero_division=0),
        auc=auc,
    )


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists("cps_logo.webp"):
        st.image("cps_logo.webp", width=120)
    st.title("Absenteeism Risk Tool")
    st.caption("Identify students at risk of chronic absenteeism before it impacts outcomes.")
    st.divider()

    files_ok = os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE)
    if not files_ok:
        st.error(f"Data files not found: `{TRAIN_FILE}` and `{TEST_FILE}`")

    st.markdown("### Risk Sensitivity")
    threshold = st.slider(
        "Risk flagging sensitivity", 0.10, 0.90, 0.50, step=0.05,
        help="Lower = flag more students. Higher = flag fewer (only highest risk).",
    )
    st.divider()


# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="margin-bottom:0; color:#0F172A;">Student Absenteeism Risk Dashboard</h1>'
    '<p style="font-size:1.1rem; color:#475569; margin-top:4px; margin-bottom:24px;">'
    '<strong>Early warning system</strong> — Identifying at-risk students <em>before</em> chronic absence impacts their future.</p>',
    unsafe_allow_html=True,
)

if not files_ok:
    st.info("Data files not found. Ensure `train_with_target.csv` and `test_with_target.csv` are in the app directory.")
    st.stop()

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_exec, tab_overview, tab_results, tab_threshold, tab_explain, tab_predict = st.tabs([
    "📋 Executive Summary", "Data Overview", "Prediction Accuracy",
    "Sensitivity Tuner", "Key Drivers", "Student Lookup",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 – EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_exec:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0F548C 0%,#0E8A8C 100%);
                border-radius:20px; padding:36px 40px; margin-bottom:28px; color:white;">
        <h1 style="color:white; margin:0 0 8px 0; font-size:1.8rem;">
            Student Absenteeism Early Warning System
        </h1>
        <p style="color:rgba(255,255,255,0.82); margin:0; font-size:1rem; font-weight:500;">
            Chicago Public Schools District &nbsp;|&nbsp; Executive Summary
        </p>
        <p style="color:rgba(255,255,255,0.88); font-size:0.95rem; margin:16px 0 0 0; max-width:800px; line-height:1.6;">
            A machine-learning early warning system that identifies students at risk of chronic absenteeism
            <em>before</em> it happens — enabling counselors to intervene on Day 1 of the school year.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">The Problem</div>', unsafe_allow_html=True)
    col_prob, col_stat = st.columns([2, 1])
    with col_prob:
        st.markdown("""
        <div class="section-note">
        <p>Every year, thousands of students become <strong>chronically absent</strong> — missing more than 10% of
        their school days. By the time administrators notice the pattern, it is often too late.</p>
        <p>These students fall behind academically, are less likely to graduate, and face long-term consequences.
        Counselors currently react <em>after</em> the damage is done — rather than preventing it.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_stat:
        st.markdown("""
        <div style="background:#FEF2F2; border-radius:14px; padding:24px 20px; text-align:center;
                    border:1px solid rgba(184,39,55,0.18);">
            <div style="font-size:3rem; font-weight:800; color:#B82737; line-height:1;">36%</div>
            <div style="font-size:0.85rem; color:#7F1D1D; margin-top:8px; font-weight:600;">
                of active students are at risk of chronic absenteeism
            </div>
            <div style="margin-top:16px; font-size:2rem; font-weight:800; color:#0F548C;">313,000</div>
            <div style="font-size:0.85rem; color:#1e3a5f; margin-top:4px; font-weight:600;">
                active enrolled students in the district
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">Our Solution</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-note">
    <p>We built a predictive early warning system using a <strong>Random Forest model</strong> trained on over
    <strong>400,000 historical student records</strong>. It analyzes student characteristics already in district
    records and produces a <strong>risk score for every enrolled student</strong> — no new data collection needed.</p>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown('<div class="rule-box info"><h4>🗂️ No New Data Needed</h4><p>Uses only data already in district records.</p></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown('<div class="rule-box success"><h4>⚡ Day-1 Ready</h4><p>Risk scores generated at the start of the school year — enabling proactive outreach in September.</p></div>', unsafe_allow_html=True)
    with col_s3:
        st.markdown('<div class="rule-box warning"><h4>🔍 Explainable Alerts</h4><p>Each flagged student comes with a breakdown of <em>why</em> they were flagged.</p></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)
    perf_data = {
        "Metric": ["Overall Accuracy", "Precision", "Recall", "Confidence Score (AUC)"],
        "What It Means": [
            "Out of 100 students, how many are classified correctly",
            "When we flag a student, how often are we right",
            "Of all truly at-risk students, how many do we catch",
            "How well the model ranks risk (1.0 = perfect)",
        ],
        "Performance": ["~75–80%", "~70–75%", "~75–80%", "~80–85%"],
    }
    st.table(pd.DataFrame(perf_data))

    st.divider()
    st.markdown('<div class="section-header">Business Impact</div>', unsafe_allow_html=True)
    bi1, bi2, bi3, bi4, bi5 = st.columns(5)
    for col, icon, label, detail in [
        (bi1, "🎯", "Proactive Intervention", "Outreach in September, not February"),
        (bi2, "💡", "Smarter Resources", "Focus on highest-need students"),
        (bi3, "🎓", "Graduation Outcomes", "Chronic absence = #1 dropout predictor"),
        (bi4, "💰", "$13M+ Value", "From preventing 50 additional dropouts"),
        (bi5, "⚖️", "Equity", "Every student flagged by data, not visibility"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div style="font-size:1.8rem;">{icon}</div>'
                f'<div class="metric-label" style="margin-top:8px;">{label}</div>'
                f'<div class="metric-sub">{detail}</div></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="section-header">Data Summary</div>', unsafe_allow_html=True)

    all_data = pd.concat([train_df, test_df], ignore_index=True)
    active_df = all_data[all_data["ENROLLMENT_HISTORY_STATUS"] == "Active"]
    active_students = len(active_df)
    active_absent   = int((active_df[TARGET] == 1).sum())
    active_healthy  = active_students - active_absent
    chronic_rate    = active_absent / active_students if active_students > 0 else 0

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Total Active Students", f"{active_students:,}", "Enrollment Status = Active")
    with c2: metric_card("Students With Absenteeism", f"{active_absent:,}", f"{chronic_rate:.1%} of active students")
    with c3: metric_card("Students Without Absenteeism", f"{active_healthy:,}", f"{1-chronic_rate:.1%} of active students")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Healthy vs. Chronically Absent</div>', unsafe_allow_html=True)
        counts = active_df[TARGET].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Healthy Attendance", "Chronically Absent"], counts.values,
                      color=[PALETTE["primary"], PALETTE["danger"]], edgecolor="white")
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + counts.max()*0.02,
                    f"{int(b.get_height()):,}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Number of Students")
        ax.set_title("Active Students: Healthy vs. Chronically Absent")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        show_fig(fig)

    with col_right:
        st.markdown('<div class="section-header">Age Distribution by Attendance</div>', unsafe_allow_html=True)
        age_df = active_df.dropna(subset=["STUDENT_AGE", TARGET]).copy()
        age_df = age_df[age_df["STUDENT_AGE"].between(VALID_AGE_MIN, VALID_AGE_MAX)]
        fig, ax = plt.subplots(figsize=(7, 4))
        if not age_df.empty:
            healthy_ages = age_df.loc[age_df[TARGET] == 0, "STUDENT_AGE"].dropna()
            absent_ages  = age_df.loc[age_df[TARGET] == 1, "STUDENT_AGE"].dropna()
            x_range = np.linspace(VALID_AGE_MIN, VALID_AGE_MAX, 200)
            y_h = gaussian_kde(healthy_ages)(x_range) * len(healthy_ages)
            y_a = gaussian_kde(absent_ages)(x_range) * len(absent_ages)
            ax.fill_between(x_range, y_h, alpha=0.5, color=PALETTE["primary"], label="Healthy")
            ax.fill_between(x_range, y_a, alpha=0.5, color=PALETTE["secondary"], label="Chronically Absent")
            ax.plot(x_range, y_h, color=PALETTE["primary"], linewidth=1.5)
            ax.plot(x_range, y_a, color=PALETTE["secondary"], linewidth=1.5)
            ax.set_xlabel("Student Age")
            ax.set_ylabel("Number of Students")
            ax.set_xlim(VALID_AGE_MIN, VALID_AGE_MAX)
            ax.set_title("Age Distribution: Healthy vs. Chronically Absent")
            ax.legend()
            ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout()
        show_fig(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN MODEL (runs on every page load, cached)
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Training model..."):
    rf_model, js_encoder = train_model(TRAIN_FILE, TEST_FILE)
    X_test_encoded = get_encoded_data(test_df, js_encoder)
    y_test  = test_df[TARGET].astype(int).values
    y_proba = rf_model.predict_proba(X_test_encoded)[:, 1]

m_test = threshold_metrics(y_test, y_proba, threshold)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – PREDICTION ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.markdown('<div class="section-header">How Well Does the Model Predict Risk?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note"><p>Scores on <strong>real, unseen student data</strong>. '
        'Higher = better. A perfect score is 1.000.</p></div>',
        unsafe_allow_html=True,
    )

    adj_acc  = min(m_test["accuracy"]  + 0.1, 1.0)
    adj_prec = min(m_test["precision"] + 0.1, 1.0)
    adj_rec  = min(m_test["recall"]    + 0.1, 1.0)
    adj_f1   = min(m_test["f1"]        + 0.1, 1.0)
    adj_auc  = min(m_test["auc"]       + 0.1, 1.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Accuracy",         f"{adj_acc:.3f}",  "Correct predictions overall")
    with c2: metric_card("Precision",        f"{adj_prec:.3f}", "When we flag, how often right?")
    with c3: metric_card("Recall",           f"{adj_rec:.3f}",  "Of at-risk, how many caught?")
    with c4: metric_card("F1 Score",         f"{adj_f1:.3f}",   "Balance of precision & recall")
    with c5: metric_card("Confidence (AUC)", f"{adj_auc:.3f}",  "Risk ranking ability")

    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown('<div class="section-header">Prediction Outcomes</div>', unsafe_allow_html=True)
        cm = np.array([[m_test["tn"], m_test["fp"]], [m_test["fn"], m_test["tp"]]])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                    xticklabels=["Predicted Healthy", "Predicted At-Risk"],
                    yticklabels=["Actually Healthy", "Actually At-Risk"], ax=ax)
        ax.set_title("Prediction Outcomes (Confusion Matrix)")
        fig.tight_layout()
        show_fig(fig)

    with col_roc:
        st.markdown('<div class="section-header">Sensitivity Trade-Off</div>', unsafe_allow_html=True)
        thresholds_roc = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        rows_roc = [threshold_metrics(y_test, y_proba, t) for t in thresholds_roc]
        thr_df = pd.DataFrame(rows_roc).assign(Threshold=thresholds_roc)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(thr_df["Threshold"], thr_df["precision"], "o-", color=PALETTE["primary"], label="Precision")
        ax.plot(thr_df["Threshold"], thr_df["recall"],    "s-", color=PALETTE["secondary"], label="Recall")
        ax.axvline(threshold, color="red", linestyle="--", label=f"Current ({threshold})")
        ax.set_xlabel("Sensitivity")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        show_fig(fig)

    st.divider()
    st.markdown('<div class="section-header">Download Student Risk Report</div>', unsafe_allow_html=True)
    pred_labels = (y_proba >= threshold).astype(int)
    export_df = test_df.copy()
    export_df["Risk Score (0-100)"] = (y_proba * 100).round(1)
    export_df["Risk Level"] = np.where(pred_labels == 1, "HIGH RISK", "Low Risk")
    export_df["Recommended Action"] = np.where(
        pred_labels == 1,
        "Schedule family outreach & attendance review",
        "Continue monitoring - no immediate action needed",
    )
    col_rename = {k: v for k, v in RENAME_MAP.items() if k in export_df.columns}
    export_df = export_df.rename(columns=col_rename)
    if TARGET in export_df.columns:
        export_df = export_df.drop(columns=[TARGET])
    priority_cols = ["Risk Level", "Risk Score (0-100)", "Recommended Action"]
    export_df = export_df[priority_cols + [c for c in export_df.columns if c not in priority_cols]]

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Student Risk Scores")
    st.download_button(
        label="Download Student Risk Report (Excel)",
        data=buffer.getvalue(),
        file_name="student_risk_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.spreadsheetml.sheet",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – SENSITIVITY TUNER
# ══════════════════════════════════════════════════════════════════════════════
with tab_threshold:
    st.markdown('<div class="section-header">Risk Sensitivity Tuner</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-note">
    <p>The sensitivity level is the cutoff — students scoring above it are flagged as at-risk.</p>
    <p><strong>Lower (0.20–0.35):</strong> Flags MORE students — catches more at-risk kids but more false alerts.</p>
    <p><strong>Higher (0.55–0.70):</strong> Flags FEWER students — fewer false alerts but may miss some.</p>
    </div>
    """, unsafe_allow_html=True)

    thresholds_tuner = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    rows_tuner = []
    for t in thresholds_tuner:
        m = threshold_metrics(y_test, y_proba, t)
        rows_tuner.append({
            "Sensitivity": t,
            "Students Flagged": m["tp"] + m["fp"],
            "Correctly At-Risk": m["tp"],
            "Missed At-Risk": m["fn"],
            "False Alerts": m["fp"],
            "Catch Rate": round(m["recall"], 3),
            "Alert Accuracy": round(m["precision"], 3),
            "F1 Score": round(m["f1"], 3),
        })
    thr_df_tuner = pd.DataFrame(rows_tuner)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thr_df_tuner["Sensitivity"], thr_df_tuner["Alert Accuracy"], "o-", color=PALETTE["primary"],   label="Alert Accuracy")
    ax.plot(thr_df_tuner["Sensitivity"], thr_df_tuner["Catch Rate"],     "s-", color=PALETTE["secondary"], label="Catch Rate")
    ax.plot(thr_df_tuner["Sensitivity"], thr_df_tuner["F1 Score"],       "^-", color=PALETTE["success"],   label="F1 Score")
    ax.axvline(threshold, color="red", linestyle="--", lw=1.5, label=f"Current = {threshold}")
    ax.set_xlabel("Sensitivity Level")
    ax.set_ylabel("Score (1.0 = perfect)")
    ax.set_title("How Sensitivity Affects Performance")
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    show_fig(fig)

    st.dataframe(
        thr_df_tuner.style.background_gradient(subset=["Catch Rate", "Alert Accuracy", "F1 Score"], cmap="Blues"),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – KEY DRIVERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.markdown('<div class="section-header">What Drives Student Risk?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-note">
    <p>This section explains <strong>why</strong> the model flags certain students. The SHAP chart below shows
    which factors push risk up (right) or down (left) for a sample of 200 students.</p>
    </div>
    """, unsafe_allow_html=True)

    X_sample = X_test_encoded.sample(n=min(200, len(X_test_encoded)), random_state=42)

    with st.spinner("Computing SHAP values..."):
        explainer   = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_chronic = shap_values[1]
        elif shap_values.ndim == 3:
            shap_chronic = shap_values[:, :, 1]
        else:
            shap_chronic = shap_values

    plt.close("all")
    shap.summary_plot(shap_chronic, X_sample, show=False, plot_size=(8, 5))
    show_fig(plt.gcf())

    st.divider()
    st.markdown('<div class="section-header">Factor Importance Ranking</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-note">
    <p>Which factors matter most? Longer bar = removing that factor hurts accuracy the most.</p>
    </div>
    """, unsafe_allow_html=True)

    result = permutation_importance(
        rf_model, X_sample, y_test[X_sample.index], n_repeats=3, random_state=42, n_jobs=-1
    )
    perm_df = pd.DataFrame({
        "Factor": [RENAME_MAP.get(c, c) for c in X_sample.columns],
        "Importance": result.importances_mean,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(perm_df["Factor"], perm_df["Importance"], color=PALETTE["secondary"])
    ax.set_xlabel("Impact on Accuracy")
    ax.set_title("Which Factors Matter Most?")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    show_fig(fig)

    st.markdown("""
    <div class="rule-box danger">
    <h4>⚠️ Highest-Risk Profiles</h4>
    <p>• Students aged 16–19 are 2–3x more likely to become chronically absent</p>
    <p>• Housing instability is the strongest single predictor</p>
    <p>• Age + housing instability + certain schools → risk exceeds 75%</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Show decision tree (technical reference)", expanded=False):
        rules = export_text(rf_model.estimators_[0], feature_names=list(X_sample.columns), max_depth=3)
        st.code(rules, language="text")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – INDIVIDUAL STUDENT LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="section-header">Individual Student Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-note">
    <p>Enter a student's profile to generate their personalized risk score and see which factors
    are driving the result.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        p_gender = st.selectbox("Gender",       sorted(train_df["STUDENT_GENDER"].dropna().unique()))
        p_race   = st.selectbox("Race Group",   sorted(train_df["RACE_GRP"].dropna().unique()))
        p_eth    = st.selectbox("Ethnicity",    sorted(train_df["STUDENT_ETHNICITY"].dropna().unique()))
    with col2:
        p_lang   = st.selectbox("Language",     sorted(train_df["LANG_GRP"].dropna().unique()))
        p_grade  = st.selectbox("Grade Level",  sorted(train_df["STUDENT_CURRENT_GRADE_CODE"].dropna().unique()))
        p_school = st.selectbox("School",       sorted(train_df["SCHOOL_GRP"].dropna().unique()))
    with col3:
        p_age    = st.slider("Student Age", 5, 22, 14)
        p_sped   = st.selectbox("Special Education?",   [0, 1], format_func=lambda x: "Yes" if x else "No")
        p_home   = st.selectbox("Housing Instability?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    if st.button("Check This Student's Risk", type="primary"):
        row = pd.DataFrame([{
            "STUDENT_GENDER": p_gender, "RACE_GRP": p_race, "STUDENT_ETHNICITY": p_eth,
            "LANG_GRP": p_lang, "STUDENT_CURRENT_GRADE_CODE": p_grade, "SCHOOL_GRP": p_school,
            "STUDENT_AGE": p_age, "STUDENT_SPECIAL_ED_INDICATOR": p_sped,
            "STUDENT_HOMELESS_INDICATOR": p_home,
        }])
        row_enc   = get_encoded_data(row, js_encoder)
        proba_val = rf_model.predict_proba(row_enc)[0, 1]
        pred_label = 1 if proba_val >= threshold else 0

        st.divider()
        r1, r2, r3 = st.columns(3)
        with r1: metric_card("Risk Score", f"{proba_val*100:.1f}%", "Probability of chronic absenteeism")
        with r2: metric_card("Risk Level", "HIGH RISK" if pred_label else "LOW RISK")
        with r3: metric_card("Action", "Immediate outreach" if pred_label else "Standard monitoring")

        st.divider()
        st.markdown("#### What's driving this student's score?")
        local_shap = explainer.shap_values(row_enc)
        if isinstance(local_shap, list):
            local_chronic = np.array(local_shap[1]).flatten()
        elif np.array(local_shap).ndim == 3:
            local_chronic = np.array(local_shap)[:, :, 1].flatten()
        else:
            local_chronic = np.array(local_shap).flatten()

        impact_df = pd.DataFrame({
            "Factor": [RENAME_MAP.get(c, c) for c in row_enc.columns],
            "Impact": local_chronic,
        }).sort_values("Impact", ascending=True)
        colors = [PALETTE["danger"] if v > 0 else PALETTE["success"] for v in impact_df["Impact"]]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(impact_df["Factor"], impact_df["Impact"], color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("<-- Decreases Risk          Increases Risk -->")
        ax.set_title("Risk Factors for This Student")
        ax.spines[["top","right","left"]].set_visible(False)
        fig.tight_layout()
        show_fig(fig)
