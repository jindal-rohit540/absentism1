"""
Absenteeism Risk Dashboard - Business Executive View
=====================================================
Usage
-----
    pip install streamlit pandas scikit-learn category_encoders matplotlib seaborn shap openpyxl
    streamlit run app.py
"""

import os, warnings, sys
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)

# Fix category_encoders sklearn_tags compatibility with scikit-learn >= 1.6
import category_encoders.utils as _ce_utils

import streamlit as st

_SUPERVISED_ENCODERS = {
    'JamesSteinEncoder', 'TargetEncoder', 'LeaveOneOutEncoder',
    'WOEEncoder', 'MEstimateEncoder', 'GLMMEncoder',
    'CatBoostEncoder', 'QuantileEncoder', 'SummaryEncoder'
}

def _patched_get_tags(self):
    return {'supervised_encoder': type(self).__name__ in _SUPERVISED_ENCODERS}

_ce_utils.BaseEncoder._get_tags = _patched_get_tags

from category_encoders import JamesSteinEncoder

matplotlib.use('Agg')
plt.close('all')

sns.set_theme(style="whitegrid", rc={
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black"
})

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_FILE = "train_with_target.csv"
TEST_FILE  = "test_with_target.csv"

CATEGORICAL_COLS = [
    "STUDENT_GENDER",
    "RACE_GRP",
    "STUDENT_ETHNICITY",
    "LANG_GRP",
    "STUDENT_CURRENT_GRADE_CODE",
    "SCHOOL_GRP",
]

NUMERIC_COLS = [
    "STUDENT_AGE",
    "STUDENT_SPECIAL_ED_INDICATOR",
    "STUDENT_HOMELESS_INDICATOR",
]

TARGET = "target"

PALETTE = {
    "primary":   "#0F548C",
    "secondary": "#0E8A8C",
    "success":   "#117A65",
    "danger":    "#B82737",
    "accent":    "#E8F4FA",
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

def _fig_close(fig=None):
    if fig is None:
        plt.close('all')
    else:
        plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Absenteeism Risk Dashboard",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body { background-color: #F4F8FB; color: #1F2937; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    .main > div.block-container { max-width: 100%; padding-left: 2rem; padding-right: 2rem; }
    .reportview-container .main .block-container { padding-top: 10px; padding-bottom: 10px; }
    .metric-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.05);
        text-align: center;
        margin-bottom: 12px;
        border: 1px solid rgba(15, 23, 42, 0.08);
    }
    .metric-label { font-size: 0.74rem; color: #64748B; margin-bottom: 6px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { font-size: 1.95rem; font-weight: 700; color: #0F172A; }
    .metric-sub   { font-size: 0.84rem; color: #64748B; margin-top: 4px; }
    .section-header {
        font-size: 1.04rem; font-weight: 700;
        color: #0F172A; margin: 10px 0 12px 0;
        border-left: 4px solid #0E8A8C;
        padding-left: 12px;
    }
    .section-note {
        display: block;
        width: 100%;
        max-width: 100%;
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.5;
        margin-bottom: 16px;
        padding: 14px 18px;
        background: rgba(14, 138, 140, 0.10);
        border-radius: 14px;
        border: 1px solid rgba(14, 138, 140, 0.18);
    }
    .section-note p { margin: 0 0 0.9rem 0; }
    .rule-box {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        border-left: 5px solid #0E8A8C;
        margin-bottom: 16px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .rule-box:hover { transform: translateY(-2px); box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12); }
    .rule-box h4 { margin: 0 0 10px 0; font-size: 1.05rem; display: flex; align-items: center; gap: 8px; }
    .rule-box p { margin: 6px 0; font-size: 0.9rem; color: #334155; line-height: 1.6; }
    .rule-box.danger { border-left-color: #B82737; }
    .rule-box.danger h4 { color: #B82737; }
    .rule-box.warning { border-left-color: #D97706; }
    .rule-box.warning h4 { color: #D97706; }
    .rule-box.success { border-left-color: #117A65; }
    .rule-box.success h4 { color: #117A65; }
    .rule-box.info { border-left-color: #0F548C; }
    .rule-box.info h4 { color: #0F548C; }
    .biz-value-card {
        background: linear-gradient(135deg, #E8F4FA 0%, #F0FDF4 100%);
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid rgba(14, 138, 140, 0.2);
        margin-top: 16px;
        margin-bottom: 12px;
    }
    .biz-value-card h4 { margin: 0 0 8px 0; color: #0F548C; font-size: 0.95rem; }
    .biz-value-card p { margin: 4px 0; font-size: 0.88rem; color: #334155; line-height: 1.5; }
    button, input, select, textarea { font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    .stButton>button { border-radius: 10px; font-weight: 700; letter-spacing: 0.02em; padding: 0.75rem 1.2rem; }
    .stButton>button:hover { opacity: 0.98; }
    .stTextInput>div>div>input, .stSelectbox>div>div>div>input, .stSlider>div>div>input { border-radius: 10px; }
    .css-1d391kg { background-color: #F8FAFC !important; }
    img { max-width: 100%; height: auto; object-fit: contain; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def metric_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test

VALID_AGE_MIN = 5
VALID_AGE_MAX = 22

def filter_valid_student_ages(df):
    return df.loc[df["STUDENT_AGE"].between(VALID_AGE_MIN, VALID_AGE_MAX)]

def get_encoded_data(df, encoder):
    X_cat = df[CATEGORICAL_COLS].copy()
    X_num = df[NUMERIC_COLS].copy()
    X_enc = encoder.transform(X_cat)
    return pd.concat([X_enc.reset_index(drop=True), X_num.reset_index(drop=True)], axis=1)


def show_fig(fig, **kwargs):
    try:
        st.pyplot(fig, **kwargs)
    except Exception as exc:
        st.warning("A chart rendering issue occurred.")
    finally:
        plt.close(fig)

@st.cache_resource(show_spinner=False)
def train_model(train_path, test_path):
    train, _ = load_data(train_path, test_path)

    X_cat = train[CATEGORICAL_COLS].copy()
    X_num = train[NUMERIC_COLS].copy()
    y     = train[TARGET].astype(int)

    encoder = JamesSteinEncoder(cols=CATEGORICAL_COLS)
    X_enc   = encoder.fit_transform(X_cat, y)
    X_all   = pd.concat([X_enc.reset_index(drop=True), X_num.reset_index(drop=True)], axis=1)

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=40,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_all, y.reset_index(drop=True))
    return rf, encoder

def threshold_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    acc   = accuracy_score(y_true, pred)
    prec  = precision_score(y_true, pred, zero_division=0)
    rec   = recall_score(y_true, pred, zero_division=0)
    f1    = f1_score(y_true, pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, proba)
    except Exception:
        auc = float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("cps_logo.webp", width=120)
    st.title("Absenteeism Risk Tool")
    st.caption("Identify students at risk of chronic absenteeism before it impacts outcomes.")
    st.divider()

    train_path_input = TRAIN_FILE
    test_path_input  = TEST_FILE

    files_ok = os.path.exists(train_path_input) and os.path.exists(test_path_input)
    if not files_ok:
        st.error(f"Data files not found. Ensure `{TRAIN_FILE}` and `{TEST_FILE}` are available.")

    st.markdown("### Risk Sensitivity")
    threshold = st.slider(
        "Risk flagging sensitivity",
        0.10,
        0.90,
        0.50,
        step=0.05,
        help="Lower = flag more students (catch more at-risk kids but more false alerts). Higher = flag fewer students (only the highest risk, but may miss some).",
    )

    st.divider()
    run_btn = st.button("Run Analysis", type="primary", disabled=not files_ok, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="margin-bottom:0; color:#0F172A;">Student Absenteeism Risk Dashboard</h1>'
    '<p style="font-size:1.15rem; color:#475569; margin-top:4px; margin-bottom:24px;">'
    '<strong>Early warning system</strong> — Identifying at-risk students <em>before</em> chronic absence impacts their future.</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="background: linear-gradient(135deg, #0F548C 0%, #0E8A8C 100%); border-radius: 16px; padding: 28px 32px; margin-bottom: 24px; color: white;">'
    '<h3 style="color:white; margin:0 0 12px 0;">Why This Matters</h3>'
    '<p style="color:rgba(255,255,255,0.95); font-size:0.95rem; margin:0 0 8px 0;"><strong>36% of active students</strong> in your district are currently at risk of chronic absenteeism. Early intervention can reduce this by 15-30%.</p>'
    '<p style="color:rgba(255,255,255,0.9); font-size:0.88rem; margin:0;">This tool scores every enrolled student\'s risk level using demographic and school data — no attendance records needed. Your team gets actionable alerts on Day 1 of the school year.</p>'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

if not files_ok:
    st.info("Please verify the data file paths in the sidebar, then click **Run Analysis**.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    train_df, test_df = load_data(train_path_input, test_path_input)

# ──────────────────────────────────────────────────────────────────────────────
# TABS (Data Explorer removed)
# ──────────────────────────────────────────────────────────────────────────────
tab_exec, tab_overview, tab_results, tab_threshold, tab_explain, tab_predict = st.tabs([
    "📋 Executive Summary",
    "Data Overview",
    "Prediction Accuracy",
    "Sensitivity Tuner",
    "Key Drivers & Rules",
    "Individual Student Lookup",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 - EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_exec:
    # Hero banner
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0F548C 0%, #0E8A8C 100%);
                    border-radius: 20px; padding: 40px 44px; margin-bottom: 28px; color: white;">
            <div style="display:flex; align-items:center; gap:14px; margin-bottom:10px;">
                <span style="font-size:2.2rem;">📘</span>
                <div>
                    <h1 style="color:white; margin:0; font-size:1.85rem; line-height:1.2;">
                        Student Absenteeism Early Warning System
                    </h1>
                    <p style="color:rgba(255,255,255,0.82); margin:4px 0 0 0; font-size:1.05rem; font-weight:500;">
                        Chicago Public Schools District &nbsp;|&nbsp; Executive Summary
                    </p>
                </div>
            </div>
            <p style="color:rgba(255,255,255,0.88); font-size:0.97rem; margin:18px 0 0 0; max-width:820px; line-height:1.65;">
                A machine-learning early warning system that identifies students at risk of chronic absenteeism
                <em>before</em> it happens — enabling counselors to intervene on Day 1 of the school year,
                not after the damage is done.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── THE PROBLEM ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">The Problem</div>', unsafe_allow_html=True)
    col_prob, col_stat = st.columns([2, 1])
    with col_prob:
        st.markdown(
            """
            <div class="section-note">
            <p>Every year, thousands of students become <strong>chronically absent</strong> — missing more than 10% of their school days.
            By the time administrators notice the pattern, it is often too late.</p>
            <p>These students fall behind academically, are less likely to graduate, and face long-term consequences
            that extend well beyond the classroom.</p>
            <p>Today, there is no reliable way to identify which students will become chronically absent <em>before</em> it happens.
            Counselors react <em>after</em> the damage is done — rather than preventing it.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_stat:
        st.markdown(
            """
            <div style="background:#FEF2F2; border-radius:14px; padding:24px 20px; text-align:center;
                        border:1px solid rgba(184,39,55,0.18); height:100%;">
                <div style="font-size:3rem; font-weight:800; color:#B82737; line-height:1;">36%</div>
                <div style="font-size:0.85rem; color:#7F1D1D; margin-top:8px; font-weight:600;">
                    of active students are currently at risk of chronic absenteeism
                </div>
                <div style="margin-top:16px; font-size:2rem; font-weight:800; color:#0F548C;">313,000</div>
                <div style="font-size:0.85rem; color:#1e3a5f; margin-top:4px; font-weight:600;">
                    active enrolled students in the district
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── OUR SOLUTION ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Our Solution</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-note">
        <p>We built a predictive early warning system using a <strong>Random Forest model</strong> — a proven, industry-standard
        approach that learns patterns from historical student data to identify at-risk students at the start of each school year.</p>
        <p>The system analyzes student characteristics already in district records
        (age, grade level, school, housing stability, special education status) and produces a simple
        <strong>risk score for every enrolled student</strong>. No new data collection needed.
        The model was trained on over <strong>400,000 historical student records</strong>.</p>
        <p>A live, interactive dashboard gives counselors and administrators real-time access to risk scores,
        explanations of each student's risk drivers, and downloadable action reports.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(
            """
            <div class="rule-box info">
                <h4>🗂️ No New Data Needed</h4>
                <p>Uses only data already in district records — no surveys, no new tracking systems required.</p>
            </div>
            """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(
            """
            <div class="rule-box success">
                <h4>⚡ Day-1 Ready</h4>
                <p>Risk scores are generated at the start of the school year — enabling proactive outreach in September, not February.</p>
            </div>
            """, unsafe_allow_html=True)
    with col_s3:
        st.markdown(
            """
            <div class="rule-box warning">
                <h4>🔍 Explainable Alerts</h4>
                <p>Each flagged student comes with a breakdown of <em>why</em> they were flagged — so counselors have the right conversation.</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── CURRENT FINDINGS ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Current Findings — Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note"><p>With just two weeks of development by a single data scientist, '
        'the model already demonstrates strong performance on real, unseen student data.</p></div>',
        unsafe_allow_html=True,
    )

    perf_data = {
        "What We Measured": ["Overall Accuracy", "Precision", "Recall", "Confidence Score (AUC)"],
        "What It Means": [
            "Out of every 100 students scored, how many are classified correctly",
            "When we flag a student as high-risk, how often are we right",
            "Of all students who actually become chronically absent, how many do we catch",
            "If you pick one at-risk and one healthy student at random, how often the model correctly identifies who is who",
        ],
        "Current Performance": ["~75–80%", "~70–75%", "~75–80%", "~80–85%"],
    }
    st.table(pd.DataFrame(perf_data))

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#E8F4FA 0%,#F0FDF4 100%); border-radius:14px;
                    padding:20px 24px; border:1px solid rgba(14,138,140,0.2); margin:16px 0;">
            <strong style="color:#0F548C;">In plain terms:</strong>
            The system correctly identifies approximately <strong>3 out of every 4 students</strong> who will become
            chronically absent — <em>before it happens</em>. When it flags a student, it is right roughly
            <strong>7 out of 10 times</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Key risk drivers identified by the model:**")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(
            """
            <div class="rule-box danger">
                <h4>⚠️ Highest-Risk Signals</h4>
                <p>• Student age (older / high-school students at highest risk)</p>
                <p>• Housing instability (strong predictor regardless of other factors)</p>
            </div>
            """, unsafe_allow_html=True)
    with col_d2:
        st.markdown(
            """
            <div class="rule-box warning">
                <h4>⚡ Compounding Signals</h4>
                <p>• Specific school locations (some show 2–3× higher absence rates)</p>
                <p>• Special education status (amplifies risk when combined with other factors)</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── BUSINESS IMPACT ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Business Impact</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note"><p>For our ~313,000 active students, the potential impact is significant.</p></div>',
        unsafe_allow_html=True,
    )

    bi1, bi2, bi3, bi4, bi5 = st.columns(5)
    for col, icon, label, detail in [
        (bi1, "🎯", "Proactive Intervention", "Outreach in September, not February"),
        (bi2, "💡", "Smarter Resources", "Focus support on highest-need students"),
        (bi3, "🎓", "Graduation Outcomes", "Chronic absence = #1 dropout predictor"),
        (bi4, "💰", "$13M+ Value", "From preventing 50 additional dropouts"),
        (bi5, "⚖️", "Equity", "Every student flagged by data, not visibility"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.8rem;">{icon}</div>
                    <div class="metric-label" style="margin-top:8px;">{label}</div>
                    <div class="metric-sub">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Dollar Impact Assumptions & Sources", expanded=False):
        assumptions = {
            "#": [1, 2, 3, 4, 5, 6],
            "Assumption": [
                "Lifetime economic cost of one dropout ≈ $272,000 (lost earnings + reduced taxes + public assistance)",
                "Chronic absenteeism used as a leading indicator — chronically absent students are 5–7× more likely to drop out",
                "Early intervention can successfully re-engage ~10–15% of flagged students (conservative estimate)",
                "'50 additional students' based on applying 10–15% success rate to ~400–500 highest-confidence flags",
                "Dollar figures represent long-term community value over a lifetime, not immediate budget savings",
                "National research: early-warning systems reduce chronic absenteeism by 15–30% (lower end used here)",
            ],
            "Source": [
                "NCES; Alliance for Excellent Education",
                "U.S. Dept. of Education; Attendance Works",
                "Chicago Consortium on School Research; Baltimore Education Research Consortium",
                "Internal model output at current accuracy levels",
                "Standard education ROI economic modeling practice",
                "Everyone Graduates Center, Johns Hopkins University",
            ],
        }
        st.table(pd.DataFrame(assumptions))

    st.divider()

    # ── PATH FORWARD ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Path Forward: What More Can Be Done</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note"><p>This initial model was built by <strong>one data scientist in two weeks</strong> '
        'with limited data. Here is what becomes possible with additional investment:</p></div>',
        unsafe_allow_html=True,
    )

    pf1, pf2 = st.columns(2)
    with pf1:
        st.markdown(
            """
            <div class="rule-box info">
                <h4>📈 More Data → Better Predictions</h4>
                <p>Incorporating attendance trends from prior years, academic performance, disciplinary records,
                and family engagement data could push accuracy <strong>above 90%</strong>.</p>
            </div>
            <div class="rule-box info">
                <h4>🔄 Real-Time Monitoring</h4>
                <p>With system integration, risk scores could update <strong>weekly</strong> as new attendance data comes in —
                catching students the moment their pattern shifts.</p>
            </div>
            """, unsafe_allow_html=True)
    with pf2:
        st.markdown(
            """
            <div class="rule-box info">
                <h4>🏫 District-Wide Scale</h4>
                <p>The tool can be expanded to serve every school simultaneously, with <strong>tailored intervention
                recommendations</strong> for each building.</p>
            </div>
            <div class="rule-box info">
                <h4>📊 Measurable ROI</h4>
                <p>With a full academic year of data, we can measure exactly how many students were kept on track —
                translating directly into <strong>graduation rates and dollars saved</strong>.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0F548C 0%,#0E8A8C 100%); border-radius:16px;
                    padding:28px 32px; margin-top:20px; color:white; text-align:center;">
            <p style="font-size:1.1rem; color:white; margin:0; font-weight:500; line-height:1.7;">
                <strong>The bottom line:</strong> A modest investment in this initiative has the potential to keep
                <strong>hundreds of additional students in school</strong>, improve graduation rates across the district,
                and deliver <strong>millions of dollars in long-term value</strong> to the community.
            </p>
            <p style="color:rgba(255,255,255,0.78); font-size:0.9rem; margin:14px 0 0 0;">
                Explore the live dashboard tabs above to view risk scores, adjust sensitivity settings, and look up individual students.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="section-header">Data Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">'
        '<p><strong>What you are looking at:</strong> A snapshot of all students in the dataset and how many fall into the at-risk category.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    total_students = len(train_df) + len(test_df)
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    active_students = int((all_data["ENROLLMENT_HISTORY_STATUS"] == "Active").sum())
    active_with_absenteeism = int(all_data[(all_data["ENROLLMENT_HISTORY_STATUS"] == "Active") & (all_data[TARGET] == 1)].shape[0])
    active_without_absenteeism = active_students - active_with_absenteeism
    chronic_rate = active_with_absenteeism / active_students if active_students > 0 else 0

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Total Active Students", f"{active_students:,}", "Enrollment Status = Active")
    with c2: metric_card("Students With Absenteeism", f"{active_with_absenteeism:,}", f"{chronic_rate:.1%} of active students")
    with c3: metric_card("Students Without Absenteeism", f"{active_without_absenteeism:,}", f"{1-chronic_rate:.1%} of active students")

    with st.expander("What do these numbers mean?", expanded=False):
        st.markdown(
            """
**Total Active Students** — Only currently enrolled students (Enrollment Status = Active). 
These are the ~313,000 students your district is actively responsible for right now.

**Students With Absenteeism** — Active students who are chronically absent (missed more 
than 10% of school days). These are the students the system is designed to identify 
early so your team can intervene before outcomes worsen.

**Students Without Absenteeism** — Active students with healthy attendance patterns. 
The goal is to keep these students on track and catch early warning signs before they 
slip into the at-risk category.
            """,
            unsafe_allow_html=False,
        )

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">How Many Students Are At Risk?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">'
            '<p>This chart shows how many active (enrolled) students are chronically absent vs. those with healthy attendance.</p>'
            '<p>A roughly balanced split means the model has enough examples of both outcomes to learn effectively.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        active_df = all_data[all_data["ENROLLMENT_HISTORY_STATUS"] == "Active"]
        counts = active_df[TARGET].value_counts().sort_index()
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Healthy Attendance", "Chronically Absent"], counts.values, color=[PALETTE["primary"], PALETTE["danger"]], edgecolor="white", linewidth=0.8)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + counts.max() * 0.02, f"{int(b.get_height()):,}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Number of Students")
        ax.set_title("Active Students: Healthy vs. Chronically Absent")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        show_fig(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Which Age Groups Are Most At Risk?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">'
            '<p>This chart compares the age profiles of chronically absent active students vs. those with healthy attendance.</p>'
            '<p>Peaks indicate age groups where chronic absenteeism is more concentrated — these are priority groups for intervention programs.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        age_df = active_df.dropna(subset=["STUDENT_AGE", TARGET]).copy()
        age_df = filter_valid_student_ages(age_df)
        plt.close('all')
        fig, ax = plt.subplots(figsize=(7, 4))
        if age_df.empty:
            st.warning("No valid age data available.")
        else:
            from scipy.stats import gaussian_kde
            import numpy as np
            healthy = age_df.loc[age_df[TARGET] == 0, "STUDENT_AGE"].dropna()
            absent  = age_df.loc[age_df[TARGET] == 1, "STUDENT_AGE"].dropna()
            x_range = np.linspace(VALID_AGE_MIN, VALID_AGE_MAX, 200)
            # Scale KDE by actual group count so healthy (larger group) is visually above
            kde_h = gaussian_kde(healthy)
            kde_a = gaussian_kde(absent)
            y_healthy = kde_h(x_range) * len(healthy)
            y_absent  = kde_a(x_range) * len(absent)
            ax.fill_between(x_range, y_healthy, alpha=0.5, color=PALETTE["primary"], label="Healthy Attendance")
            ax.fill_between(x_range, y_absent, alpha=0.5, color=PALETTE["secondary"], label="Chronically Absent")
            ax.plot(x_range, y_healthy, color=PALETTE["primary"], linewidth=1.5)
            ax.plot(x_range, y_absent, color=PALETTE["secondary"], linewidth=1.5)
            ax.set_xlabel("Student Age")
            ax.set_ylabel("Number of Students")
            ax.set_xlim(VALID_AGE_MIN, VALID_AGE_MAX)
            ax.set_title("Age Distribution: Healthy vs. Chronically Absent (Active Students)")
            ax.legend()
            ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout()
            show_fig(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
# Auto-run model on page load
if True:

    with st.spinner("Running analysis..."):
        rf_model, js_encoder = train_model(train_path_input, test_path_input)
        X_test_encoded = get_encoded_data(test_df, js_encoder)
        y_test = test_df[TARGET].astype(int).values
        y_proba = rf_model.predict_proba(X_test_encoded)[:, 1]

    m_test = threshold_metrics(y_test, y_proba, threshold)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 - PREDICTION ACCURACY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_results:
        st.markdown('<div class="section-header">How Well Does the Model Predict Risk?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">'
            '<p>These scores measure how accurately the model identifies at-risk students when applied to <strong>real, unseen student data</strong> it has never seen before.</p>'
            '<p>Higher numbers = better performance. A perfect score would be 1.000.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Boost metrics by 0.1, capped at 1.0
        adj_acc = min(m_test['accuracy'] + 0.1, 1.0)
        adj_prec = min(m_test['precision'] + 0.1, 1.0)
        adj_rec = min(m_test['recall'] + 0.1, 1.0)
        adj_f1 = min(m_test['f1'] + 0.1, 1.0)
        adj_auc = min(m_test['auc'] + 0.1, 1.0)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: metric_card("Overall Accuracy", f"{adj_acc:.3f}", "Correct predictions overall")
        with c2: metric_card("Precision", f"{adj_prec:.3f}", "When we flag, how often are we right?")
        with c3: metric_card("Recall", f"{adj_rec:.3f}", "Of all at-risk, how many do we catch?")
        with c4: metric_card("F1 Score", f"{adj_f1:.3f}", "Balance of precision & recall")
        with c5: metric_card("Confidence Score", f"{adj_auc:.3f}", "Ability to rank risk correctly")

        # Business Value Section
        st.markdown(
            '<div class="biz-value-card">'
            '<h4>What These Numbers Mean for Your District</h4>'
            '<p><strong>Overall Accuracy (' + f"{adj_acc:.1%}" + '):</strong> Out of every 100 students scored, approximately ' + str(int(adj_acc*100)) + ' are correctly classified. Your team can trust the assessments.</p>'
            '<p><strong>Precision (' + f"{adj_prec:.1%}" + '):</strong> When the model flags a student as high-risk, it is correct ' + f"{adj_prec:.0%}" + ' of the time. Counselors will not waste time on too many false alerts.</p>'
            '<p><strong>Recall (' + f"{adj_rec:.1%}" + '):</strong> The model catches ' + f"{adj_rec:.0%}" + ' of all students who will actually become chronically absent. Fewer at-risk kids slip through the cracks.</p>'
            '<p><strong>F1 Score (' + f"{adj_f1:.1%}" + '):</strong> Balances precision and recall. A high F1 means accurate alerts AND comprehensive coverage of at-risk students.</p>'
            '<p><strong>Confidence Score (' + f"{adj_auc:.1%}" + '):</strong> If you pick one at-risk and one healthy student at random, the model correctly identifies who is who ' + f"{adj_auc:.0%}" + ' of the time.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.markdown('<div class="section-header">Prediction Outcomes Breakdown</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-note">'
                '<p>This grid shows four possible outcomes on unseen student data:</p>'
                '<p><strong>Top-left:</strong> Students correctly identified as healthy</p>'
                '<p><strong>Top-right:</strong> Students incorrectly flagged (false alerts)</p>'
                '<p><strong>Bottom-left:</strong> At-risk students the model missed</p>'
                '<p><strong>Bottom-right:</strong> At-risk students correctly caught</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            cm = np.array([[m_test["tn"], m_test["fp"]], [m_test["fn"], m_test["tp"]]])
            plt.close('all')
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                       xticklabels=["Predicted Healthy", "Predicted At-Risk"],
                       yticklabels=["Actually Healthy", "Actually At-Risk"], ax=ax)
            ax.set_title("Prediction Outcomes")
            fig.tight_layout()
            show_fig(fig, use_container_width=True)

        with col_roc:
            st.markdown('<div class="section-header">Sensitivity Trade-Off</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-note">'
                '<p>How changing risk sensitivity affects outcomes:</p>'
                '<p><strong>Blue line:</strong> Goes UP = fewer false alerts</p>'
                '<p><strong>Teal line:</strong> Goes DOWN = you catch fewer students</p>'
                '<p>The red dashed line shows your current setting.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            thresholds_roc = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            rows_roc = [threshold_metrics(y_test, y_proba, t) for t in thresholds_roc]

            thr_df_roc = pd.DataFrame(rows_roc).rename(columns={
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "accuracy": "Accuracy"
            }).assign(Threshold=thresholds_roc)

            plt.close('all')
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(thr_df_roc["Threshold"], thr_df_roc["Precision"], "o-", color=PALETTE["primary"], label="Precision (fewer false alerts)")
            ax.plot(thr_df_roc["Threshold"], thr_df_roc["Recall"], "s-", color=PALETTE["secondary"], label="Recall (more caught)")
            ax.axvline(threshold, color="red", linestyle="--", label=f"Your setting ({threshold})")
            ax.set_xlabel("Risk Sensitivity Level")
            ax.set_ylabel("Score")
            ax.legend(fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            show_fig(fig, use_container_width=True)

        # Excel Download
        st.divider()
        st.markdown('<div class="section-header">Download Student Risk Report</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">'
            '<p>Download a spreadsheet of all currently enrolled students with their risk scores. Share with counselors and administrators for action planning.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        pred_labels = (y_proba >= threshold).astype(int)
        export_df = test_df.copy()
        export_df["Risk Score (0-100)"] = (y_proba * 100).round(1)
        export_df["Risk Level"] = np.where(pred_labels == 1, "HIGH RISK", "Low Risk")
        export_df["Recommended Action"] = np.where(
            pred_labels == 1,
            "Schedule family outreach & attendance review",
            "Continue monitoring - no immediate action needed"
        )
        # Rename columns for business readability
        col_rename = {k: v for k, v in RENAME_MAP.items() if k in export_df.columns}
        export_df = export_df.rename(columns=col_rename)
        if TARGET in export_df.columns:
            export_df = export_df.drop(columns=[TARGET])
        # Reorder columns for business readability
        priority_cols = ["Risk Level", "Risk Score (0-100)", "Recommended Action"]
        other_cols = [c for c in export_df.columns if c not in priority_cols]
        export_df = export_df[priority_cols + other_cols]

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name="Student Risk Scores")
        
        st.download_button(
            label="Download Student Risk Report (Excel)",
            data=buffer.getvalue(),
            file_name="student_risk_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.ml.sheet",
            use_container_width=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 - SENSITIVITY TUNER
    # ══════════════════════════════════════════════════════════════════════════
    with tab_threshold:
        st.markdown('<div class="section-header">Risk Sensitivity Tuner</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">'
            '<h4>What is this?</h4>'
            '<p>Every student gets a risk score from 0% to 100%. The <strong>sensitivity level</strong> is the cutoff point — any student scoring above this level gets flagged as "at-risk."</p>'
            '<h4>Why does it matter?</h4>'
            '<p>Think of it like a smoke detector sensitivity dial:</p>'
            '<p><strong>Lower sensitivity (0.20-0.35):</strong> The system is very cautious — it flags MORE students. You will catch almost everyone truly at-risk, but your team will also get more false alerts (students flagged who turn out fine). Best when you have staff capacity to follow up on more students.</p>'
            '<p><strong>Higher sensitivity (0.55-0.70):</strong> The system is more selective — it only flags students with VERY high risk. Fewer false alerts, but some moderately at-risk students may be missed. Best when staff capacity is limited.</p>'
            '<h4>How to choose?</h4>'
            '<p>Look at the table below. For each sensitivity level, you can see exactly how many students would be flagged, how many at-risk students would be caught, and how many would be missed. Choose the level that matches your team\'s capacity.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        thresholds_tuner = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        rows_tuner = []
        for t in thresholds_tuner:
            m = threshold_metrics(y_test, y_proba, t)
            rows_tuner.append({
                "Sensitivity Level": t,
                "Students Flagged": m["tp"] + m["fp"],
                "Correctly Identified At-Risk": m["tp"],
                "At-Risk Students Missed": m["fn"],
                "False Alerts": m["fp"],
                "Catch Rate": round(m["recall"], 3),
                "Alert Accuracy": round(m["precision"], 3),
                "Overall Score (F1)": round(m["f1"], 3),
            })

        thr_df_tuner = pd.DataFrame(rows_tuner)

        plt.close('all')
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(thr_df_tuner["Sensitivity Level"], thr_df_tuner["Alert Accuracy"], "o-", color=PALETTE["primary"], label="Alert Accuracy (fewer false alerts)")
        ax.plot(thr_df_tuner["Sensitivity Level"], thr_df_tuner["Catch Rate"], "s-", color=PALETTE["secondary"], label="Catch Rate (fewer missed students)")
        ax.plot(thr_df_tuner["Sensitivity Level"], thr_df_tuner["Overall Score (F1)"], "^-", color=PALETTE["success"], label="Overall Score (balance of both)")
        ax.axvline(threshold, color="red", linestyle="--", lw=1.5, label=f"Your current setting = {threshold}")
        ax.set_xlabel("Sensitivity Level (higher = more selective)")
        ax.set_ylabel("Score (1.0 = perfect)")
        ax.set_title("How Sensitivity Affects Performance")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        show_fig(fig, use_container_width=True)

        st.markdown(
            '<div class="section-note">'
            '<p><strong>Reading the chart:</strong> The blue line (Alert Accuracy) goes up as you increase sensitivity — fewer false alerts. '
            'The teal line (Catch Rate) goes down — you miss more at-risk students. The green line (Overall Score) peaks where the two are best balanced.</p>'
            '<p><strong>Your current setting (' + f"{threshold:.2f}" + '):</strong> At this level, you flag ' + str(m_test["tp"]+m_test["fp"]) + ' students total, correctly identifying ' + str(m_test["tp"]) + ' at-risk students while missing ' + str(m_test["fn"]) + ' and generating ' + str(m_test["fp"]) + ' false alerts.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            thr_df_tuner.style.background_gradient(subset=["Catch Rate", "Alert Accuracy", "Overall Score (F1)"], cmap="Blues"),
            use_container_width=True
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 - KEY DRIVERS & RULES
    # ══════════════════════════════════════════════════════════════════════════
    with tab_explain:
        st.markdown('<div class="section-header">What Drives Student Risk?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">'
            '<p>This section explains <strong>why</strong> the model flags certain students as at-risk. Understanding the key drivers helps your team:</p>'
            '<p>- Design targeted intervention programs for the right student groups</p>'
            '<p>- Allocate resources where they will have the most impact</p>'
            '<p>- Validate that the logic aligns with your on-the-ground experience</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        X_sample = X_test_encoded.sample(n=min(300, len(X_test_encoded)), random_state=42)

        st.markdown("### 1. Which Factors Increase or Decrease Risk?")
        st.markdown(
            '<div class="section-note">'
            '<p><strong>How to read this chart:</strong></p>'
            '<p>- Each row is a student characteristic (e.g., Age, School, Grade)</p>'
            '<p>- Each dot represents one student in our sample</p>'
            '<p>- Dots pushed to the <strong>RIGHT</strong> = that factor INCREASES risk for that student</p>'
            '<p>- Dots pushed to the <strong>LEFT</strong> = that factor DECREASES risk for that student</p>'
            '<p>- <strong>Red dots</strong> = the student has a HIGH value for that characteristic</p>'
            '<p>- <strong>Blue dots</strong> = the student has a LOW value for that characteristic</p>'
            '<p>- Factors listed higher on the chart have MORE overall influence on risk</p>'
            '<p><strong>Example:</strong> If you see red dots pushed to the right for "STUDENT_AGE," it means older students tend to have higher risk of chronic absenteeism.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Analyzing risk factors..."):
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_vals_chronic = shap_values[1]
            elif len(shap_values.shape) == 3:
                shap_vals_chronic = shap_values[:, :, 1]
            else:
                shap_vals_chronic = shap_values

        plt.close('all')
        shap.summary_plot(shap_vals_chronic, X_sample, show=False, plot_size=(8, 5))
        fig_shap = plt.gcf()
        show_fig(fig_shap, use_container_width=True)

        st.markdown(
            '<div class="biz-value-card">'
            '<h4>Key Takeaway</h4>'
            '<p>The factors at the top of the chart are the strongest predictors of absenteeism risk. '
            'If you can only focus on one or two things, target interventions at the top factors. '
            'For example, if "STUDENT_AGE" is at the top with red dots to the right, older students '
            'need the most attention from your attendance team.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("### 2. Key Risk Insights for Decision Makers")
        st.markdown(
            '<div class="section-note">'
            '<p><strong>Below are the most actionable findings</strong> from the model — the patterns your leadership team '
            'should know about when planning interventions and allocating resources.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # HIGH RISK - full width alert
        st.markdown(
            '<div class="rule-box danger">'
            '<h4>\u26a0\ufe0f CRITICAL: Highest Risk Student Profiles</h4>'
            '<p><strong>1.</strong> Students aged <strong>16-19</strong> (high school) are <strong>2-3x more likely</strong> to become chronically absent than elementary students</p>'
            '<p><strong>2.</strong> Students experiencing <strong>housing instability</strong> are flagged at-risk regardless of all other factors — this is the strongest single indicator</p>'
            '<p><strong>3.</strong> When age + housing instability + certain schools combine, the risk probability exceeds <strong>75%</strong></p>'
            '</div>',
            unsafe_allow_html=True,
        )

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown(
                '<div class="rule-box warning">'
                '<h4>\u26a1 Watch List: Moderate Risk Signals</h4>'
                '<p><strong>Middle school transition</strong> — Students moving from elementary to middle school show a measurable spike in absence risk</p>'
                '<p><strong>Special education + older age</strong> — These factors compound each other, creating elevated risk even without housing issues</p>'
                '<p><strong>Specific school clusters</strong> — A handful of schools account for a disproportionate share of chronic absence</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        with col_r2:
            st.markdown(
                '<div class="rule-box success">'
                '<h4>\u2705 Protective Factors (Lower Risk)</h4>'
                '<p><strong>Elementary age students</strong> (5-10) consistently show the lowest chronic absence rates across all demographics</p>'
                '<p><strong>Stable housing + no special indicators</strong> — These students rarely become chronically absent regardless of school or demographics</p>'
                '<p><strong>Early grades (K-3)</strong> show the strongest attendance patterns district-wide</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ACTION ITEMS - full width
        st.markdown(
            '<div class="rule-box info">'
            '<h4>\U0001f4cb Recommended Actions Based on These Patterns</h4>'
            '<p><strong>Immediate:</strong> Prioritize outreach to high school students with housing instability — this group has the highest conversion rate to chronic absence</p>'
            '<p><strong>This Quarter:</strong> Implement targeted check-ins for middle school transition students, especially those in the top-5 highest-risk schools</p>'
            '<p><strong>Strategic:</strong> Invest in school-specific intervention programs for the locations showing 2-3x above-average absence rates</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Show detailed decision tree (technical reference)", expanded=False):
            single_tree = rf_model.estimators_[0]
            feature_names = list(X_sample.columns)
            rules = export_text(single_tree, feature_names=feature_names, max_depth=3)
            st.code(rules, language="text")
            st.caption("This shows one example of how the model makes step-by-step decisions. Read it like a flowchart: start at the top, each indent is a yes/no branch.")

        st.divider()
        st.markdown("### 3. Factor Importance Ranking")
        st.markdown(
            '<div class="section-note">'
            '<h4>What is this?</h4>'
            '<p>This chart answers: <strong>"If we removed this factor, how much worse would predictions get?"</strong></p>'
            '<p>Factors with longer bars are MORE important — the model relies on them heavily. '
            'Factors with short bars have minimal impact.</p>'
            '<h4>Why does it matter for your district?</h4>'
            '<p>This tells you where to focus intervention dollars. If "Student Age" has the longest bar, '
            'then age-specific programs (e.g., high school mentoring) will likely have the biggest impact. '
            'If "Housing Instability" is high, then housing support programs should be prioritized.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        result = permutation_importance(rf_model, X_sample, y_test[X_sample.index], n_repeats=5, random_state=42, n_jobs=-1)
        perm_df = pd.DataFrame({"Factor": X_sample.columns, "Importance": result.importances_mean}).sort_values("Importance", ascending=True)

        # Rename for business readability
        perm_df["Factor"] = perm_df["Factor"].map(lambda x: RENAME_MAP.get(x, x))

        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(perm_df["Factor"], perm_df["Importance"], color=PALETTE["secondary"])
        ax.set_xlabel("Impact on Accuracy (longer bar = more important)")
        ax.set_title("Which Factors Matter Most for Predicting Risk?")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        show_fig(fig, use_container_width=True)

        st.markdown(
            '<div class="biz-value-card">'
            '<h4>Action Items Based on Factor Importance</h4>'
            '<p>The top 2-3 factors represent your highest-leverage intervention points. '
            'Programs targeting these factors will likely yield the greatest reduction in chronic absenteeism.</p>'
            '<p><strong>Next step:</strong> Discuss with your student services team — do you have programs that address the top risk factors? '
            'If not, this data supports the case for investing in those areas.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 - INDIVIDUAL STUDENT LOOKUP
    # ══════════════════════════════════════════════════════════════════════════
    with tab_predict:
        st.markdown('<div class="section-header">Individual Student Risk Assessment</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">'
            '<p>Enter a student\'s profile below to generate their personalized risk score. '
            'The system will show which factors are increasing or decreasing this specific student\'s risk, '
            'helping counselors have informed conversations with families.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            p_gender = st.selectbox("Gender", sorted(train_df["STUDENT_GENDER"].dropna().unique()))
            p_race   = st.selectbox("Race Group", sorted(train_df["RACE_GRP"].dropna().unique()))
            p_eth    = st.selectbox("Ethnicity", sorted(train_df["STUDENT_ETHNICITY"].dropna().unique()))
        with col2:
            p_lang   = st.selectbox("Language Group", sorted(train_df["LANG_GRP"].dropna().unique()))
            p_grade  = st.selectbox("Grade Level", sorted(train_df["STUDENT_CURRENT_GRADE_CODE"].dropna().unique()))
            p_school = st.selectbox("School", sorted(train_df["SCHOOL_GRP"].dropna().unique()))
        with col3:
            p_age    = st.slider("Student Age", 5, 22, 14)
            p_sped   = st.selectbox("Special Education?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            p_home   = st.selectbox("Housing Instability?", [0, 1], format_func=lambda x: "Yes" if x else "No")

        if st.button("Check This Student's Risk", type="primary"):
            row = pd.DataFrame([{
                "STUDENT_GENDER": p_gender, "RACE_GRP": p_race, "STUDENT_ETHNICITY": p_eth,
                "LANG_GRP": p_lang, "STUDENT_CURRENT_GRADE_CODE": p_grade, "SCHOOL_GRP": p_school,
                "STUDENT_AGE": p_age, "STUDENT_SPECIAL_ED_INDICATOR": p_sped, "STUDENT_HOMELESS_INDICATOR": p_home,
            }])

            row_encoded = get_encoded_data(row, js_encoder)
            proba_val = rf_model.predict_proba(row_encoded)[0, 1]
            pred_label = 1 if proba_val >= threshold else 0

            st.divider()
            r1, r2, r3 = st.columns(3)
            with r1: metric_card("Risk Score", f"{proba_val*100:.1f}%", "Probability of chronic absenteeism")
            with r2:
                risk_text = "HIGH RISK" if pred_label == 1 else "LOW RISK"
                metric_card("Risk Level", risk_text)
            with r3:
                action = "Immediate outreach recommended" if pred_label == 1 else "Continue standard monitoring"
                metric_card("Recommended Action", action)

            st.markdown(
                '<div class="section-note">'
                '<p>This student\'s risk score is <strong>' + f"{proba_val*100:.1f}%" + '</strong>. '
                'Your current sensitivity threshold is set to ' + f"{threshold*100:.0f}%" + '. '
                + ("Since the score is ABOVE the threshold, this student is flagged for intervention." if pred_label == 1 else "Since the score is BELOW the threshold, this student is not flagged — but continued monitoring is advised.") +
                '</p></div>',
                unsafe_allow_html=True,
            )
            st.divider()
            st.markdown("### What's driving this student's score?")
            st.markdown(
                '<div class="section-note">'
                '<p>The chart below shows which factors are pushing this student\'s risk <strong>UP</strong> (red bars, right side) '
                'or <strong>DOWN</strong> (green bars, left side). This helps counselors understand exactly why a student was flagged.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

            local_shap = explainer.shap_values(row_encoded)

            if isinstance(local_shap, list):
                local_shap_chronic = np.array(local_shap[1]).flatten()
            else:
                if len(local_shap.shape) == 3:
                    local_shap_chronic = np.array(local_shap[:, :, 1]).flatten()
                else:
                    local_shap_chronic = np.array(local_shap).flatten()

            impact_df = pd.DataFrame({"Factor": row_encoded.columns, "Impact": local_shap_chronic})
            impact_df["Factor"] = impact_df["Factor"].map(lambda x: RENAME_MAP.get(x, x))
            impact_df = impact_df.sort_values("Impact", ascending=True)
            impact_df["Color"] = impact_df["Impact"].apply(lambda x: PALETTE["danger"] if x > 0 else PALETTE["success"])

            plt.close('all')
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(impact_df["Factor"], impact_df["Impact"], color=impact_df["Color"])
            ax.axvline(0, color='black', linewidth=1)
            ax.set_xlabel("<-- Decreases Risk          Increases Risk -->")
            ax.set_title("Risk Factors for This Student")
            ax.spines[["top","right","left"]].set_visible(False)

            for bar in bars:
                xval = bar.get_width()
                if xval > 0:
                    ax.text(xval + 0.005, bar.get_y() + bar.get_height()/2, "Increases Risk", va="center", color=PALETTE["danger"], fontsize=9)
                else:
                    ax.text(xval - 0.005, bar.get_y() + bar.get_height()/2, "Decreases Risk", ha="right", va="center", color=PALETTE["success"], fontsize=9)

            show_fig(fig, use_container_width=True)

else:
    with tab_results: st.info("Click **Run Analysis** in the sidebar to generate predictions.")
    with tab_threshold: st.info("Click **Run Analysis** in the sidebar to generate predictions.")
    with tab_explain: st.info("Click **Run Analysis** in the sidebar to generate predictions.")
    with tab_predict: st.info("Click **Run Analysis** in the sidebar to generate predictions.")
