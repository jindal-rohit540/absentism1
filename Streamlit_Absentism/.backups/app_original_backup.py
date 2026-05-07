"""
Absenteeism Classification – Streamlit App & Explainer
======================================================
Usage
-----
    pip install streamlit pandas scikit-learn category_encoders matplotlib seaborn shap
    streamlit run app.py
"""

import os, warnings
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from category_encoders import JamesSteinEncoder

# 🛑 FIX 1: Nuke Matplotlib Memory Leaks
matplotlib.use('Agg')
plt.close('all')

# 🛑 FIX 2: Force White Backgrounds so Charts never turn black in Dark Mode
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
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body { background-color: #F4F8FB; color: #1F2937; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    .main > div.block-container { max-width: 1080px; margin: auto; }
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
def metric_card(label: str, value, sub: str = ""):
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
def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test

VALID_AGE_MIN = 5
VALID_AGE_MAX = 22

def filter_valid_student_ages(df: pd.DataFrame):
    return df.loc[df["STUDENT_AGE"].between(VALID_AGE_MIN, VALID_AGE_MAX)]

def get_encoded_data(df: pd.DataFrame, encoder):
    X_cat = df[CATEGORICAL_COLS].copy()
    X_num = df[NUMERIC_COLS].copy()
    X_enc = encoder.transform(X_cat)
    return pd.concat([X_enc.reset_index(drop=True), X_num.reset_index(drop=True)], axis=1)


def show_fig(fig, **kwargs):
    try:
        st.pyplot(fig, **kwargs)
    except Exception as exc:
        st.warning("A plot rendering error occurred while updating the page.")
        st.write(f"Plot error: {exc}")
    finally:
        plt.close(fig)

@st.cache_resource(show_spinner=False)
def train_model(train_path: str, test_path: str):
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
    st.image("https://img.icons8.com/ios/80/0f548c/school-building.png", width=60)
    st.title("Absenteeism Classifier")
    st.caption("Predictive model for chronic absenteeism risk assessment.")
    st.divider()

    st.markdown("### Data Files")
    train_path_input = st.text_input("Training dataset", value=TRAIN_FILE)
    test_path_input  = st.text_input("Testing dataset", value=TEST_FILE)

    files_ok = os.path.exists(train_path_input) and os.path.exists(test_path_input)
    if not files_ok:
        st.error(f"CSV files not found. Ensure `{TRAIN_FILE}` and `{TEST_FILE}` are available in the working directory.")

    st.divider()
    st.markdown("### Evaluation Threshold")
    threshold = st.slider(
        "Probability threshold",
        0.10,
        0.90,
        0.50,
        step=0.05,
        help="Higher values prioritize precision, lower values prioritize recall.",
    )

    st.divider()
    run_btn = st.button("Train and Evaluate", type="primary", disabled=not files_ok, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("Student Absenteeism Risk Assessment")
st.markdown("**Random Forest Classifier** for evaluating the likelihood of chronic absenteeism.")

st.markdown(
    '<div class="section-note">'
    '<h4>Model Explanation</h4>'
    '<p><strong>Numbers Quoted:</strong> The metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC) assess how well the model predicts chronic absenteeism (defined as year-to-date absence rate > 10%). These are evaluated on the test dataset at the selected probability threshold.</p>'
    '<p><strong>Tables Used:</strong> Data is sourced from lakehouse tables: <code>attendance_feature_set</code> (attendance metrics), <code>dim_student</code> (demographics), <code>dim_student_detail</code> (additional info), and <code>dim_class</code> (class data).</p>'
    '<p><strong>Student Types:</strong> Training includes all students (enrolled, not enrolled, graduated) for broad pattern learning. Testing is done on currently enrolled students to simulate real-world prediction.</p>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-note">'
    '<p>This executive dashboard provides a concise view of student risk, model performance, and explainability.</p>'
    '<p>Use the tabs to move from data quality to model validation, then to risk segmentation and individual prediction explanations.</p>'
    '</div>',
    unsafe_allow_html=True,
)
st.divider()

if not files_ok:
    st.info("Please verify the CSV file paths in the sidebar, then click **Train and Evaluate**.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA (always)
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading datasets …"):
    train_df, test_df = load_data(train_path_input, test_path_input)

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_overview, tab_results, tab_threshold, tab_explore, tab_explain, tab_predict = st.tabs([
    "Data Overview",
    "Model Results",
    "Threshold Tuner",
    "Data Explorer",
    "Explainability & Rules",
    "Single Prediction",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">'
        '<p>The dataset summary identifies how many samples are available for training and testing.</p>'
        '<p>It also highlights the total number of input features that the model uses to predict chronic absenteeism.</p>'
        '<p>Review this section first to confirm that the data volume and feature coverage are sufficient for reliable modeling.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Train rows",  f"{len(train_df):,}")
    with c2: metric_card("Test rows",   f"{len(test_df):,}")
    with c3: metric_card("Features",    f"{len(CATEGORICAL_COLS)+len(NUMERIC_COLS)}")
    with c4:
        rate = train_df[TARGET].mean()
        metric_card("Chronic rate (train)", f"{rate:.1%}", "ytd_absence_rate > 10%")

    with st.expander("How are these numbers derived?", expanded=False):
        st.markdown(
            """
            **Train Rows**

            The full dataset is sourced from four lakehouse tables:
            `attendance_feature_set` (attendance metrics), `dim_student` (demographics),
            `dim_student_detail` (birth date & parent language), and `dim_class` (school groupings).
            **All students — enrolled, not enrolled, and graduated — are included in the training set.**
            This gives the model the broadest possible population to learn absenteeism patterns from,
            including historical cohorts whose outcomes are already known.
            The split is **not random**; it is determined entirely by enrollment status.

            ---

            **Test Rows**

            Testing is scoped **exclusively to currently enrolled students** (enrollment status filter,
            not a random hold-out). This simulates real-world inference: the model scores only the students
            who are currently attending school. Restricting the test set this way ensures that the
            explainability analysis (SHAP values, permutation importance) reflects the exact population
            the model will be used on in practice.

            ---

            **Features — 9**

            Leakage columns (direct absence totals such as `ytd_absences`, `attendance_value_ytd`,
            `absent_value_ytd`, `ytd_absence_rate`) were dropped before training to prevent the model from
            trivially memorising the target. The final feature set is:

            | # | Feature | Type | Notes |
            |---|---------|------|-------|
            | 1 | `STUDENT_GENDER` | Categorical | James-Stein encoded |
            | 2 | `RACE_GRP` | Categorical | Top 5 races kept; rest → OTHER |
            | 3 | `STUDENT_ETHNICITY` | Categorical | James-Stein encoded |
            | 4 | `LANG_GRP` | Categorical | Bucketed to English / Spanish / Other |
            | 5 | `STUDENT_CURRENT_GRADE_CODE` | Categorical | James-Stein encoded |
            | 6 | `SCHOOL_GRP` | Categorical | Top 20 schools kept; rest → OTHER |
            | 7 | `STUDENT_AGE` | Numeric | Strongest individual predictor (importance 0.319) |
            | 8 | `STUDENT_SPECIAL_ED_INDICATOR` | Numeric | Binary 0/1 |
            | 9 | `STUDENT_HOMELESS_INDICATOR` | Numeric | Binary 0/1 |

            ---

            **Chronic Rate (Train)**

            A student is labelled **chronic** (target = 1) when their year-to-date absence rate exceeds 10 %
            (`ytd_absence_rate > 0.10`). Because the training set includes all enrollment statuses, it captures
            a wide range of attendance patterns — including students who left or graduated with high absence
            histories. The near-balanced chronic / non-chronic ratio means the model does not face severe
            class imbalance, though `class_weight="balanced"` is still applied in the Random Forest to further
            equalise the per-class loss contribution.
            """,
            unsafe_allow_html=False,
        )

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Target Distribution</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">'
            '<p>This chart shows the balance between chronic and non-chronic outcomes in the training dataset.</p>'
            '<p>Class imbalance can affect model behavior, so it is important to see whether chronic absenteeism is underrepresented or overrepresented.</p>'
            '<p>If one class dominates, the model may require additional calibration or resampling to avoid biased predictions.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        counts = train_df[TARGET].value_counts().sort_index()
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Non-Chronic (0)", "Chronic (1)"], counts.values, color=[PALETTE["primary"], PALETTE["secondary"]], edgecolor="white", linewidth=0.8)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + counts.max() * 0.02, f"{int(b.get_height()):,}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_title("Training Target Breakdown")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        show_fig(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Age Distribution by Target</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">'
            '<p>This chart compares age profiles for students with chronic and non-chronic absenteeism.</p>'
            '<p>The distribution curves make it easy to see whether risk is concentrated among certain age groups.</p>'
            '<p>Use this insight to identify age cohorts that may require targeted interventions.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        age_df = train_df.dropna(subset=["STUDENT_AGE", TARGET]).copy()
        age_df = filter_valid_student_ages(age_df)
        plt.close('all')
        fig, ax = plt.subplots(figsize=(7, 4))
        if age_df.empty:
            st.warning("No valid age data available for the selected range.")
        else:
            sns.kdeplot(
                data=age_df,
                x="STUDENT_AGE",
                hue=TARGET,
                fill=True,
                palette=[PALETTE["primary"], PALETTE["secondary"]],
                ax=ax,
                alpha=0.5,
            )
            ax.set_xlabel("Student Age")
            ax.set_ylabel("Density")
            ax.set_xlim(VALID_AGE_MIN, VALID_AGE_MAX)
            ax.set_title("Age Distribution by Outcome")
            ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout()
            show_fig(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
if run_btn or st.session_state.get("model_trained", False):
    if run_btn:
        st.session_state.model_trained = True

    with st.spinner("Training Model …"):
        rf_model, js_encoder = train_model(train_path_input, test_path_input)
        X_test_encoded = get_encoded_data(test_df, js_encoder)
        y_test = test_df[TARGET].astype(int).values
        y_proba = rf_model.predict_proba(X_test_encoded)[:, 1]

    m_test  = threshold_metrics(y_test, y_proba, threshold)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – MODEL RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_results:
        st.markdown('<div class="section-header">Test-Set Performance</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">The model is evaluated on held-out test data. These metrics indicate how well the classifier identifies chronic absenteeism, including precision and recall for the high-risk class.</div>', unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: metric_card("Accuracy",  f"{m_test['accuracy']:.3f}")
        with c2: metric_card("Precision", f"{m_test['precision']:.3f}", "Chronic class")
        with c3: metric_card("Recall",    f"{m_test['recall']:.3f}",    "Chronic class")
        with c4: metric_card("F1 Score",  f"{m_test['f1']:.3f}",        "Chronic class")
        with c5: metric_card("ROC-AUC",   f"{m_test['auc']:.3f}")

        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-note">This matrix shows true/false positives and negatives for the selected threshold. It is useful for understanding the types of classification errors the model makes.</div>', unsafe_allow_html=True)
            cm = np.array([[m_test["tn"], m_test["fp"]], [m_test["fn"], m_test["tp"]]])
            plt.close('all')
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", xticklabels=["Pred Non-Chronic", "Pred Chronic"], yticklabels=["Actual Non-Chronic", "Actual Chronic"], ax=ax)
            ax.set_title(f"Threshold = {threshold}")
            fig.tight_layout()
            show_fig(fig, use_container_width=True)
            
        with col_roc:
            st.markdown('<div class="section-header">Precision and Recall over Thresholds</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-note">This plot compares precision and recall for different probability thresholds. Choose a threshold based on whether reducing false alarms or capturing more chronic cases is more important.</div>', unsafe_allow_html=True)
            thresholds_roc = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            rows_roc = [threshold_metrics(y_test, y_proba, t) for t in thresholds_roc]
            
            thr_df_roc = pd.DataFrame(rows_roc).rename(columns={
                "precision": "Precision", 
                "recall": "Recall", 
                "f1": "F1", 
                "accuracy": "Accuracy"
            }).assign(Threshold=thresholds_roc)
            
            plt.close('all')
            fig, ax = plt.subplots(figsize=(5, 3)) # Locked Small Size
            ax.plot(thr_df_roc["Threshold"], thr_df_roc["Precision"], "o-", color=PALETTE["primary"], label="Precision")
            ax.plot(thr_df_roc["Threshold"], thr_df_roc["Recall"], "s-", color=PALETTE["secondary"], label="Recall")
            ax.axvline(threshold, color="red", linestyle="--", label="Selected")
            ax.set_xlabel("Threshold"); ax.legend(); ax.spines[["top","right"]].set_visible(False)
            show_fig(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – THRESHOLD TUNER
    # ══════════════════════════════════════════════════════════════════════════
    with tab_threshold:
        st.markdown('<div class="section-header">Threshold Tuner</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">'
            '<p>This tab helps you select a decision threshold that matches your business priorities.</p>'
            '<p>Lower thresholds increase recall and capture more chronic absenteeism cases, but also raise false positive rates. Higher thresholds increase precision, reducing false alerts but potentially missing some at-risk students.</p>'
            '<p>Review the chart and the summary table to choose the threshold that balances intervention capacity with detection coverage.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        thresholds_tuner  = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        rows_tuner = []
        for t in thresholds_tuner:
            m = threshold_metrics(y_test, y_proba, t)
            rows_tuner.append({
                "Threshold": t,
                "Precision": round(m["precision"],3),
                "Recall":    round(m["recall"],3),
                "F1":        round(m["f1"],3),
                "Accuracy":  round(m["accuracy"],3),
                "TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"],
            })
        
        thr_df_tuner = pd.DataFrame(rows_tuner)

        plt.close('all')
        fig, ax = plt.subplots(figsize=(7, 4)) # Locked Small Size
        ax.plot(thr_df_tuner["Threshold"], thr_df_tuner["Precision"], "o-", color=PALETTE["primary"], label="Precision")
        ax.plot(thr_df_tuner["Threshold"], thr_df_tuner["Recall"],    "s-", color=PALETTE["secondary"], label="Recall")
        ax.plot(thr_df_tuner["Threshold"], thr_df_tuner["F1"],        "^-", color=PALETTE["success"], label="F1")
        ax.axvline(threshold, color="red", linestyle="--", lw=1.5, label=f"Current t={threshold}")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.set_title("Precision / Recall / F1 vs Threshold")
        ax.legend(); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        show_fig(fig, use_container_width=True)

        st.dataframe(thr_df_tuner.style.background_gradient(subset=["Precision","Recall","F1"], cmap="Blues"), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – DATA EXPLORER
    # ──────────────────────────────────────────────────────────────────────────
    with tab_explore:
        st.markdown('<div class="section-header">Interactive Data Explorer</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">'
            '<p>Explore how absenteeism patterns differ across student subgroups, schools, and grade levels.</p>'
            '<p>Apply filters to focus on a single demographic or compare multiple cohorts side-by-side. This makes it easier to identify the groups at highest risk and prioritize interventions.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        df_view = train_df.copy()

        # filter_mapping covers all filterable columns
        filter_mapping = {
            "Gender":                             "STUDENT_GENDER",
            "Race":                               "RACE_GRP",
            "Ethnicity":                          "STUDENT_ETHNICITY",
            "Language":                           "LANG_GRP",
            "Grade":                              "STUDENT_CURRENT_GRADE_CODE",
            "School":                             "SCHOOL_GRP",
            "Target (0=Non-Chronic, 1=Chronic)":  TARGET,
        }
        # Keys that are valid for group-by axis (everything except the outcome)
        groupable_keys = [k for k in filter_mapping if k != "Target (0=Non-Chronic, 1=Chronic)"]

        st.markdown("##### 1. Enable Filters")
        st.markdown(
            '<div class="section-note">'
            '<p>Select the filter categories you want to use. The active filters determine the available attribute selectors below.</p>'
            '<p>You can explore different demographics, school groups, and the target outcome to compare chronic absenteeism rates.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        active_filters = st.multiselect(
            "Activate Filters:",
            options=list(filter_mapping.keys()),
            default=["Gender", "Race", "Grade", "Target (0=Non-Chronic, 1=Chronic)"],
        )
        df_filtered = df_view.copy()

        if active_filters:
            st.markdown("##### 2. Apply Filters")
            fcols = st.columns(min(len(active_filters), 4))
            for i, filter_label in enumerate(active_filters):
                col_name = filter_mapping[filter_label]
                with fcols[i % 4]:
                    opts = sorted(df_filtered[col_name].dropna().unique())
                    default_vals = opts if len(opts) <= 30 else []
                    selected_opts = st.multiselect(
                        f"{filter_label}", options=opts, default=default_vals,
                        key=f"filter_{filter_label}",
                    )
                    if selected_opts:
                        df_filtered = df_filtered[df_filtered[col_name].isin(selected_opts)]
                    else:
                        st.info(f"No selection made for {filter_label}. This filter is ignored.")
                        # Treat empty selection as no filtering for this field
                        continue

        st.markdown("---")
        if not df_filtered.empty:
            # Only show group-by options that the user has actually enabled as filters
            # (and exclude Target since it's the outcome variable)
            active_groupable = [k for k in active_filters if k in groupable_keys]

            if not active_groupable:
                st.info("Enable at least one non-Target filter above to see a group-by chart.")
            else:
                group_label = st.selectbox(
                    "Select Category to Group By & Visualize:",
                    options=active_groupable,
                    key="grp_col",
                )
                group_col = filter_mapping[group_label]

                grp = (
                    df_filtered.groupby(group_col)[TARGET]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "Chronic Rate", "count": "N"})
                    .dropna()
                )
                if len(grp) > 0:
                    grp = grp.sort_values("N", ascending=False)
                    if len(grp) > 15:
                        st.warning(
                            f"{group_col} has {len(grp)} unique values. Limiting the chart to the top 15 categories."
                        )
                        grp = grp.head(15)
                    grp = grp.sort_values("Chronic Rate", ascending=True)

                    # Scale height with number of bars so labels never overlap
                    fig_h = max(2.5, len(grp) * 0.45 + 0.8)

                    _fig_close()
                    fig, ax = plt.subplots(figsize=(6, fig_h))
                    bars = ax.barh(grp.index.astype(str), grp["Chronic Rate"],
                                   color=PALETTE["primary"], height=0.6)
                    for b, (_, row) in zip(bars, grp.iterrows()):
                        ax.text(
                            b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
                            f"{row['Chronic Rate']:.1%} (n={int(row['N']):,})",
                            va="center", fontsize=9,
                        )
                    ax.set_xlabel("Chronic Absence Rate")
                    ax.set_title(f"Chronic Rate by {group_col}")
                    ax.spines[["top", "right"]].set_visible(False)
                    fig.tight_layout()

                    chart_col, _ = st.columns([1, 1])
                    with chart_col:
                        show_fig(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")

    # TAB 5 – EXPLAINABILITY (RULES, PERMUTATION, SHAP)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_explain:
        st.markdown('<div class="section-header">Global Model Explainability</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Understand which features most strongly influence model predictions and which factors are associated with increased chronic absenteeism risk.</div>', unsafe_allow_html=True)
        
        X_sample = X_test_encoded.sample(n=min(300, len(X_test_encoded)), random_state=42)
        
        st.markdown("### 1. Feature Direction & Impact (SHAP)")
        st.markdown("This plot shows how each feature contributes to the prediction across the dataset. Red points indicate high feature values, and points to the right increase the model's predicted probability of chronic absenteeism.")
        
        with st.spinner("Calculating SHAP values..."):
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
            '<div class="section-note">'
            '<p>SHAP values explain how each feature contributes to the prediction for individual observations.</p>'
            '<p>Points on the right increase the probability of chronic absenteeism, while points on the left decrease it. The color indicates whether the feature value is relatively high (red) or low (blue).</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("### 2. Sample Human-Readable Rules")
        st.markdown(
            '<div class="section-note">'
            '<p>This section shows a clear, full-width example of one representative decision tree from the Random Forest ensemble.</p>'
            '<p>The rule text below is expanded fully so you can see the complete tree structure without squeezing it into a narrow column.</p>'
            '<p><strong>How to read the rule tree:</strong> start at the top, then follow the indentation to see which conditions lead to a prediction.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        single_tree = rf_model.estimators_[0]
        rules = export_text(single_tree, feature_names=list(X_sample.columns), max_depth=3)
        st.code(rules, language="text")
        st.markdown(
            '<div class="section-note">'
            '<p><strong>Interpretation tip:</strong> focus on the first few splits to understand the main decision paths. The deeper lines are more specific sub-rules.</p>'
            '<p>This is an illustrative example of model logic, not the full ensemble. It is intended to help explain the type of decisions the model makes.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### 3. Permutation Importance")
        st.markdown("This chart shows how much model accuracy decreases when each feature is randomly shuffled. Higher importance means the feature is more critical to the model.")
        result = permutation_importance(rf_model, X_sample, y_test[X_sample.index], n_repeats=5, random_state=42, n_jobs=-1)
        perm_df = pd.DataFrame({"Feature": X_sample.columns, "Importance": result.importances_mean}).sort_values("Importance", ascending=True)
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(perm_df["Feature"], perm_df["Importance"], color=PALETTE["secondary"])
        ax.set_xlabel("Mean Accuracy Decrease")
        ax.spines[["top","right"]].set_visible(False)
        show_fig(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 – SINGLE PREDICTION (LOCAL EXPLAINABILITY)
    # ══════════════════════════════════════════════════════════════════════════════
    with tab_predict:
        st.markdown('<div class="section-header">Single Student Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Enter a student profile to generate a personalized risk score and explanation showing which features contributed most to this prediction.</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            p_gender = st.selectbox("STUDENT_GENDER", sorted(train_df["STUDENT_GENDER"].dropna().unique()))
            p_race   = st.selectbox("RACE_GRP", sorted(train_df["RACE_GRP"].dropna().unique()))
            p_eth    = st.selectbox("STUDENT_ETHNICITY", sorted(train_df["STUDENT_ETHNICITY"].dropna().unique()))
        with col2:
            p_lang   = st.selectbox("LANG_GRP", sorted(train_df["LANG_GRP"].dropna().unique()))
            p_grade  = st.selectbox("STUDENT_CURRENT_GRADE_CODE", sorted(train_df["STUDENT_CURRENT_GRADE_CODE"].dropna().unique()))
            p_school = st.selectbox("SCHOOL_GRP", sorted(train_df["SCHOOL_GRP"].dropna().unique()))
        with col3:
            p_age    = st.slider("STUDENT_AGE", 5, 22, 14)
            p_sped   = st.selectbox("STUDENT_SPECIAL_ED_INDICATOR", [0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)")
            p_home   = st.selectbox("STUDENT_HOMELESS_INDICATOR",   [0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)")

        if st.button("Predict Risk", type="primary"):
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
            with r1: metric_card("Risk Score", f"{proba_val:.3f}", f"at threshold {threshold}")
            with r2: metric_card("Prediction", "Chronic Risk" if pred_label == 1 else "Low Risk")
            with r3: metric_card("Probability", f"{proba_val*100:.1f}%", "of chronic absenteeism")

            st.markdown(
                '<div class="section-note">'
                '<p>The displayed risk score is the model’s predicted probability for chronic absenteeism given the selected profile.</p>'
                '<p>This score should be interpreted relative to the decision threshold: students above the threshold are flagged as high risk.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.divider()
            st.markdown("### Local Explanation: Why did this student receive this score?")
            
            # 🛑 FIX 4: BULLETPROOF 1-DIMENSIONAL SHAP FLATTENING FOR PANDAS
            local_shap = explainer.shap_values(row_encoded)
            
            if isinstance(local_shap, list):
                local_shap_chronic = np.array(local_shap[1]).flatten()
            else:
                if len(local_shap.shape) == 3:
                    local_shap_chronic = np.array(local_shap[:, :, 1]).flatten()
                else:
                    local_shap_chronic = np.array(local_shap).flatten()
            
            impact_df = pd.DataFrame({"Feature": row_encoded.columns, "Impact": local_shap_chronic})
            impact_df = impact_df.sort_values("Impact", ascending=True)
            impact_df["Color"] = impact_df["Impact"].apply(lambda x: PALETTE["danger"] if x > 0 else PALETTE["success"])
            
            plt.close('all')
            fig, ax = plt.subplots(figsize=(7, 4)) # Locked Small Size
            bars = ax.barh(impact_df["Feature"], impact_df["Impact"], color=impact_df["Color"])
            ax.axvline(0, color='black', linewidth=1)
            ax.set_xlabel("Impact on Prediction Score (SHAP Value)")
            ax.spines[["top","right","left"]].set_visible(False)
            
            for bar in bars:
                xval = bar.get_width()
                if xval > 0:
                    ax.text(xval + 0.005, bar.get_y() + bar.get_height()/2, f"+{xval:.3f}", va="center", color=PALETTE["danger"], fontsize=9)
                else:
                    ax.text(xval - 0.005, bar.get_y() + bar.get_height()/2, f"{xval:.3f}", ha="right", va="center", color=PALETTE["success"], fontsize=9)
            
            show_fig(fig, use_container_width=True)

else:
    with tab_results: st.info("Click **Train and Evaluate** in the sidebar.")
    with tab_threshold: st.info("Click **Train and Evaluate** in the sidebar.")
    with tab_explore: st.info("Click **Train and Evaluate** in the sidebar.")
    with tab_explain: st.info("Click **Train and Evaluate** in the sidebar.")
    with tab_predict: st.info("Click **Train and Evaluate** in the sidebar.")