"""
Generate a professional single-page PDF Executive Summary.
Uses matplotlib's PDF backend (no external PDF library needed).
Everything fits on ONE page.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import textwrap

OUTPUT_PATH = "/Users/Z00D0J9/Desktop/Streamlit_Absentism/Executive_One_Pager.pdf"

# ── Colors ──
HEADER_COLOR = "#0F548C"
SUBHEADER_COLOR = "#0E8A8C"
BODY_COLOR = "#1F2937"

def render_page(pdf):
    """Render the entire one-pager on a single page."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')
    
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    y = 0.975
    L = 0.06  # left margin
    
    # ── TITLE ──
    ax.text(0.5, y, "Student Absenteeism Early Warning System", fontsize=13, fontweight='bold',
            color=HEADER_COLOR, ha='center', va='top', fontfamily='sans-serif')
    y -= 0.025
    ax.text(0.5, y, "Chicago Public Schools District  |  Executive Summary", fontsize=8,
            color='#64748B', ha='center', va='top', fontfamily='sans-serif')
    y -= 0.018
    ax.axhline(y=y, xmin=0.05, xmax=0.95, color='#CBD5E1', linewidth=0.7)
    y -= 0.012

    # ── HELPER FUNCTIONS ──
    def section_head(title):
        nonlocal y
        rect = Rectangle((L - 0.01, y - 0.005), 0.005, 0.014, color=SUBHEADER_COLOR)
        ax.add_patch(rect)
        ax.text(L + 0.005, y, title, fontsize=8.5, fontweight='bold', color=HEADER_COLOR,
                va='top', fontfamily='sans-serif')
        y -= 0.020

    def body(text, indent=L, width=110, size=7, spacing=0.012):
        nonlocal y
        wrapped = textwrap.fill(text, width=width)
        for line in wrapped.split('\n'):
            ax.text(indent, y, line, fontsize=size, color=BODY_COLOR,
                    va='top', fontfamily='sans-serif')
            y -= spacing

    def bullet(text, bold_prefix=False, indent=L+0.01, size=7, spacing=0.012):
        nonlocal y
        if bold_prefix and ":" in text:
            parts = text.split(":", 1)
            ax.text(indent, y, "\u2022 " + parts[0] + ":", fontsize=size, fontweight='bold',
                    color=BODY_COLOR, va='top', fontfamily='sans-serif')
            rest = textwrap.fill(parts[1].strip(), width=105)
            y -= spacing
            for line in rest.split('\n'):
                ax.text(indent + 0.015, y, line, fontsize=size, color=BODY_COLOR,
                        va='top', fontfamily='sans-serif')
                y -= spacing
        else:
            wrapped = textwrap.fill(text, width=105)
            lines = wrapped.split('\n')
            for i, line in enumerate(lines):
                prefix = "\u2022 " if i == 0 else "   "
                ax.text(indent, y, prefix + line, fontsize=size, color=BODY_COLOR,
                        va='top', fontfamily='sans-serif')
                y -= spacing

    def gap(g=0.006):
        nonlocal y
        y -= g

    # ═══════════════════════════════════════════════
    # SECTION 1: THE PROBLEM
    # ═══════════════════════════════════════════════
    section_head("The Problem")
    body("Every year, thousands of students in our district become chronically absent \u2014 missing more than 10% of school days. By the time administrators notice, it is often too late. These students fall behind, are less likely to graduate, and face long-term consequences. Today, there is no reliable way to identify at-risk students before it happens. Staff are reacting after the damage is done, rather than preventing it.")
    gap(0.008)

    # ═══════════════════════════════════════════════
    # SECTION 2: OUR SOLUTION
    # ═══════════════════════════════════════════════
    section_head("Our Solution")
    body("We built a predictive early warning system using a Random Forest model \u2014 a proven approach that learns patterns from 400,000+ historical student records to score every enrolled student's risk at the start of each year. It uses data already in district systems (age, grade, school, housing stability, special education status). No new data collection is needed. A live interactive dashboard lets counselors view scores and download action reports.")
    gap(0.008)

    # ═══════════════════════════════════════════════
    # SECTION 3: CURRENT FINDINGS
    # ═══════════════════════════════════════════════
    section_head("Current Findings (2 weeks, 1 data scientist)")
    
    # Table
    col_positions = [L + 0.005, 0.30, 0.72]
    headers = ["Metric", "Meaning", "Result"]
    for h, p in zip(headers, col_positions):
        ax.text(p, y, h, fontsize=6.5, fontweight='bold', color=HEADER_COLOR,
                va='top', fontfamily='sans-serif')
    y -= 0.004
    ax.axhline(y=y, xmin=0.06, xmax=0.92, color='#CBD5E1', linewidth=0.4)
    y -= 0.010
    
    rows = [
        ("Overall Accuracy", "Students classified correctly out of 100", "~75\u201380%"),
        ("Precision", "When we flag a student, how often we're right", "~70\u201375%"),
        ("Recall", "Of all truly at-risk students, how many we catch", "~75\u201380%"),
        ("Confidence (AUC)", "Ability to rank risk correctly", "~80\u201385%"),
    ]
    for r in rows:
        for val, p in zip(r, col_positions):
            ax.text(p, y, val, fontsize=6.5, color=BODY_COLOR, va='top', fontfamily='sans-serif')
        y -= 0.013

    gap(0.004)
    body("In plain terms: The model catches ~3 out of 4 students who will become chronically absent, before it happens. Key drivers: student age, housing instability, school location, and special education status.", size=6.8)
    gap(0.008)

    # ═══════════════════════════════════════════════
    # SECTION 4: BUSINESS IMPACT
    # ═══════════════════════════════════════════════
    section_head("Business Impact (313,000 active students)")
    bullet("Proactive intervention: Reach at-risk families in September, not February. Research shows 15\u201330% reduction in absenteeism.", bold_prefix=True, size=6.8, spacing=0.011)
    bullet("Graduation outcomes: Chronic absenteeism is the #1 predictor of dropping out. Keep students on track to graduate.", bold_prefix=True, size=6.8, spacing=0.011)
    bullet("Dollar savings: Preventing just 50 students from dropping off track = $13M+ in long-term community value.", bold_prefix=True, size=6.8, spacing=0.011)
    bullet("Resource focus: Target support at specific schools, ages, and populations with highest risk instead of spreading thin.", bold_prefix=True, size=6.8, spacing=0.011)
    bullet("Equity: Every student gets attention based on data-driven need, not visibility.", bold_prefix=True, size=6.8, spacing=0.011)
    gap(0.008)

    # ═══════════════════════════════════════════════
    # SECTION 5: ASSUMPTIONS
    # ═══════════════════════════════════════════════
    section_head("Assumptions Behind Dollar Estimates")
    
    assumptions = [
        "Lifetime cost of one dropout = ~$272,000 (lost earnings + public services). Source: Alliance for Excellent Education.",
        "Chronically absent students are 5\u20137x more likely to drop out. Source: U.S. Dept. of Education.",
        "Conservative 10\u201315% intervention success rate applied to ~400\u2013500 highest-risk students = ~50 students saved.",
        "Figures represent long-term community value over a lifetime, not immediate annual district budget savings.",
        "15\u201330% absenteeism reduction from early-warning systems; we use the lower end. Source: Johns Hopkins University.",
    ]
    for a in assumptions:
        ax.text(L + 0.01, y, "\u2022 ", fontsize=6.5, color=SUBHEADER_COLOR, va='top', fontfamily='sans-serif')
        wrapped = textwrap.fill(a, width=112)
        for line in wrapped.split('\n'):
            ax.text(L + 0.02, y, line, fontsize=6.3, color=BODY_COLOR, va='top', fontfamily='sans-serif')
            y -= 0.011
    gap(0.008)

    # ═══════════════════════════════════════════════
    # SECTION 6: PATH FORWARD
    # ═══════════════════════════════════════════════
    section_head("Path Forward")
    body("Built by 1 data scientist in 2 weeks. With further investment:", size=6.8)
    bullet("More data (prior attendance, grades, engagement) \u2192 accuracy above 90%", size=6.8, spacing=0.011)
    bullet("Real-time weekly score updates \u2192 catch students the moment patterns shift", size=6.8, spacing=0.011)
    bullet("District-wide scale \u2192 every school, tailored recommendations per building", size=6.8, spacing=0.011)
    bullet("Measurable ROI \u2192 track exactly how many students stayed on track due to early intervention", size=6.8, spacing=0.011)
    gap(0.006)
    body("Bottom line: A modest investment can keep hundreds more students in school, improve graduation rates, and deliver millions in long-term community value.", size=7, indent=L)

    # ── FOOTER ──
    ax.text(0.5, 0.015, "For a live demonstration, visit the interactive dashboard to explore risk scores and individual student lookups.",
            fontsize=6.5, fontstyle='italic', color='#94A3B8', ha='center', va='bottom', fontfamily='sans-serif')

    pdf.savefig(fig)
    plt.close(fig)


# ── Generate PDF ──
with PdfPages(OUTPUT_PATH) as pdf:
    render_page(pdf)

print(f"PDF generated successfully: {OUTPUT_PATH}")
