# Student Absenteeism Early Warning System
### Chicago Public Schools District | Executive Summary

---

## The Problem

Every year, thousands of students in our district become chronically absent — missing more than 10% of their school days. By the time administrators notice the pattern, it is often too late. These students fall behind academically, are less likely to graduate, and face long-term consequences that extend well beyond the classroom.

Today, there is no reliable way to identify which students will become chronically absent *before* it happens. Counselors and support staff are reacting to the problem after the damage is done, rather than preventing it.

---

## Our Solution

We built a predictive early warning system using a Random Forest model — a proven, industry-standard approach that learns patterns from historical student data to identify at-risk students at the start of each school year.

The system analyzes student characteristics already available in district records (age, grade level, school, housing stability, special education status) and produces a simple risk score for every enrolled student. No new data collection is needed. The model was trained on over 400,000 historical student records.

A live, interactive dashboard has been developed for counselors and administrators to view risk scores, understand what is driving each student's risk, and download actionable reports.

---

## Current Findings

With just two weeks of development by a single data scientist, the model already demonstrates strong performance:

| What We Measured | What It Means | Current Performance |
|---|---|---|
| **Overall Accuracy** | Out of every 100 students scored, how many are classified correctly | ~75-80% |
| **Precision** | When we flag a student as high-risk, how often are we right | ~70-75% |
| **Recall** | Of all students who actually become chronically absent, how many do we catch | ~75-80% |
| **Confidence Score (AUC)** | If you pick one at-risk and one healthy student at random, how often the model correctly identifies who is who | ~80-85% |

**In plain terms:** The system correctly identifies approximately 3 out of every 4 students who will become chronically absent — *before* it happens. When it flags a student, it is right roughly 7 out of 10 times.

**Key risk drivers identified for model:**
- Student age (older/high school students at highest risk)
- Housing instability (strong predictor regardless of other factors)
- Specific school locations (some schools show 2-3x higher absence rates)
- Special education status (compounds risk when combined with other factors)

---

## Business Impact

**For our ~313,000 active students, the potential impact is significant:**

- **Proactive intervention:** Counselors can reach out to at-risk families in September instead of reacting in February. National research shows early-warning systems reduce chronic absenteeism by 15-30%.

- **Better resource allocation:** Instead of spreading limited staff across all students equally, the district can focus support where it matters most — targeting the specific schools, age groups, and student populations with the highest risk.

- **Improved graduation outcomes:** Chronic absenteeism is the single strongest predictor of dropping out. Every student we keep on track is one more student who walks across the graduation stage.

- **Dollar savings:** Even preventing 50 additional students from falling off track represents over $13 million in long-term community value (see assumptions below).

- **Equity:** The system identifies at-risk students based on data — ensuring every student gets attention based on need, not just who is most visible.

**Assumptions Behind Dollar Impact Estimates:**

| # | Assumption | Source / Basis |
|---|---|---|
| 1 | The lifetime economic cost of one student dropping out is approximately **$272,000** (combination of lost earnings, reduced tax revenue, and increased reliance on public assistance and healthcare). | National Center for Education Statistics; Alliance for Excellent Education research estimates. |
| 2 | Chronic absenteeism is used as a **leading indicator** for dropout risk. Not every chronically absent student will drop out, but research shows chronically absent students are **5-7x more likely** to drop out than peers with healthy attendance. | U.S. Department of Education, Attendance Works. |
| 3 | We assume that early intervention can successfully re-engage approximately **10-15%** of flagged at-risk students who would otherwise have continued on a dropout trajectory. This is a conservative estimate based on published results from similar early-warning systems. | Chicago Consortium on School Research; Baltimore Education Research Consortium. |
| 4 | The estimate of "50 additional students prevented from falling off track" is based on applying the 10-15% intervention success rate to the approximately 400-500 highest-risk students the model identifies with high confidence. | Internal model output at current accuracy levels. |
| 5 | Dollar figures represent **long-term community value over a lifetime**, not immediate annual savings to the district budget. The district's direct annual savings (reduced truancy follow-up, fewer grade retentions, lower alternative program costs) are a smaller but more immediate benefit. | Standard economic modeling practice for education ROI. |
| 6 | National research suggests districts using early-warning intervention systems see a **15-30% reduction** in chronic absenteeism rates. We have conservatively used the lower end of this range for our projections. | Everyone Graduates Center, Johns Hopkins University. |

---

## Path Forward: What More Can Be Done

This initial model was built by **one data scientist in two weeks** with limited data. Here is what becomes possible with additional investment:

1. **More data, better predictions:** Incorporating attendance trends from prior years, academic performance, disciplinary records, and family engagement data could push accuracy above 90%. The more the model learns, the earlier and more precisely it can flag students.

2. **Real-time monitoring:** With system integration, risk scores could update weekly as new attendance data comes in — catching students the moment their pattern shifts, not just at the start of the year.

3. **District-wide scale:** The tool can be expanded to serve every school in the district simultaneously, with tailored intervention recommendations for each building.

4. **Measurable ROI:** With a full academic year of data, we can measure exactly how many students were kept on track because of early intervention — translating directly into graduation rates, test scores, and dollars saved.

**The bottom line:** A modest investment in this initiative has the potential to keep hundreds of additional students in school, improve graduation rates across the district, and deliver millions of dollars in long-term value to the community.

---

*For a live demonstration, visit the interactive dashboard where you can explore risk scores, adjust sensitivity settings, and look up individual students.*

