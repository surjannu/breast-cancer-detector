"""Build breast_cancer_presentation.pptx — 10 slides, correct narrative arc."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Palette ───────────────────────────────────────────────────────────
NAVY    = RGBColor(0x0A, 0x23, 0x42)
TEAL    = RGBColor(0x02, 0x80, 0x90)
MID_NAV = RGBColor(0x1C, 0x72, 0x93)
LT_TEAL = RGBColor(0xA8, 0xD5, 0xE2)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
GRAY    = RGBColor(0x4A, 0x5E, 0x6F)
DARK    = RGBColor(0x1E, 0x29, 0x3B)
ORANGE  = RGBColor(0xE6, 0x9F, 0x00)
BLUE    = RGBColor(0x00, 0x72, 0xB2)
GREEN   = RGBColor(0x05, 0x96, 0x69)
PURPLE  = RGBColor(0x78, 0x37, 0x96)
LT_BLU  = RGBColor(0xF4, 0xF9, 0xFF)
LT_GRN  = RGBColor(0xF0, 0xFF, 0xF4)
BORD    = RGBColor(0xC8, 0xDC, 0xF0)
WARN_BG = RGBColor(0xFF, 0xF8, 0xE1)
WARN_TX = RGBColor(0x7B, 0x5A, 0x00)
STAGE_B = RGBColor(0x12, 0x2D, 0x52)

PROJECT = r"C:\Users\sures\DSI\personal_projects\breast-cancer-detector"
FIGURES = os.path.join(PROJECT, "reports", "figures")
OUTPUT  = os.path.join(PROJECT, "breast_cancer_presentation.pptx")

prs = Presentation()
prs.slide_width  = Inches(10)
prs.slide_height = Inches(5.625)
blank = prs.slide_layouts[6]

# ── Helpers ───────────────────────────────────────────────────────────

def set_bg(slide, rgb):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = rgb

def add_rect(slide, x, y, w, h, fill_rgb, border_rgb=None):
    s = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill_rgb
    if border_rgb:
        s.line.color.rgb = border_rgb
        s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    return s

def add_oval(slide, x, y, w, h, fill_rgb):
    s = slide.shapes.add_shape(9, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill_rgb
    s.line.fill.background()
    return s

def add_txt(slide, text, x, y, w, h, size, color,
            bold=False, align=PP_ALIGN.LEFT, italic=False, mono=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.bold = bold
    r.font.italic = italic
    if mono:
        r.font.name = "Consolas"
    return tb

def add_img(slide, fname, x, y, w, h):
    path = os.path.join(FIGURES, fname)
    if os.path.exists(path):
        slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    else:
        print(f"  [WARN] {path} not found")

def section_tag(slide, label, color):
    add_rect(slide, 0.40, 0.13, 2.10, 0.30, color)
    add_txt(slide, label, 0.40, 0.13, 2.10, 0.30, 8.5, WHITE,
            bold=True, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# Arc: hook the audience with the mission statement
# ═══════════════════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(blank)
set_bg(s1, NAVY)
add_oval(s1,  6.0, -1.8, 5.8, 5.8, MID_NAV)
add_oval(s1, -2.0,  3.6, 3.8, 3.8, MID_NAV)
add_rect(s1, 0.45, 1.38, 0.07, 2.45, TEAL)
add_txt(s1, "Breast Cancer Detection",  0.65, 1.38, 8.8, 0.85, 38, WHITE, bold=True)
add_txt(s1, "with Machine Learning",    0.65, 2.18, 8.8, 0.85, 38, WHITE, bold=True)
add_txt(s1, "How computers help doctors catch cancer early",
        0.65, 3.14, 7.8, 0.62, 18, LT_TEAL)
add_txt(s1, "UCI Wisconsin Dataset  ·  569 patients  ·  5 AI models compared",
        0.65, 5.06, 8.8, 0.35, 10, GRAY)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 2 — THE CHALLENGE
# Arc: establish the problem before showing any solution
# ═══════════════════════════════════════════════════════════════════════
s2 = prs.slides.add_slide(blank)
set_bg(s2, WHITE)
add_txt(s2, "The Challenge", 0.4, 0.18, 9.2, 0.58, 30, NAVY, bold=True)
add_txt(s2, "Diagnosing breast cancer requires careful analysis of complex data from biopsy images.",
        0.4, 0.78, 9.2, 0.38, 12.5, GRAY)
card_data = [
    (0.25,  TEAL,   "\U0001f52c", "569 Patient Samples",       "From real clinical biopsy records"),
    (3.50,  BLUE,   "\U0001f4ca", "30 Measurements Per Scan",  "Precise data from each microscope image"),
    (6.75,  ORANGE, "❓",         "One Critical Question",     "Benign or Malignant? Every answer matters"),
]
for cx, ic_col, icon, heading, desc in card_data:
    cw = 3.0
    add_rect(s2, cx, 1.22, cw, 3.62, LT_BLU, border_rgb=BORD)
    io_x = cx + cw / 2 - 0.46
    add_oval(s2, io_x, 1.45, 0.92, 0.92, ic_col)
    add_txt(s2, icon,    io_x - 0.04, 1.44, 1.0,       0.95, 24,   WHITE, align=PP_ALIGN.CENTER)
    add_txt(s2, heading, cx + 0.10,   2.50, cw - 0.20, 0.65, 13.5, DARK,  bold=True, align=PP_ALIGN.CENTER)
    add_txt(s2, desc,    cx + 0.10,   3.22, cw - 0.20, 0.75, 11.5, GRAY,  align=PP_ALIGN.CENTER)
for ax in [3.27, 6.52]:
    add_txt(s2, "→", ax, 2.65, 0.22, 0.44, 20, TEAL, bold=True, align=PP_ALIGN.CENTER)
add_rect(s2, 0.25, 5.02, 9.5, 0.42, WARN_BG, border_rgb=RGBColor(0xF5, 0xC5, 0x18))
add_txt(s2, "⚠  Catching malignant cancer early can be life-saving — every missed case has real consequences",
        0.45, 5.06, 9.1, 0.35, 11, WARN_TX)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 3 — DATASET OVERVIEW  (NEW)
# Arc: ground both audiences in the actual data before explaining the approach
# ═══════════════════════════════════════════════════════════════════════
s3 = prs.slides.add_slide(blank)
set_bg(s3, WHITE)
add_txt(s3, "Dataset Overview", 0.4, 0.15, 9.2, 0.55, 28, NAVY, bold=True)

# Left: class distribution image
add_img(s3, "class_distribution.png", 0.2, 0.82, 5.15, 3.55)
add_txt(s3, "Source: run_pipeline.py Step 3 — EDA visualisation",
        0.2, 4.44, 5.15, 0.26, 9, GRAY, italic=True)

# Right: 2×2 stat boxes
stat_rows = [
    [("569",           "Total Samples",  TEAL),   ("30",            "Features",      BLUE)],
    [("357  (62.7%)",  "Benign",         BLUE),   ("212  (37.3%)",  "Malignant",     ORANGE)],
]
for ri, row in enumerate(stat_rows):
    for ci, (val, lbl, col) in enumerate(row):
        bx = 5.55 + ci * 2.12
        by = 0.82 + ri * 0.90
        add_rect(s3, bx, by, 2.00, 0.80, col)
        vsize = 22 if len(val) <= 3 else 16
        add_txt(s3, val, bx, by + 0.06, 2.00, 0.44, vsize, WHITE, bold=True, align=PP_ALIGN.CENTER)
        add_txt(s3, lbl, bx, by + 0.53, 2.00, 0.24, 10,    WHITE, align=PP_ALIGN.CENTER)

# Right: info card
add_rect(s3, 5.55, 2.75, 4.12, 2.62, LT_BLU, border_rgb=BORD)

add_txt(s3, "Data Source", 5.70, 2.83, 3.90, 0.30, 12,   DARK, bold=True)
add_txt(s3, "UCI Breast Cancer Wisconsin Diagnostic (id=17)",
        5.70, 3.12, 3.90, 0.26, 10.5, DARK)
add_txt(s3, "ucimlrepo(id=17)  |  sklearn fallback",
        5.70, 3.36, 3.90, 0.24,  9.5, DARK, mono=True, italic=True)

add_rect(s3, 5.75, 3.64, 3.75, 0.03, BORD)   # separator

add_txt(s3, "Data Quality Checks", 5.70, 3.72, 3.90, 0.30, 12, DARK, bold=True)
quality_checks = [
    ("✓", "0 missing values — median impute ready"),
    ("✓", "0 duplicate rows — no cleaning needed"),
    ("✓", "Stratified 80/20 split  ·  random_state=42"),
]
for qi, (mark, txt) in enumerate(quality_checks):
    qy = 4.02 + qi * 0.30
    add_txt(s3, mark, 5.70, qy, 0.28, 0.27, 11, GREEN, bold=True)
    add_txt(s3, txt,  5.98, qy, 3.62, 0.27, 10, DARK)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 4 — OUR APPROACH  (overview — non-technical)
# Arc: show the 5-step workflow before diving into technical detail
# ═══════════════════════════════════════════════════════════════════════
s4 = prs.slides.add_slide(blank)
set_bg(s4, WHITE)
add_txt(s4, "Our Approach — Step by Step", 0.4, 0.18, 9.2, 0.58, 28, NAVY, bold=True)
add_txt(s4, "No technical knowledge needed — here is what happens behind the scenes",
        0.4, 0.80, 9.2, 0.38, 12, GRAY)
steps = [
    ("\U0001f3e5", "Patient",   "Data",       TEAL,    "1"),
    ("\U0001f9f9", "Clean &",   "Prepare",    MID_NAV, "2"),
    ("\U0001f916", "Train 5",   "AI Models",  BLUE,    "3"),
    ("\U0001f3c6", "Pick Best", "Model",      PURPLE,  "4"),
    ("✅",         "Predict",   "Diagnosis",  GREEN,   "5"),
]
sw, aw = 1.5, 0.42
sx0 = (10 - (len(steps) * sw + (len(steps) - 1) * aw)) / 2
sy, sh = 1.45, 3.30
for i, (icon, line1, line2, color, num) in enumerate(steps):
    sx = sx0 + i * (sw + aw)
    add_rect(s4, sx, sy, sw, sh, color)
    nc_x = sx + sw / 2 - 0.24
    add_oval(s4, nc_x, sy + 0.12, 0.48, 0.48, WHITE)
    add_txt(s4, num,   nc_x, sy + 0.12, 0.48, 0.48, 14, color, bold=True, align=PP_ALIGN.CENTER)
    add_txt(s4, icon,  sx,   sy + 0.72, sw,   0.68, 28, WHITE, align=PP_ALIGN.CENTER)
    add_txt(s4, line1, sx,   sy + 1.52, sw,   0.38, 12, WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_txt(s4, line2, sx,   sy + 1.90, sw,   0.38, 12, WHITE, bold=True, align=PP_ALIGN.CENTER)
    if i < len(steps) - 1:
        add_txt(s4, "▶", sx + sw + 0.06, sy + sh / 2 - 0.28, aw - 0.08, 0.56,
                15, RGBColor(0xB0, 0xC4, 0xD8), align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 5 — FEATURE ENGINEERING & PREPROCESSING  (technical)
# Arc: data layer detail for engineers — what enters the models
# ═══════════════════════════════════════════════════════════════════════
s5 = prs.slides.add_slide(blank)
set_bg(s5, WHITE)
section_tag(s5, "DATA LAYER", TEAL)
add_txt(s5, "Feature Engineering & Preprocessing", 0.4, 0.55, 9.2, 0.55, 24, NAVY, bold=True)
add_rect(s5, 4.88, 0.55, 0.04, 5.0, BORD)   # vertical divider

# Left: feature schema
add_txt(s5, "Feature Schema", 0.25, 1.20, 4.50, 0.38, 13.5, DARK, bold=True)
add_txt(s5, "10 cellular measurements  ×  3 statistical groups  =  30 features",
        0.25, 1.58, 4.50, 0.35, 10, GRAY)
pills = [("_mean", TEAL), ("_se", BLUE), ("_worst", ORANGE)]
for pi, (plabel, pcolor) in enumerate(pills):
    add_rect(s5, 0.25 + pi * 1.54, 2.05, 1.38, 0.34, pcolor)
    add_txt(s5, plabel, 0.25 + pi * 1.54, 2.05, 1.38, 0.34, 10.5, WHITE,
            bold=True, align=PP_ALIGN.CENTER, mono=True)
measurements = [
    ("radius",     "texture"),
    ("perimeter",  "area"),
    ("smoothness", "compactness"),
    ("concavity",  "concave_points"),
    ("symmetry",   "fractal_dimension"),
]
COL1_X, COL2_X, COL_W, ROW_H = 0.25, 2.55, 2.22, 0.39
for ri, (m1, m2) in enumerate(measurements):
    ry = 2.50 + ri * ROW_H
    bg = LT_BLU if ri % 2 == 0 else WHITE
    add_rect(s5, COL1_X, ry, COL_W, ROW_H, bg, border_rgb=BORD)
    add_rect(s5, COL2_X, ry, COL_W, ROW_H, bg, border_rgb=BORD)
    add_txt(s5, m1, COL1_X + 0.10, ry + 0.08, COL_W - 0.12, ROW_H - 0.10, 10.5, DARK, mono=True)
    add_txt(s5, m2, COL2_X + 0.10, ry + 0.08, COL_W - 0.12, ROW_H - 0.10, 10.5, DARK, mono=True)
add_rect(s5, 0.25, 4.50, 4.60, 0.52, RGBColor(0xE8, 0xF4, 0xFF), border_rgb=TEAL)
add_txt(s5, "Each measurement → 3 columns:  {name}_mean  ·  {name}_se  ·  {name}_worst",
        0.35, 4.54, 4.42, 0.44, 9.5, DARK, mono=True)

# Right: preprocessing steps
add_txt(s5, "Preprocessing Pipeline", 5.05, 1.20, 4.72, 0.38, 13.5, DARK, bold=True)
add_txt(s5, "src/data/preprocess.py  —  runs on every pipeline execution",
        5.05, 1.58, 4.72, 0.35, 10, GRAY, italic=True, mono=True)
preproc_steps = [
    (TEAL,   "1", "Drop unnamed index columns",   "df.loc[:, ~df.columns.str.match(r'^Unnamed')]"),
    (BLUE,   "2", "Median impute missing values",  "df.fillna(df.median(numeric_only=True))"),
    (PURPLE, "3", "Remove duplicate rows",          "df.drop_duplicates().reset_index(drop=True)"),
    (ORANGE, "4", "Label encode diagnosis",          "'M'→1  /  'B'→0    (sklearn inversion handled)"),
    (GREEN,  "5", "Fit & apply StandardScaler",     "scaler.fit(X_train)  ←  NO fit on test data!"),
    (DARK,   "6", "Stratified train_test_split",    "test_size=0.2, random_state=42  →  455 / 114"),
]
for si, (color, num, title, detail) in enumerate(preproc_steps):
    sy = 2.10 + si * 0.55
    add_oval(s5, 5.05, sy + 0.04, 0.36, 0.36, color)
    add_txt(s5, num,    5.05, sy + 0.04, 0.36, 0.36, 11, WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_txt(s5, title,  5.50, sy,        4.20, 0.28, 11, DARK,  bold=True)
    add_txt(s5, detail, 5.50, sy + 0.28, 4.20, 0.24,  9, GRAY,  italic=True, mono=True)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 6 — END-TO-END PIPELINE ARCHITECTURE  (technical)
# Arc: show the engineering blueprint — how all the pieces connect
# ═══════════════════════════════════════════════════════════════════════
s6 = prs.slides.add_slide(blank)
set_bg(s6, NAVY)
add_oval(s6, 6.8, -1.5, 5.0, 5.0, MID_NAV)
section_tag(s6, "PIPELINE OVERVIEW", TEAL)
add_txt(s6, "End-to-End Pipeline Architecture", 0.4, 0.60, 9.2, 0.55, 26, WHITE, bold=True)
add_txt(s6, "Orchestrated by run_pipeline.py — 6 sequential steps, all artifacts persisted as local files",
        0.4, 1.20, 9.2, 0.35, 11, LT_TEAL, italic=True)

BOX_W, GAP, BX0 = 2.12, 0.24, 0.40
BOX_Y, HDR_H, BODY_H = 1.68, 0.50, 2.90
stages = [
    ("INGEST",       TEAL,   ["ucimlrepo(id=17)  [primary]",  "sklearn fallback  [on error]",
                               "→ data/raw/",                  "   breast_cancer.csv",
                               "   569 rows × 31 cols"]),
    ("PREPROCESS",   BLUE,   ["Encode labels  M→1 / B→0",     "StandardScaler.fit(X_train)",
                               "Stratified 80/20 split",       "→ data/processed/",
                               "   6 CSVs + scaler.pkl"]),
    ("TRAIN & EVAL", PURPLE, ["5 classifiers trained",         "Acc/Prec/Rec/F1/AUC scored",
                               "Ranked by F1-Score",           "→ models/",
                               "   5×.pkl + best_model.pkl"]),
    ("SERVE",        GREEN,  ["streamlit run app.py",          "@st.cache_resource [model]",
                               "@st.cache_data [datasets]",   "Manual slider + CSV batch",
                               "→ predict_proba(input)"]),
]
for i, (label, color, bullets) in enumerate(stages):
    bx = BX0 + i * (BOX_W + GAP)
    add_rect(s6, bx, BOX_Y, BOX_W, HDR_H, color)
    add_txt(s6, label, bx, BOX_Y, BOX_W, HDR_H, 11, WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_rect(s6, bx, BOX_Y + HDR_H, BOX_W, BODY_H, STAGE_B)
    for j, bullet in enumerate(bullets):
        c = LT_TEAL if bullet.startswith("→") or bullet.startswith("   ") else WHITE
        add_txt(s6, bullet, bx + 0.12, BOX_Y + HDR_H + 0.12 + j * 0.53,
                BOX_W - 0.16, 0.48, 9, c, mono=True)
    if i < len(stages) - 1:
        add_txt(s6, "▶", bx + BOX_W + 0.04, BOX_Y + HDR_H + BODY_H / 2 - 0.22,
                GAP - 0.06, 0.44, 13, WHITE, align=PP_ALIGN.CENTER)
add_rect(s6, 0.40, 5.26, 9.2, 0.32, STAGE_B)
add_txt(s6, "All artifacts are plain files (CSV, pkl, PNG) — no database, no cloud, no message queue",
        0.56, 5.29, 8.9, 0.26, 9.5, GRAY, italic=True)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 7 — MODEL REGISTRY, TRAINING & EVALUATION  (technical)
# Arc: what models were trained, how the winner was selected
# ═══════════════════════════════════════════════════════════════════════
s7 = prs.slides.add_slide(blank)
set_bg(s7, WHITE)
section_tag(s7, "MODEL LAYER", PURPLE)
add_txt(s7, "Model Registry, Training & Evaluation", 0.4, 0.55, 9.2, 0.55, 24, NAVY, bold=True)
add_rect(s7, 5.22, 0.55, 0.04, 5.0, BORD)

# Left: model registry
add_txt(s7, "Model Registry — build_models()", 0.25, 1.20, 4.85, 0.38, 13, DARK, bold=True)
add_txt(s7, "One dict entry per model — evaluation & serialization are fully automatic",
        0.25, 1.58, 4.85, 0.35, 10, GRAY, italic=True)
model_rows = [
    ("Logistic Regression", "max_iter=1000, solver='lbfgs'",  "logistic_regression.pkl", TEAL,   False),
    ("Random Forest  ★",    "n_estimators=100, n_jobs=-1",    "random_forest.pkl",       GREEN,  True),
    ("SVM",                  "kernel='rbf', probability=True", "svm.pkl",                 BLUE,   False),
    ("XGBoost",              "n_estimators=100, verbosity=0",  "xgboost.pkl",             PURPLE, False),
    ("KNN",                  "n_neighbors=5, n_jobs=-1",       "knn.pkl",                 ORANGE, False),
]
for ri, (name, params, artifact, color, winner) in enumerate(model_rows):
    ry = 2.08 + ri * 0.65
    add_rect(s7, 0.25, ry, 4.85, 0.58, LT_GRN if winner else LT_BLU,
             border_rgb=GREEN if winner else BORD)
    add_rect(s7, 0.25, ry, 0.14, 0.58, color)
    add_txt(s7, name,     0.46, ry + 0.03, 4.55, 0.26, 10.5,
            GREEN if winner else DARK, bold=winner)
    add_txt(s7, params,   0.46, ry + 0.30, 2.60, 0.22, 9, GRAY, italic=True, mono=True)
    add_txt(s7, artifact, 3.22, ry + 0.30, 1.90, 0.22, 8.5, TEAL, mono=True)
add_rect(s7, 0.25, 5.28, 4.85, 0.32, LT_BLU, border_rgb=BORD)
add_txt(s7, "Best: sort_values('F1-Score').iloc[0]  →  best_model.pkl",
        0.35, 5.30, 4.72, 0.26, 8.5, DARK, mono=True)

# Right: evaluation results
add_txt(s7, "Evaluation Results  (held-out test set)", 5.38, 1.20, 4.42, 0.38, 13, DARK, bold=True)
add_txt(s7, "114 samples — never seen during training  ·  scored on 5 metrics",
        5.38, 1.58, 4.42, 0.35, 10, GRAY, italic=True)
add_img(s7, "model_comparison.png", 5.38, 2.00, 4.42, 2.55)
for mi, (val, lbl, col) in enumerate([("0.974","Accuracy",TEAL), ("0.963","F1-Score",BLUE), ("0.995","ROC-AUC",GREEN)]):
    mx = 5.38 + mi * 1.50
    add_rect(s7, mx, 4.65, 1.44, 0.65, col)
    add_txt(s7, val, mx, 4.68, 1.44, 0.36, 19, WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_txt(s7, lbl, mx, 5.05, 1.44, 0.22,  9, WHITE,            align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 8 — RESULTS AT A GLANCE
# Arc: now reveal the headline numbers — audience has the context to appreciate them
# ═══════════════════════════════════════════════════════════════════════
s8 = prs.slides.add_slide(blank)
set_bg(s8, WHITE)
add_txt(s8, "Results at a Glance", 0.4, 0.15, 9.2, 0.55, 28, NAVY, bold=True)
add_img(s8, "model_comparison.png",      0.20, 0.78, 5.15, 2.82)
add_img(s8, "confusion_matrix_best.png", 5.50, 0.78, 4.25, 2.82)
for i, (val, label, color) in enumerate([("97.4%","Accuracy",TEAL), ("96.3%","F1-Score",BLUE), ("Random Forest","Best Model",ORANGE)]):
    sx, scw = 0.25 + i * 3.2, 3.05
    add_rect(s8, sx, 3.72, scw, 1.30, color)
    add_txt(s8, val,   sx, 3.80, scw, 0.72, 19 if val == "Random Forest" else 28,
            WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_txt(s8, label, sx, 4.54, scw, 0.36, 11.5, WHITE, align=PP_ALIGN.CENTER)
add_txt(s8, "Out of 114 test cases, the model correctly identified nearly all of them",
        0.4, 5.15, 9.2, 0.28, 10, GRAY, italic=True, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 9 — WHY THIS MATTERS
# Arc: connect the technical achievement back to real-world impact
# ═══════════════════════════════════════════════════════════════════════
s9 = prs.slides.add_slide(blank)
set_bg(s9, NAVY)
add_oval(s9,  6.5, -1.0, 4.8, 4.8, MID_NAV)
add_oval(s9, -1.2,  3.5, 3.6, 3.6, MID_NAV)
add_txt(s9, "Why This Matters", 0.4, 0.18, 9.2, 0.58, 30, WHITE, bold=True)
matters = [
    ("\U0001f50d", "Early Detection",
     "The model flags suspicious cases before a specialist reviews them", TEAL),
    ("⚡",         "Speed",
     "Instant predictions — no waiting weeks for manual review",           MID_NAV),
    ("\U0001f91d", "Support Tool",
     "Designed to assist doctors, not replace them",
     RGBColor(0x14, 0x64, 0x82)),
]
for i, (icon, heading, body, color) in enumerate(matters):
    cx, cw, ch = 0.25 + i * 3.2, 3.05, 3.70
    add_rect(s9, cx, 1.08, cw, ch, color)
    icx = cx + cw / 2 - 0.43
    add_oval(s9, icx, 1.28, 0.86, 0.86, WHITE)
    add_txt(s9, icon,    icx - 0.04, 1.26, 0.94, 0.90, 26, NAVY,    align=PP_ALIGN.CENTER)
    add_txt(s9, heading, cx + 0.10,  2.28, cw - 0.2, 0.52, 14, WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_txt(s9, body,    cx + 0.15,  2.88, cw - 0.3, 1.35, 11.5, LT_TEAL, align=PP_ALIGN.CENTER)
add_txt(s9, "Smarter tools. Better outcomes.",
        0.4, 4.98, 9.2, 0.44, 20, WHITE, bold=True, italic=True, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 10 — CONCLUSION & NEXT STEPS  (NEW)
# Arc: close the loop — what was achieved, what comes next
# ═══════════════════════════════════════════════════════════════════════
s10 = prs.slides.add_slide(blank)
set_bg(s10, NAVY)
add_oval(s10,  6.8, -1.5, 5.2, 5.2, MID_NAV)
add_oval(s10, -1.5,  3.8, 3.5, 3.5, MID_NAV)
section_tag(s10, "CONCLUSION", GREEN)
add_txt(s10, "Conclusion & Next Steps", 0.4, 0.60, 9.2, 0.58, 28, WHITE, bold=True)

# Vertical divider
add_rect(s10, 4.95, 0.60, 0.04, 4.90, RGBColor(0x2A, 0x5A, 0x7A))

# Left: key achievements
add_txt(s10, "Key Achievements", 0.40, 1.28, 4.42, 0.38, 13, LT_TEAL, bold=True)
achievements = [
    "97.4% accuracy on 114 held-out samples — strong generalisation",
    "Random Forest ranked best across 5 trained classifiers",
    "Zero data leakage — scaler fit only on X_train",
    "One-command pipeline + interactive Streamlit dashboard",
]
for ai, ach in enumerate(achievements):
    ay = 1.78 + ai * 0.72
    add_oval(s10, 0.40, ay + 0.06, 0.34, 0.34, GREEN)
    add_txt(s10, "✓",  0.40, ay + 0.06, 0.34, 0.34, 11, WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_txt(s10, ach,  0.84, ay, 4.02, 0.62, 11.5, WHITE)

# Right: next steps
add_txt(s10, "Potential Improvements", 5.15, 1.28, 4.55, 0.38, 13, LT_TEAL, bold=True)
next_steps = [
    ("Hyperparameter Tuning",    "GridSearchCV / Optuna per classifier"),
    ("Class Imbalance (SMOTE)",  "imbalanced-learn — address 63/37 split"),
    ("Explainability (SHAP)",    "Per-prediction feature attribution"),
    ("Experiment Tracking",      "MLflow — versioned artifacts & metrics"),
    ("Test Suite",               "pytest — unit + integration coverage"),
]
for ni, (tech, desc) in enumerate(next_steps):
    ny = 1.78 + ni * 0.65
    add_rect(s10, 5.15, ny + 0.06, 0.06, 0.38, TEAL)
    add_txt(s10, tech, 5.28, ny,        4.42, 0.30, 11.5, WHITE, bold=True)
    add_txt(s10, desc, 5.28, ny + 0.30, 4.42, 0.28, 10,   LT_TEAL, italic=True)

# Closing line
add_txt(s10, "Thank you  ·  Questions?",
        0.4, 5.20, 9.2, 0.36, 16, WHITE, bold=True, italic=True, align=PP_ALIGN.CENTER)

# ── Save ──────────────────────────────────────────────────────────────
prs.save(OUTPUT)
print(f"Saved: {OUTPUT}  ({prs.slides.__len__()} slides)")
