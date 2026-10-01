# workshops/staged-data-leakage/code.py
#
# Leakage, staged. One honest evaluation, then four leaks added one at a time.
# Every stage is scored twice: by its own cross-validation, and on admissions
# the model could not have seen — new patients, a later window, features as
# they exist at the moment of discharge.

# --8<-- [start:setup]
import azimuth_nb as azimuth

env = azimuth.setup(SLUG, lang=LANG, profile=PROFILE)
# --8<-- [end:setup]


# --8<-- [start:build_world]
import numpy as np
import pandas as pd

cfg = env.cfg
CLINICAL = ["age", "chronic", "los", "n_meds", "lab_a", "lab_b", "emergency", "prior"]
MARKERS = [f"marker_{i:03d}" for i in range(cfg["n_markers"])]
POST_DISCHARGE = "followup_calls"


def cohort(n_patients, t_start, t_end, rng):
    """Admissions for n_patients over [t_start, t_end) months.

    Each patient carries a hidden frailty that raises the risk of every one of
    their admissions. It is never a column, but it makes one patient's visits
    resemble each other — which is what a careless split will exploit.
    """
    rows = []
    for pid in range(n_patients):
        frailty = rng.normal()
        age = rng.normal(65, 12)
        chronic = max(0.0, rng.normal(2 + 0.8 * frailty, 1.2))
        n_visits = 1 + rng.poisson(cfg["visits_mean"])
        times = np.sort(rng.uniform(t_start, t_end, n_visits))
        for j, t in enumerate(times):
            los = rng.gamma(2, 2 + 0.5 * max(frailty, -1.5))
            n_meds = rng.poisson(5 + chronic)
            lab_a = rng.normal(0.4 * frailty, 1)
            lab_b = rng.normal(0, 1)
            emergency = float(rng.random() < 0.3 + 0.1 * (frailty > 0))
            logit = (
                -1.3
                + 1.5 * frailty
                + 0.25 * (chronic - 2)
                + 0.12 * (los - 5)
                + 0.4 * emergency
                + 0.3 * lab_a
                + 0.15 * j
                + 0.01 * (age - 65)
            )
            readmitted = int(rng.random() < 1 / (1 + np.exp(-logit)))
            rows.append(
                {
                    "patient": pid,
                    "month": t,
                    "age": age,
                    "chronic": chronic,
                    "los": los,
                    "n_meds": n_meds,
                    "lab_a": lab_a,
                    "lab_b": lab_b,
                    "emergency": emergency,
                    "prior": float(j),
                    "readmitted": readmitted,
                }
            )
    df = pd.DataFrame(rows)
    for col in ["lab_a", "lab_b"]:  # labs are not always drawn
        df.loc[rng.random(len(df)) < 0.2, col] = np.nan
    markers = pd.DataFrame(rng.normal(size=(len(df), len(MARKERS))), columns=MARKERS)
    return pd.concat([df, markers], axis=1)


rng = np.random.default_rng(cfg["seed"])
dev = cohort(cfg["dev_patients"], 0, 24, rng)
holdout = cohort(cfg["holdout_patients"], 24, 36, rng)
holdout["patient"] += 1_000_000  # new people, not new visits by old ones

# The warehouse also holds care-coordination calls logged in the 30 days AFTER
# discharge. Readmitted patients get many of them, because the readmission is
# what triggers the calls. At the moment the model must decide, none have
# happened yet — so in the holdout, built as the model will meet the world,
# the column is zero.
dev[POST_DISCHARGE] = rng.poisson(0.2 + 1.5 * dev["readmitted"]).astype(float)
holdout[POST_DISCHARGE] = 0.0

y_dev = dev["readmitted"].to_numpy()
y_holdout = holdout["readmitted"].to_numpy()
groups = dev["patient"].to_numpy()

# Captured values are rounded where they are made, so the page quotes the
# same digits the notebook prints.
dev_rate_pct = round(100 * y_dev.mean(), 1)
holdout_rate_pct = round(100 * y_holdout.mean(), 1)
dev_admissions = len(dev)
holdout_admissions = len(holdout)

if env.lang == "ar":
    # Counts go after a colon: Arabic number-noun agreement depends on the
    # number, and these numbers come from the profile.
    print(
        f"التطوير — الأشهر 0–24 | المرضى: {cfg['dev_patients']} | "
        f"حالات الدخول: {dev_admissions} | نسبة العودة: {dev_rate_pct}٪"
    )
    print(
        f"البيانات المحجوزة — الأشهر 24–36 | مرضى جدد: {cfg['holdout_patients']} | "
        f"حالات الدخول: {holdout_admissions} | نسبة العودة: {holdout_rate_pct}٪"
    )
    print(
        f"الأعمدة — سريرية: {len(CLINICAL)} | مؤشّرات مخبرية: {len(MARKERS)} | "
        f"بعد الخروج: {POST_DISCHARGE}"
    )
else:
    print(
        f"development: {dev_admissions} admissions from {cfg['dev_patients']} patients, "
        f"months 0-24, readmission rate {dev_rate_pct}%"
    )
    print(
        f"holdout:     {holdout_admissions} admissions from {cfg['holdout_patients']} new "
        f"patients, months 24-36, readmission rate {holdout_rate_pct}%"
    )
    print(
        f"columns: {len(CLINICAL)} clinical, {len(MARKERS)} lab markers, "
        f"and one post-discharge field ({POST_DISCHARGE})"
    )
# --8<-- [end:build_world]


# --8<-- [start:protocol]
import warnings

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="The groups parameter is ignored")

STAGES = {
    0: "honest",
    1: "+ scale before split",
    2: "+ select before split",
    3: "+ split rows, not patients",
    4: "+ post-discharge column",
}
if env.lang == "ar":
    STAGES = {
        0: "نزيه",
        1: "+ توحيد قبل التقسيم",
        2: "+ انتقاء قبل التقسيم",
        3: "+ تقسيم الزيارات لا المرضى",
        4: "+ عمود ما بعد الخروج",
    }


def boosted():
    return HistGradientBoostingClassifier(
        max_iter=cfg["trees"], learning_rate=0.05, random_state=cfg["seed"]
    )


def evaluate(stage, labels=None, make_model=boosted):
    """Score one protocol by its own cross-validation, then on the holdout.

    Leaks are cumulative: stage 3 carries the leaks of stages 1 and 2. Each one
    is a single line below, and each line is one an analyst might write.
    """
    y = y_dev if labels is None else labels
    cols = CLINICAL + MARKERS
    if stage >= 4:
        cols = [*cols, POST_DISCHARGE]  # LEAK 4: a field that is filled in later
    X, X_out = dev[cols].to_numpy(), holdout[cols].to_numpy()

    steps = []
    if stage >= 1:  # LEAK 1: impute and scale using every development row
        imputer, scaler = SimpleImputer(), StandardScaler()
        X = scaler.fit_transform(imputer.fit_transform(X))
        X_out = scaler.transform(imputer.transform(X_out))
    else:
        steps += [SimpleImputer(), StandardScaler()]

    if stage >= 2:  # LEAK 2: choose features by their link to the label, on every row
        keep = SelectKBest(f_classif, k=cfg["k_select"]).fit(X, y).get_support()
        if stage >= 4:
            keep[-1] = True  # the analyst keeps the field; it ranks first anyway
        X, X_out = X[:, keep], X_out[:, keep]
    else:
        steps += [SelectKBest(f_classif, k=cfg["k_select"])]

    if stage >= 3:  # LEAK 3: shuffle admissions, so a patient lands on both sides
        splitter = KFold(cfg["folds"], shuffle=True, random_state=cfg["seed"])
    else:
        splitter = GroupKFold(cfg["folds"])

    model = make_pipeline(*steps, make_model())
    oof = cross_val_predict(model, X, y, cv=splitter, groups=groups, method="predict_proba")[:, 1]
    # The care team can call one patient in five: the threshold is set on the
    # cross-validated scores, exactly as it would be before launch.
    threshold = np.quantile(oof, 1 - cfg["alert_rate"])

    model.fit(X, y)
    p_out = model.predict_proba(X_out)[:, 1]
    return {
        "cv_auc": roc_auc_score(y, oof),
        "holdout_auc": roc_auc_score(y_holdout, p_out),
        "cv_recall": (oof[y == 1] >= threshold).mean(),
        "holdout_recall": (p_out[y_holdout == 1] >= threshold).mean(),
        "holdout_flagged": (p_out >= threshold).mean(),
        "p_out": p_out,
    }


honest = evaluate(0)
cv_auc_honest = round(honest["cv_auc"], 3)
holdout_auc_honest = round(honest["holdout_auc"], 3)
honest_gap = round(honest["cv_auc"] - honest["holdout_auc"], 3)

if env.lang == "ar":
    print(f"التقييم النزيه — AUC بالتحقّق المتقاطع: {cv_auc_honest:.3f}")
    print(f"                  AUC على البيانات المحجوزة: {holdout_auc_honest:.3f}")
    print(f"                  الفرق: {honest_gap:+.3f}")
else:
    print(f"honest protocol — cross-validated AUC: {cv_auc_honest:.3f}")
    print(f"                  holdout AUC:         {holdout_auc_honest:.3f}")
    print(f"                  gap:                 {honest_gap:+.3f}")
# --8<-- [end:protocol]


# --8<-- [start:ladder]
results = {0: honest}
for stage in range(1, 5):
    results[stage] = evaluate(stage)

header = (
    ("المرحلة", "AUC تحقّق", "AUC محجوزة", "الفرق")
    if env.lang == "ar"
    else ("stage", "CV AUC", "holdout AUC", "gap")
)
print(f"{header[0]:<30}{header[1]:>12}{header[2]:>14}{header[3]:>9}")
for stage, r in results.items():
    print(
        f"{stage} {STAGES[stage]:<28}{r['cv_auc']:>12.3f}{r['holdout_auc']:>14.3f}"
        f"{r['cv_auc'] - r['holdout_auc']:>+9.3f}"
    )

cv_auc_leaky = round(results[4]["cv_auc"], 3)
holdout_auc_leaky = round(results[4]["holdout_auc"], 3)
leaky_gap = round(results[4]["cv_auc"] - results[4]["holdout_auc"], 3)
cv_auc_stage3 = round(results[3]["cv_auc"], 3)
cv_gain_stage1 = round(results[1]["cv_auc"] - results[0]["cv_auc"], 3)
cv_gain_stage2 = round(results[2]["cv_auc"] - results[1]["cv_auc"], 3)
cv_gain_stage3 = round(results[3]["cv_auc"] - results[2]["cv_auc"], 3)
cv_gain_stage4 = round(results[4]["cv_auc"] - results[3]["cv_auc"], 3)
holdout_drop_leaky = round(results[0]["holdout_auc"] - results[4]["holdout_auc"], 3)

import matplotlib.pyplot as plt
from matplotlib import font_manager


def ar(text):
    """Shape and order Arabic for matplotlib, which draws isolated glyphs in
    logical order. A no-op in the English build."""
    if env.lang != "ar":
        return text
    import arabic_reshaper
    from bidi.algorithm import get_display

    # Line by line: the bidi pass would otherwise reorder whole lines.
    return "\n".join(get_display(arabic_reshaper.reshape(line)) for line in text.split("\n"))


if env.lang == "ar":
    from pathlib import Path

    # matplotlib's bundled DejaVu has no Arabic glyphs. Amiri comes from the
    # apt line in the dependencies cell; Ubuntu releases name the file
    # differently, so match case-insensitively.
    amiri = [
        p for p in Path("/usr/share/fonts").rglob("*.ttf") if p.name.lower() == "amiri-regular.ttf"
    ]
    if amiri:
        font_manager.fontManager.addfont(str(amiri[0]))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(amiri[0])).get_name()
    else:
        print("تنبيه: لم يُعثر على خطّ Amiri؛ ستظهر النصوص العربية في الرسوم مربّعات فارغة.")

stages = list(results)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(
    stages,
    [results[s]["cv_auc"] for s in stages],
    "o-",
    color="#c2410c",
    lw=2,
    label=ar("ما قاله التحقّق المتقاطع") if env.lang == "ar" else "what cross-validation said",
)
ax.plot(
    stages,
    [results[s]["holdout_auc"] for s in stages],
    "o-",
    color="#1d4ed8",
    lw=2,
    label=ar("ما حدث على بيانات لم تُرَ") if env.lang == "ar" else "what happened on unseen data",
)
# AUC's floor is 0.5, not 0: a coin flip. Anchoring there keeps "flat" honest.
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color="#999", lw=0.8)
ax.set_xticks(stages)
if env.lang == "ar":
    ax.set_xticklabels(
        [
            ar(s)
            for s in [
                "نزيه",
                "+ توحيد\nقبل التقسيم",
                "+ انتقاء\nقبل التقسيم",
                "+ تقسيم الزيارات\nلا المرضى",
                "+ عمود\nبعد الخروج",
            ]
        ]
    )
    ax.set_ylabel("AUC")
    ax.set_title(ar("كل تسريب يرفع الدرجة المُعلنة، ولا يرفع الأداء الحقيقي"))
else:
    ax.set_xticklabels(
        [
            "honest",
            "+ scale\nbefore split",
            "+ select\nbefore split",
            "+ split rows,\nnot patients",
            "+ post-discharge\ncolumn",
        ]
    )
    ax.set_ylabel("AUC")
    ax.set_title("Every leak raises the reported score. None raises the real one.")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
plt.show()
# --8<-- [end:ladder]


# --8<-- [start:same_model]
# Stages 0 to 3 differ only in how the score was ESTIMATED. The model each one
# would ship is trained on all development rows either way. Compare what those
# four models predict for the holdout patients.
reference = results[0]["p_out"]
same_model_max_diff = float(max(np.abs(results[s]["p_out"] - reference).max() for s in (1, 2, 3)))
cv_inflation_stage3 = round(results[3]["cv_auc"] - results[0]["cv_auc"], 3)

if env.lang == "ar":
    print(f"أكبر فرق بين تنبّؤات المراحل 0–3 على المرضى الجدد: {same_model_max_diff:.2e}")
    print(f"ومع ذلك ارتفع AUC المُعلَن من {cv_auc_honest:.3f} إلى {cv_auc_stage3:.3f}")
else:
    print(f"largest difference in holdout predictions across stages 0-3: {same_model_max_diff:.2e}")
    print(f"while the reported AUC rose from {cv_auc_honest:.3f} to {cv_auc_stage3:.3f}")
# --8<-- [end:same_model]


# --8<-- [start:alert]
recall_cv_leaky = round(results[4]["cv_recall"], 3)
recall_holdout_leaky = round(results[4]["holdout_recall"], 3)
recall_cv_honest = round(results[0]["cv_recall"], 3)
recall_holdout_honest = round(results[0]["holdout_recall"], 3)
recall_lost = round(results[0]["holdout_recall"] - results[4]["holdout_recall"], 3)
# Percentages for prose: "caught 0.2% of readmissions" reads; "0.002" does not.
recall_cv_leaky_pct = round(100 * results[4]["cv_recall"], 1)
recall_holdout_leaky_pct = round(100 * results[4]["holdout_recall"], 1)
flagged_holdout_leaky_pct = round(100 * results[4]["holdout_flagged"], 1)
recall_cv_honest_pct = round(100 * results[0]["cv_recall"], 1)
recall_holdout_honest_pct = round(100 * results[0]["holdout_recall"], 1)
alert_rate_pct = round(100 * cfg["alert_rate"])

if env.lang == "ar":
    print(f"{'':<30}{'الالتقاط (تحقّق)':>16}{'الالتقاط بعد الإطلاق':>22}{'نسبة المُنبَّه عليهم':>22}")
else:
    print(f"{'':<30}{'recall (CV)':>14}{'recall (holdout)':>18}{'share alerted':>16}")
for stage in (0, 4):
    r = results[stage]
    print(
        f"{stage} {STAGES[stage]:<28}{r['cv_recall']:>14.1%}{r['holdout_recall']:>18.1%}"
        f"{r['holdout_flagged']:>16.1%}"
    )

fig, ax = plt.subplots(figsize=(6.5, 3.6))
x = np.arange(2)
cv_bars = [recall_cv_honest, recall_cv_leaky]
out_bars = [recall_holdout_honest, recall_holdout_leaky]
ax.bar(
    x - 0.18,
    cv_bars,
    0.36,
    color="#c2410c",
    label=ar("في التحقّق المتقاطع") if env.lang == "ar" else "in cross-validation",
)
ax.bar(
    x + 0.18,
    out_bars,
    0.36,
    color="#1d4ed8",
    label=ar("بعد الإطلاق") if env.lang == "ar" else "after launch",
)
ax.set_ylim(0, 1)  # recall starts at zero, and zero is part of the story
ax.set_xticks(x)
if env.lang == "ar":
    ax.set_xticklabels([ar("نزيه"), ar("مع العمود المُسرَّب")])
    ax.set_ylabel(ar("نسبة حالات العودة التي التُقطت"))
    ax.set_title(ar("العتبة نفسها، بعد الإطلاق"))
else:
    ax.set_xticklabels(["honest", "with the leaked column"])
    ax.set_ylabel("share of readmissions caught")
    ax.set_title("The same threshold, after launch")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
# --8<-- [end:alert]


# --8<-- [start:model_swap]
# Is this a property of gradient boosting? Swap in a random forest and repeat
# the two ends of the ladder.
from sklearn.ensemble import RandomForestClassifier


def forest():
    return RandomForestClassifier(n_estimators=cfg["trees"], n_jobs=-1, random_state=cfg["seed"])


rf_honest = evaluate(0, make_model=forest)
rf_leaky = evaluate(4, make_model=forest)
rf_gap_honest = round(rf_honest["cv_auc"] - rf_honest["holdout_auc"], 3)
rf_gap_leaky = round(rf_leaky["cv_auc"] - rf_leaky["holdout_auc"], 3)
rf_cv_auc_leaky = round(rf_leaky["cv_auc"], 3)
rf_recall_holdout_leaky_pct = round(100 * rf_leaky["holdout_recall"], 1)

if env.lang == "ar":
    print(f"{'غابة عشوائية':<30}{'AUC تحقّق':>12}{'AUC محجوزة':>14}{'الالتقاط بعد الإطلاق':>22}")
else:
    print(f"{'random forest':<30}{'CV AUC':>12}{'holdout AUC':>14}{'recall after launch':>22}")
for stage, r in ((0, rf_honest), (4, rf_leaky)):
    print(
        f"{stage} {STAGES[stage]:<28}{r['cv_auc']:>12.3f}{r['holdout_auc']:>14.3f}"
        f"{r['holdout_recall']:>22.1%}"
    )
# --8<-- [end:model_swap]


# --8<-- [start:shuffle_audit]
# A cheap test anyone can run: destroy the labels, rerun the WHOLE protocol.
# With no signal left, an honest protocol must score about 0.5. Anything
# reliably above it was learned from the procedure, not from the patients.
audit = {}
for stage in STAGES:
    scores = []
    for k in range(cfg["shuffles"]):
        shuffled = np.random.default_rng(cfg["seed"] + 100 + k).permutation(y_dev)
        scores.append(evaluate(stage, labels=shuffled)["cv_auc"])
    audit[stage] = float(np.mean(scores))

head = (
    ("المرحلة", "AUC على تسميات مخلوطة", "AUC المُعلَن")
    if env.lang == "ar"
    else ("stage", "AUC on shuffled labels", "reported AUC")
)
print(f"{head[0]:<30}{head[1]:>24}{head[2]:>22}")
for stage in STAGES:
    print(f"{stage} {STAGES[stage]:<28}{audit[stage]:>24.3f}{results[stage]['cv_auc']:>22.3f}")

audit_auc_honest = round(audit[0], 3)
audit_auc_stage2 = round(audit[2], 3)
audit_auc_leaky = round(audit[4], 3)
# --8<-- [end:shuffle_audit]


# --8<-- [start:screen_columns]
# YOUR TURN.
#
# The shuffle audit tests the procedure. This screen tests the columns: score
# each one ALONE against the label. A single field that nearly matches the
# whole model is a question to ask whoever owns that field, not a gift.
SUSPICIOUS_AUC = 0.80

solo = {}
for col in CLINICAL + MARKERS + [POST_DISCHARGE]:
    values = dev[col].fillna(dev[col].median()).to_numpy()
    a = roc_auc_score(y_dev, values)
    solo[col] = max(a, 1 - a)  # direction does not matter for a screen

ranked = sorted(solo.items(), key=lambda kv: kv[1], reverse=True)
for col, score in ranked[:8]:
    flag = (
        "  <-- " + ("اسأل عنه" if env.lang == "ar" else "ask about it")
        if score >= SUSPICIOUS_AUC
        else ""
    )
    print(f"{col:<18}{score:.3f}{flag}")

solo_auc_post_discharge = round(solo[POST_DISCHARGE], 3)
solo_auc_best_clinical = round(max(solo[c] for c in CLINICAL), 3)
solo_auc_best_marker = round(max(solo[c] for c in MARKERS), 3)
# --8<-- [end:screen_columns]


# --8<-- [start:verify]
# The control first: if the honest model never learned to rank patients, every
# comparison below is a comparison against nothing.
honest_learns_ok = env.check("honest-model-learns", holdout_auc_honest)
honest_gap_ok = env.check("honest-cv-tracks-holdout", honest_gap)
recall_lost_ok = env.check("leak-costs-recall", recall_lost)
# --8<-- [end:verify]


# --8<-- [start:finish]
receipt = env.receipt()
# --8<-- [end:finish]
