# workshops/fit-your-own-scaling-law/code.py
#
# Measure a scaling law instead of quoting one. Train a grid of small byte-level
# transformers at fixed compute budgets (IsoFLOP curves), locate the best model
# size at each budget, fit power laws to those optima, extrapolate to a budget
# four times larger than anything fitted, then train that model and compare.

# --8<-- [start:setup]
import azimuth_nb as azimuth

env = azimuth.setup(SLUG, lang=LANG, profile=PROFILE)
# --8<-- [end:setup]


# --8<-- [start:data]
import math

import numpy as np
import torch


def tr(en, ar):
    """Pick the printed string for the notebook's language."""
    return ar if env.lang == "ar" else en


cfg = env.cfg
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(cfg["seed"])

raw = np.frombuffer(env.assets["tinystories-v2-head.txt"].read_bytes(), dtype=np.uint8)
raw = raw[: int(cfg["corpus_bytes"])]
n_val = int(cfg["val_bytes"])
val_bytes, train_bytes = raw[:n_val], raw[n_val:]

# The zero-skill reference: predict every byte from its overall frequency.
counts = np.bincount(train_bytes, minlength=256).astype(np.float64)
probs = counts[counts > 0] / counts.sum()
unigram_loss = round(float(-(probs * np.log(probs)).sum()), 4)

# Cut the training bytes into non-overlapping windows and shuffle them ONCE.
# Every run reads a prefix of this same order, so no run sees a byte twice and
# all runs at all sizes see the same data in the same sequence.
ctx = int(cfg["context"])
n_chunks = len(train_bytes) // (ctx + 1)
train_chunks = torch.from_numpy(
    train_bytes[: n_chunks * (ctx + 1)].copy()
).view(n_chunks, ctx + 1).to(device)
order = torch.randperm(n_chunks, generator=torch.Generator().manual_seed(cfg["seed"])).to(device)
max_train_tokens = n_chunks * ctx

n_val_chunks = len(val_bytes) // (ctx + 1)
val_chunks = torch.from_numpy(
    val_bytes[: n_val_chunks * (ctx + 1)].copy()
).view(n_val_chunks, ctx + 1).to(device)

print(tr(
    f"device {device} · train {len(train_bytes)/1e6:.1f} MB · val {len(val_bytes)/1e6:.2f} MB",
    f"الجهاز {device} · بيانات التدريب {len(train_bytes)/1e6:.1f} ميغابايت · التحقق {len(val_bytes)/1e6:.2f} ميغابايت",
))
print(tr(
    f"unigram baseline: {unigram_loss:.3f} nats/byte",
    f"خط الأساس الأحادي: {unigram_loss:.3f} نات لكل بايت",
))
print(train_bytes[:300].tobytes().decode("utf-8", errors="replace"))
# --8<-- [end:data]


# --8<-- [start:model]
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    """A byte-level GPT. The output layer shares weights with the input embedding."""

    def __init__(self, n_layer, d_model):
        super().__init__()
        self.tok = nn.Embedding(256, d_model)
        self.pos = nn.Embedding(ctx, d_model)
        self.blocks = nn.ModuleList(Block(d_model, d_model // 16) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x) @ self.tok.weight.T


def count_params(n_layer, d_model):
    """Two bookkeeping conventions for the same network."""
    block = 12 * d_model * d_model + 13 * d_model      # attention + MLP + norms
    embedding = 256 * d_model + ctx * d_model           # token + position tables
    non_embedding = n_layer * block + 2 * d_model       # + the final norm
    return {"total": non_embedding + embedding, "non_embedding": non_embedding}


# A pool of shapes, close to continuous in size. Width and depth grow together,
# kept within a band of width-to-depth ratios.
ladder = [(n_layer, d_model)
          for n_layer in range(1, cfg["max_layers"] + 1)
          for d_model in range(16, cfg["max_width"] + 1, 16)
          if cfg["aspect_min"] <= d_model / n_layer <= cfg["aspect_max"]]
sizes = {s: count_params(*s) for s in ladder}
ladder.sort(key=lambda s: sizes[s]["total"])

# The formula must agree with the network itself.
for shape in (ladder[0], ladder[-1]):
    real = sum(p.numel() for p in TinyLM(*shape).parameters())
    assert real == sizes[shape]["total"], (shape, real, sizes[shape])


def nearest_shape(n_target):
    return min(ladder, key=lambda s: abs(math.log(sizes[s]["total"] / n_target)))


print(tr(
    f"{len(ladder)} shapes, from {sizes[ladder[0]]['total']:,} to {sizes[ladder[-1]]['total']:,} parameters",
    f"{len(ladder)} شكلاً، من {sizes[ladder[0]]['total']:,} إلى {sizes[ladder[-1]]['total']:,} معاملاً",
))
print(tr("layers  width   params (total)   non-embedding", "الطبقات  العرض   المعاملات (الكلّ)   دون التضمين"))
for shape in ladder[:: max(1, len(ladder) // 8)]:
    n = sizes[shape]
    print(f"{shape[0]:>6}  {shape[1]:>5}   {n['total']:>14,}   {n['non_embedding']:>13,}")
# --8<-- [end:model]


# --8<-- [start:trainer]
import time


def train_run(shape, tokens, seed):
    """Train one model on exactly `tokens` fresh tokens; return its validation loss.

    The cosine schedule is stretched to this run's own length. That detail is the
    one the Chinchilla paper singles out: a schedule set for a longer run leaves
    a shorter run undertrained, and the scaling law inherits the error.
    """
    batch = int(cfg["batch"])
    steps = max(1, round(tokens / (batch * ctx)))
    assert steps * batch <= n_chunks, "run would repeat data; lower the budget"
    torch.manual_seed(seed)
    model = TinyLM(*shape).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95),
                            weight_decay=0.1)
    warmup = max(1, int(0.05 * steps))

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    for step in range(steps):
        rows = train_chunks[order[step * batch:(step + 1) * batch]].long()
        x, y = rows[:, :-1], rows[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = F.cross_entropy(model(x).float().view(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
    return evaluate(model), steps * batch * ctx


@torch.no_grad()
def evaluate(model):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(val_chunks), 256):
        rows = val_chunks[i:i + 256].long()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(rows[:, :-1]).float()
        total += F.cross_entropy(logits.reshape(-1, 256), rows[:, 1:].reshape(-1),
                                 reduction="sum").item()
        n += rows[:, 1:].numel()
    return total / n


# One short run to make sure the loop learns before the grid spends real time.
t0 = time.time()
probe_loss, _ = train_run(ladder[0], int(cfg["min_tokens"]), cfg["seed"])
probe_loss = round(probe_loss, 4)
print(tr(
    f"probe run: loss {probe_loss:.3f} nats/byte (unigram {unigram_loss:.3f}) in {time.time()-t0:.1f}s",
    f"تشغيل تجريبي: الخسارة {probe_loss:.3f} نات لكل بايت (الأحادي {unigram_loss:.3f}) خلال {time.time()-t0:.1f} ث",
))
# --8<-- [end:trainer]


# --8<-- [start:plan]
# Compute budgets, spaced by a constant factor. Within a budget C, each run
# splits the same compute differently: N parameters trained on D = C / (6 N)
# tokens. The splits are chosen by tokens-per-parameter ratio, D / N, over a
# wide range, because we do not know in advance where the best split lies.
budgets = [cfg["budget_min"] * cfg["budget_factor"] ** i for i in range(cfg["n_budgets"])]

plan = []
for C in budgets:
    chosen = set()
    for ratio in cfg["ratios"]:
        shape = nearest_shape(math.sqrt(C / (6 * ratio)))
        tokens = C / (6 * sizes[shape]["total"])
        if shape not in chosen and cfg["min_tokens"] <= tokens <= max_train_tokens:
            chosen.add(shape)
            plan.append({"C": C, "shape": shape, "tokens": tokens})

planned_tokens = sum(p["tokens"] for p in plan)
print(tr(
    f"{len(plan)} runs across {len(budgets)} budgets · {planned_tokens/1e6:.0f}M tokens in total",
    f"{len(plan)} تشغيلاً على {len(budgets)} ميزانيات · {planned_tokens/1e6:.0f} مليون رمز إجمالاً",
))
for C in budgets:
    row = [p for p in plan if p["C"] == C]
    shown = ", ".join(f"{sizes[p['shape']]['total']/1e3:.0f}k" for p in sorted(row, key=lambda p: p["tokens"]))
    print(f"  C = {C:.1e} FLOPs  →  N = {shown}")
# --8<-- [end:plan]


# --8<-- [start:grid]
runs = []
t_grid = time.time()
for i, p in enumerate(plan):
    loss, tokens = train_run(p["shape"], p["tokens"], cfg["seed"] + i)
    n = sizes[p["shape"]]
    runs.append({"C": p["C"], "shape": p["shape"], "N": n["total"],
                 "N_ne": n["non_embedding"], "D": tokens, "loss": loss})
    if i == 0 or (i + 1) % 5 == 0 or i + 1 == len(plan):
        done = sum(r["D"] for r in runs) / planned_tokens
        eta = max(0.0, (time.time() - t_grid) / done * (1 - done) / 60)
        print(tr(f"  {i+1}/{len(plan)} runs · ~{eta:.0f} min left",
                 f"  {i+1}/{len(plan)} تشغيلاً · بقي نحو {eta:.0f} دقيقة"))

print()
print(tr("   budget C     params N     tokens D    D/N    loss",
         "   الميزانية C   المعاملات N   الرموز D    D/N    الخسارة"))
for r in runs:
    print(f"  {r['C']:.1e}  {r['N']:>10,}  {r['D']/1e6:>8.2f}M  {r['D']/r['N']:>6.1f}  {r['loss']:.4f}")

n_runs = len(runs)
grid_minutes = round((time.time() - t_grid) / 60, 1)
worst_margin = round(min(unigram_loss - r["loss"] for r in runs), 3)
peak_vram_gb = 0.0
if device == "cuda":
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(tr(f"peak GPU memory {peak_vram_gb:.2f} GB", f"ذروة ذاكرة المعالج الرسومي {peak_vram_gb:.2f} غيغابايت"))
# --8<-- [end:grid]


# --8<-- [start:figures]
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, NullFormatter

if env.lang == "ar":
    # matplotlib draws Arabic as isolated letters in logical order. Reshape the
    # letters into their joined forms and reorder for display, and use a font
    # that actually contains Arabic glyphs (the default DejaVu does not).
    import arabic_reshaper
    from bidi.algorithm import get_display

    font_path = env.assets["NotoNaskhArabic-Regular.ttf"]
    font_manager.fontManager.addfont(str(font_path))
    arabic_font = font_manager.FontProperties(fname=str(font_path)).get_name()
    # Naskh has no Latin glyphs; DejaVu fills in digits, "C", "N" and symbols.
    plt.rcParams["font.family"] = [arabic_font, "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.unicode_minus"] = False


def lab(en, ar):
    """A figure label in the notebook's language, ready for matplotlib."""
    if env.lang != "ar":
        return en
    return get_display(arabic_reshaper.reshape(ar))


def log_axis(axis):
    """Log scale with plain tick text such as 3e12, which renders in either font."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}".replace("e+0", "e").replace("e+", "e")))
    axis.set_minor_formatter(NullFormatter())


budget_colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(budgets)))
print(tr("figure text ready", "نصوص الرسوم جاهزة"))
# --8<-- [end:figures]


# --8<-- [start:isoflop]
def isoflop_optima(runs, count="total"):
    """Locate the best model size at each budget.

    Fit a parabola in log N to the lowest-loss run and up to two neighbours on
    each side; its vertex is the optimum. The optimum counts as measured only if
    the best run has a neighbour on both sides, so the minimum is bracketed.

    `count` chooses which parameter count goes on the x-axis. The training runs
    are the same either way; only the bookkeeping changes.
    """
    key = "N" if count == "total" else "N_ne"
    optima = []
    for C in budgets:
        group = sorted((r for r in runs if r["C"] == C), key=lambda r: r[key])
        if len(group) < 3:
            continue
        best = int(np.argmin([r["loss"] for r in group]))
        lo = max(0, min(best - 2, len(group) - 3))
        window = group[lo:max(lo + 3, best + 3)]
        x = np.log([r[key] for r in window])
        y = np.array([r["loss"] for r in window])
        a2, a1, a0 = np.polyfit(x, y, 2)
        x_star = -a1 / (2 * a2) if a2 > 0 else float("nan")
        interior = bool(0 < best < len(group) - 1 and a2 > 0 and x.min() < x_star < x.max())
        n_opt = math.exp(x_star) if a2 > 0 else float("nan")
        # Tokens at the optimum: D = C / (6 N_total). Under the non-embedding
        # convention, map N_ne back to N_total across the fitted runs first.
        xt = np.log([r["N"] for r in window])
        n_total_opt = math.exp(np.interp(x_star, x, xt)) if a2 > 0 else float("nan")
        d_opt = C / (6 * n_total_opt)
        optima.append({"C": C, "C_count": 6 * n_opt * d_opt, "N_opt": n_opt, "D_opt": d_opt,
                       "L_opt": a0 + a1 * x_star + a2 * x_star ** 2 if a2 > 0 else float("nan"),
                       "interior": interior, "fit": (a2, a1, a0), "group": group,
                       "window": (x.min(), x.max())})
    return optima


def frontier_points(optima):
    """The optima a frontier may be fitted to, and whether there are enough."""
    good = [o for o in optima if o["interior"]]
    if len(good) >= 3:
        return good, True
    return [o for o in optima if math.isfinite(o["N_opt"])], False


optima = isoflop_optima(runs)
bracketed_share = round(sum(o["interior"] for o in optima) / len(budgets), 2)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
for o in optima:
    color = budget_colors[budgets.index(o["C"])]
    xs = np.array([r["N"] for r in o["group"]])
    ax.scatter(xs, [r["loss"] for r in o["group"]], color=color, s=28, zorder=3)
    grid_x = np.linspace(o["window"][0] - 0.1, o["window"][1] + 0.1, 100)
    ax.plot(np.exp(grid_x), np.polyval(o["fit"], grid_x), color=color, lw=1.4,
            label=f"C = {o['C']:.1e}")
    if o["interior"]:
        ax.scatter([o["N_opt"]], [o["L_opt"]], marker="*", s=160, color=color,
                   edgecolor="black", linewidth=0.6, zorder=4)
ax.set_xscale("log")
log_axis(ax.xaxis)
ax.set_xlabel(lab("parameters N (log scale)", "عدد المعاملات N (مقياس لوغاريتمي)"))
ax.set_ylabel(lab("validation loss (nats/byte)", "خسارة التحقق (نات لكل بايت)"))
ax.set_title(lab("IsoFLOP curves: same compute, different splits",
                 "منحنيات الحوسبة الثابتة: الحوسبة نفسها بتقسيمات مختلفة"))
ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.show()

for o in optima:
    if not math.isfinite(o["N_opt"]):
        print(f"C = {o['C']:.1e}:  " + tr("no minimum: loss only falls or only rises across these sizes",
                                         "لا يوجد قاع: الخسارة تنخفض فقط أو ترتفع فقط عبر هذه الأحجام"))
        continue
    flag = "" if o["interior"] else tr("  ← edge of the sampled range", "  ← على حافة المدى المُجرَّب")
    print(f"C = {o['C']:.1e}:  N_opt ≈ {o['N_opt']/1e3:,.0f}k   D_opt ≈ {o['D_opt']/1e6:.1f}M   "
          f"L ≈ {o['L_opt']:.4f}{flag}")
# --8<-- [end:isoflop]


# --8<-- [start:frontier]
def power_fit(xs, ys):
    """Fit y = k · x^p in log space. Returns (p, k, r2)."""
    lx, ly = np.log(xs), np.log(ys)
    p, logk = np.polyfit(lx, ly, 1)
    resid = ly - (p * lx + logk)
    r2 = 1 - (resid ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return float(p), float(math.exp(logk)), float(r2)


def loss_fit(cs, ls, with_floor=True):
    """Fit L(C) = E + A · C^(-alpha). With with_floor=False, E is fixed at zero.

    E is found by scanning: for each candidate floor, the rest is a straight line
    in log space, so the scan is exact and needs no optimizer.
    """
    cs, ls = np.asarray(cs), np.asarray(ls)
    floors = np.linspace(0.0, ls.min() * 0.995, 400) if with_floor else np.array([0.0])
    best = None
    for e in floors:
        slope, logA = np.polyfit(np.log(cs), np.log(ls - e), 1)
        sse = ((e + np.exp(logA) * cs ** slope - ls) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, float(e), float(math.exp(logA)), float(-slope))
    return best[1], best[2], best[3]          # E, A, alpha


good, enough_interior = frontier_points(optima)
if not enough_interior:
    # A minimum outside the sampled sizes is an extrapolation of a parabola, not
    # a measurement. Fall back so the cell still runs, but the check will fail.
    print(tr("⚠ fewer than three budgets have a bracketed optimum; widen `ratios`",
             "⚠ أقلّ من ثلاث ميزانيات لها حجم أمثل محصور بين تجربتين؛ وسّع قائمة ratios"))
C_fit = np.array([o["C"] for o in good])
exponent_a, k_N, frontier_r2 = power_fit(C_fit, [o["N_opt"] for o in good])
exponent_b, k_D, _ = power_fit(C_fit, [o["D_opt"] for o in good])
fitted_E, fit_A, alpha_c = loss_fit(C_fit, [o["L_opt"] for o in good])
tokens_per_param = round(good[-1]["D_opt"] / good[-1]["N_opt"], 1)
a_measured, E_frontier, frontier_r2 = round(exponent_a, 3), round(fitted_E, 3), round(frontier_r2, 3)
if not enough_interior:
    frontier_r2 = 0.0    # a line through extrapolated minima is not a measured fit

print(tr(
    f"N_opt ∝ C^{exponent_a:.3f}   (R² {frontier_r2:.3f})\n"
    f"D_opt ∝ C^{exponent_b:.3f}\n"
    f"L_opt(C) = {fitted_E:.3f} + {fit_A:.3g} · C^-{alpha_c:.3f}\n"
    f"tokens per parameter at the largest fitted budget: {tokens_per_param:.1f}",
    f"N_opt ∝ C^{exponent_a:.3f}   (R² {frontier_r2:.3f})\n"
    f"D_opt ∝ C^{exponent_b:.3f}\n"
    f"L_opt(C) = {fitted_E:.3f} + {fit_A:.3g} · C^-{alpha_c:.3f}\n"
    f"عدد الرموز لكل معامل عند أكبر ميزانية: {tokens_per_param:.1f}",
))
print(tr("Chinchilla reported a ≈ 0.50 and roughly 20 tokens per parameter.",
         "للمقارنة: وجدت ورقة Chinchilla أنّ a ≈ 0.50، ونحو 20 رمزاً لكل معامل."))
# --8<-- [end:frontier]


# --8<-- [start:predict]
# Commit to a number BEFORE training: the budget is beyond every fitted point.
C_holdout = budgets[-1] * cfg["holdout_factor"]
N_pred = k_N * C_holdout ** exponent_a
L_pred = fitted_E + fit_A * C_holdout ** (-alpha_c)

# Use the shape closest to the predicted optimum that the data can feed.
candidates = sorted(ladder, key=lambda s: abs(math.log(sizes[s]["total"] / N_pred)))
holdout_shape = next(s for s in candidates
                     if C_holdout / (6 * sizes[s]["total"]) <= max_train_tokens)
holdout_params = sizes[holdout_shape]["total"]
holdout_tokens_m = C_holdout / (6 * holdout_params) / 1e6
predicted_loss = round(L_pred, 4)
holdout_tokens_m = round(holdout_tokens_m, 1)

print(tr(
    f"holdout budget {C_holdout:.1e} FLOPs ({cfg['holdout_factor']}× the largest fitted budget)\n"
    f"predicted optimum N ≈ {N_pred/1e3:,.0f}k → nearest shape "
    f"{holdout_shape} with {holdout_params:,} params, {holdout_tokens_m:.1f}M tokens\n"
    f"predicted loss: {predicted_loss:.4f} nats/byte",
    f"ميزانية الاختبار {C_holdout:.1e} FLOPs (أي {cfg['holdout_factor']} أضعاف أكبر ميزانية في التوفيق)\n"
    f"الحجم الأمثل المتوقَّع N ≈ {N_pred/1e3:,.0f}k ← أقرب حجم متاح "
    f"{holdout_shape} بعدد {holdout_params:,} معاملاً و{holdout_tokens_m:.1f} مليون رمز\n"
    f"الخسارة المتوقَّعة: {predicted_loss:.4f} نات لكل بايت",
))
# --8<-- [end:predict]


# --8<-- [start:holdout]
t0 = time.time()
measured_loss, holdout_D = train_run(holdout_shape, C_holdout / (6 * holdout_params), cfg["seed"] + 999)
prediction_error_pct = round(100 * abs(L_pred - measured_loss) / measured_loss, 2)
holdout_minutes = round((time.time() - t0) / 60, 1)

print(tr(
    f"predicted {predicted_loss:.4f}   measured {measured_loss:.4f}   "
    f"error {prediction_error_pct:.2f}%   ({holdout_minutes:.1f} min)",
    f"المتوقَّع {predicted_loss:.4f}   المقيس {measured_loss:.4f}   "
    f"الخطأ {prediction_error_pct:.2f}%   ({holdout_minutes:.1f} دقيقة)",
))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
cs = np.geomspace(C_fit.min() / 1.5, C_holdout * 1.5, 100)
ax1.scatter(C_fit, [o["N_opt"] for o in good], color="#3b5bdb", zorder=3,
            label=lab("IsoFLOP optima", "أمثل حجم لكل ميزانية"))
ax1.plot(cs, k_N * cs ** exponent_a, color="#3b5bdb", lw=1.2, ls="--",
         label=f"N ∝ C^{exponent_a:.2f}")
ax1.scatter([C_holdout], [holdout_params], marker="D", color="#e8590c", zorder=4,
            label=lab("holdout model", "نموذج الاختبار"))
ax1.set_xscale("log")
ax1.set_yscale("log")
log_axis(ax1.xaxis)
log_axis(ax1.yaxis)
ax1.set_xlabel(lab("compute C (FLOPs)", "الحوسبة C بوحدات FLOPs"))
ax1.set_ylabel(lab("compute-optimal N", "الحجم الأمثل N"))
ax1.legend(fontsize=8, frameon=False)

ax2.scatter(C_fit, [o["L_opt"] for o in good], color="#3b5bdb", zorder=3,
            label=lab("fitted points", "نقاط التوفيق"))
ax2.plot(cs, fitted_E + fit_A * cs ** (-alpha_c), color="#3b5bdb", lw=1.2, ls="--",
         label=lab("extrapolated frontier", "الحدّ المُستقرَأ"))
ax2.scatter([C_holdout], [measured_loss], marker="D", color="#e8590c", zorder=4,
            label=lab("measured", "المقيس"))
ax2.axhline(fitted_E, color="grey", lw=0.8, ls=":")
ax2.set_xscale("log")
log_axis(ax2.xaxis)
ax2.set_xlabel(lab("compute C (FLOPs)", "الحوسبة C بوحدات FLOPs"))
ax2.set_ylabel(lab("loss (nats/byte)", "الخسارة (نات لكل بايت)"))
ax2.legend(fontsize=8, frameon=False)
fig.suptitle(lab("Fitted on small budgets, tested on a larger one",
                 "توفيق على ميزانيات صغيرة، واختبار على ميزانية أكبر"))
plt.tight_layout()
plt.show()
# --8<-- [end:holdout]


# --8<-- [start:parametric]
# Chinchilla's third approach: fit one surface L(N, D) = E + A/N^alpha + B/D^beta
# to every run at once, then read the allocation exponent off it.
import itertools

def fit_surface(runs):
    logN = torch.tensor([math.log(r["N"]) for r in runs], dtype=torch.float64)
    logD = torch.tensor([math.log(r["D"]) for r in runs], dtype=torch.float64)
    logL = torch.tensor([math.log(r["loss"]) for r in runs], dtype=torch.float64)
    def objective(th):
        # log L = log(exp(a - alpha·logN) + exp(b - beta·logD) + exp(e)), Huber in log space,
        # exactly as in the Chinchilla paper.
        pred = torch.logsumexp(torch.stack([th[0] - th[3] * logN, th[1] - th[4] * logD,
                                            th[2].expand_as(logN)]), dim=0)
        return F.huber_loss(pred, logL, delta=1e-3, reduction="sum")

    best = None
    # A small grid of starting points; the fit has several local minima.
    for a0, b0, e0, al0, be0 in itertools.product((0.0, 5.0, 10.0), (0.0, 5.0, 10.0),
                                                  (-1.0, 0.0, 0.5), (0.2, 0.5, 0.8),
                                                  (0.2, 0.5, 0.8)):
        th = torch.tensor([a0, b0, e0, al0, be0], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th], max_iter=200, line_search_fn="strong_wolfe")

        def closure(th=th, opt=opt):
            opt.zero_grad()
            loss = objective(th)
            loss.backward()
            return loss

        try:
            obj = opt.step(closure).item()
        except RuntimeError:
            continue
        if math.isfinite(obj) and (best is None or obj < best[0]):
            best = (obj, th.detach().clone())
    a, b, e, alpha, beta = best[1].tolist()
    return {"A": math.exp(a), "B": math.exp(b), "E": math.exp(e), "alpha": alpha, "beta": beta}


surface = fit_surface(runs)
param_exponent_a = round(surface["beta"] / (surface["alpha"] + surface["beta"]), 3)
param_E = round(surface["E"], 3)
surface_pred = round(surface["E"] + surface["A"] / holdout_params ** surface["alpha"]
                     + surface["B"] / holdout_D ** surface["beta"], 4)

print(tr(
    f"L(N, D) = {surface['E']:.3f} + {surface['A']:.3g}/N^{surface['alpha']:.3f} "
    f"+ {surface['B']:.3g}/D^{surface['beta']:.3f}\n"
    f"allocation exponent from the surface: a = β/(α+β) = {param_exponent_a:.3f}"
    f"   (IsoFLOP method: {exponent_a:.3f})\n"
    f"irreducible loss E: surface {param_E:.3f} · frontier {fitted_E:.3f}\n"
    f"holdout: surface predicts {surface_pred:.4f}, measured {measured_loss:.4f}",
    f"L(N, D) = {surface['E']:.3f} + {surface['A']:.3g}/N^{surface['alpha']:.3f} "
    f"+ {surface['B']:.3g}/D^{surface['beta']:.3f}\n"
    f"أُسّ التوزيع من السطح: a = β/(α+β) = {param_exponent_a:.3f}"
    f"   (بطريقة الحوسبة الثابتة: {exponent_a:.3f})\n"
    f"الخسارة غير القابلة للاختزال E: من السطح {param_E:.3f} · من الحدّ {fitted_E:.3f}\n"
    f"نموذج الاختبار: يتوقّع السطح {surface_pred:.4f}، والمقيس {measured_loss:.4f}",
))
# --8<-- [end:parametric]


# --8<-- [start:exercise_floor]
# YOUR TURN — the floor.
#
# The frontier fit above includes an irreducible loss E: a level no amount of
# compute gets below. This cell refits WITHOUT it, as a pure power law
# L = A · C^(-alpha), on exactly the same points. Before you run it, guess:
# will the pure power law predict a holdout loss that is too high or too low?
# Then set WITH_FLOOR = True and confirm you get the frontier fit back.
WITH_FLOOR = False

E_try, A_try, alpha_try = loss_fit(C_fit, [o["L_opt"] for o in good], with_floor=WITH_FLOOR)
floor_pred = round(E_try + A_try * C_holdout ** (-alpha_try), 4)
floor_error_pct = round(100 * (floor_pred - measured_loss) / measured_loss, 2)

in_range = np.array([E_try + A_try * c ** (-alpha_try) for c in C_fit])
in_range_rmse = float(np.sqrt(((in_range - np.array([o["L_opt"] for o in good])) ** 2).mean()))
print(tr(
    f"with_floor={WITH_FLOOR}: E={E_try:.3f}, alpha={alpha_try:.3f}\n"
    f"  error on the fitted points (RMSE): {in_range_rmse:.4f}\n"
    f"  holdout prediction {floor_pred:.4f} vs measured {measured_loss:.4f}"
    f"  → signed error {floor_error_pct:+.2f}%",
    f"with_floor={WITH_FLOOR}: E={E_try:.3f}, alpha={alpha_try:.3f}\n"
    f"  الخطأ على نقاط التوفيق (RMSE): {in_range_rmse:.4f}\n"
    f"  التوقّع لنموذج الاختبار {floor_pred:.4f} مقابل المقيس {measured_loss:.4f}"
    f"  ← الخطأ بإشارته {floor_error_pct:+.2f}%",
))
# --8<-- [end:exercise_floor]


# --8<-- [start:exercise_count]
# YOUR TURN — the bookkeeping.
#
# Kaplan et al. counted parameters WITHOUT the embedding tables; Chinchilla
# counted all of them. This cell refits the frontier with the Kaplan count,
# from the very same training runs. Nothing is retrained. Only the number you
# write next to each model changes. Set COUNT = "total" to get the original back.
COUNT = "non_embedding"

recount, _ = frontier_points(isoflop_optima(runs, count=COUNT))
count_exponent_a, _, count_r2 = power_fit([o["C_count"] for o in recount],
                                          [o["N_opt"] for o in recount])
count_exponent_a = round(count_exponent_a, 3)
count_tokens_per_param = round(recount[-1]["D_opt"] / recount[-1]["N_opt"], 1)
print(tr(
    f"count={COUNT}:  N_opt ∝ C^{count_exponent_a:.3f}  (R² {count_r2:.3f})   "
    f"tokens/param at the largest budget {count_tokens_per_param:.1f}\n"
    f"count=total:  N_opt ∝ C^{exponent_a:.3f}   tokens/param {tokens_per_param:.1f}",
    f"count={COUNT}:  N_opt ∝ C^{count_exponent_a:.3f}  (R² {count_r2:.3f})   "
    f"رموز لكل معامل عند أكبر ميزانية {count_tokens_per_param:.1f}\n"
    f"count=total:  N_opt ∝ C^{exponent_a:.3f}   رموز لكل معامل {tokens_per_param:.1f}",
))
# --8<-- [end:exercise_count]


# --8<-- [start:verify]
learned_ok = env.check("runs-learned", worst_margin)
bracketed_ok = env.check("optima-bracketed", bracketed_share)
frontier_ok = env.check("frontier-fit", frontier_r2)
prediction_ok = env.check("prediction-error", prediction_error_pct)
# --8<-- [end:verify]


# --8<-- [start:finish]
receipt = env.receipt()
# --8<-- [end:finish]
