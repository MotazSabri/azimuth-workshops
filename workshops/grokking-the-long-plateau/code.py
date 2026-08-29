# workshops/grokking-the-long-plateau/code.py
#
# The executable spine. One file, runnable start to finish, cut into regions
# that the notebook builder lifts into cells. Regions are marked with 8<--
# start/end pairs naming the region, and a `cell` block in workshop.yaml names
# one with `ref:`.
#
# Everything the run is sized by comes from env.cfg. There is exactly one
# training run here and it is long, so the temptation to hardcode "just the
# step count" while iterating is real — and a step count that disagrees with
# the profile is how the prose ends up describing a run nobody executed.
#
# Identifiers are ASCII in both language builds; only comments and printed
# strings are localized, at runtime, via env.lang.

# --8<-- [start:setup]
import azimuth_nb as azimuth

env = azimuth.setup(SLUG, lang=LANG, profile=PROFILE)
# --8<-- [end:setup]


# --8<-- [start:data]
import numpy as np

p = env.cfg["modulus"]

# Every pair the task admits, enumerated. This is what makes an algorithmic
# dataset useful here: "unseen" is exact rather than approximate, so the
# held-out number means what it says.
a_all, b_all = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
a_all, b_all = a_all.ravel(), b_all.ravel()

# Token p is the "=" that separates the operands from the answer position.
# Vocabulary is p operand tokens plus that one; the output layer predicts over
# the p possible answers only .
EQUALS = p
inputs = np.stack([a_all, b_all, np.full_like(a_all, EQUALS)], axis=1)
targets = (a_all + b_all) % p

rng = np.random.default_rng(env.cfg["seed"])

# THE SPLIT IS OVER UNORDERED PAIRS, NOT OVER ROWS.
#
# Addition commutes, so (a, b) and (b, a) have the same answer. Splitting the
# 3481 rows at random puts one ordering in training and the other in the
# held-out set for a large fraction of them, and a model that has merely
# MEMORISED the training row can answer its transpose. The first run of this
# workshop did exactly that: the held-out curve rose immediately to a plateau
# and sat there for thousands of steps, which read as partial generalization
# and was not.
#
# Grouping by the unordered pair and assigning whole groups keeps both
# orderings on the same side of the split. The `leakage` cell measures that it
# worked rather than assuming it.
lo, hi = np.minimum(a_all, b_all), np.maximum(a_all, b_all)
group = lo * p + hi

groups = rng.permutation(np.unique(group))
n_train_groups = round(env.cfg["trainFrac"] * len(groups))
train_groups = set(groups[:n_train_groups].tolist())
is_train = np.array([g in train_groups for g in group.tolist()])

train_in, test_in = inputs[is_train], inputs[~is_train]
train_out, test_out = targets[is_train], targets[~is_train]

n_train, n_test = len(train_in), len(test_in)

# Chance is one over the modulus, NOT zero. A held-out curve resting just
# above the axis is a model guessing uniformly, and mistaking that floor for
# zero makes the eventual jump look larger than it is.
chance = 1.0 / p

if env.lang == "ar":
    print(f"القياس {p} · {len(inputs)} زوجاً ممكناً")
    print(f"تدريب {n_train} · اختبار {n_test}")
    print(f"مستوى الصدفة {chance:.4f}")
else:
    print(f"modulus {p} · {len(inputs)} possible pairs")
    print(f"train {n_train} · test {n_test}")
    print(f"chance level {chance:.4f}")
# --8<-- [end:data]


# --8<-- [start:leakage]
# What the split gave away. Set arithmetic only — nothing trains here.
#
# `leaked_pairs` is how many held-out rows could be answered by transposing a
# row the model was trained on. It must be zero. `naive_leaked` is what a
# random split over rows would have handed over instead, and it is the height
# of the floor the held-out curve would have rested on: not chance, and not a
# result.
train_rows = set(zip(train_in[:, 0].tolist(), train_in[:, 1].tolist()))
test_rows = list(zip(test_in[:, 0].tolist(), test_in[:, 1].tolist()))
leaked_pairs = sum(1 for x, y in test_rows if (y, x) in train_rows)

naive_order = np.random.default_rng(env.cfg["seed"]).permutation(len(inputs))
naive_cut = int(env.cfg["trainFrac"] * len(inputs))
naive_train = set(
    zip(
        inputs[naive_order[:naive_cut], 0].tolist(),
        inputs[naive_order[:naive_cut], 1].tolist(),
    )
)
naive_test = list(
    zip(
        inputs[naive_order[naive_cut:], 0].tolist(),
        inputs[naive_order[naive_cut:], 1].tolist(),
    )
)
naive_leaked = sum(1 for x, y in naive_test if (y, x) in naive_train) / len(naive_test)

env.explain("leakage")
if env.lang == "ar":
    print(f"أزواج محجوزة يمكن الإجابة عنها بالتبديل: {leaked_pairs}")
    print(f"القسمة الساذجة على الصفوف كانت لتسلّم {naive_leaked:.3f} منها")
else:
    print(f"held-out pairs answerable by transposition: {leaked_pairs}")
    print(f"a naive split over rows would have handed over {naive_leaked:.3f} of them")

split_ok = env.check("split-is-clean", leaked_pairs)
# --8<-- [end:leakage]


# --8<-- [start:model]
import torch
import torch.nn as nn

torch.manual_seed(env.cfg["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 3


class GrokFormer(nn.Module):
    """One pre-norm transformer block, reading the answer off the last position.

    Deliberately the smallest thing that groks rather than a scaled-down copy
    of anything. Depth is not the variable here — the training run is.
    """

    def __init__(self, vocab: int, d_model: int, n_heads: int, d_ff: int, n_answers: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model) * 0.02)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.unembed = nn.Linear(d_model, n_answers, bias=False)

    def forward(self, x):
        h = self.tok(x) + self.pos
        normed = self.ln1(h)
        h = h + self.attn(normed, normed, normed, need_weights=False)[0]
        h = h + self.mlp(self.ln2(h))
        return self.unembed(h[:, -1])


model = GrokFormer(
    vocab=p + 1,
    d_model=env.cfg["dModel"],
    n_heads=env.cfg["nHeads"],
    d_ff=env.cfg["dFF"],
    n_answers=p,
).to(device)

param_count = sum(w.numel() for w in model.parameters())

env.explain("memorization")
if env.lang == "ar":
    print(f"{param_count:,} معامل مقابل {n_train} زوج تدريب")
else:
    print(f"{param_count:,} parameters against {n_train} training pairs")
# --8<-- [end:model]


# --8<-- [start:train]
import time

train_x = torch.from_numpy(train_in).long().to(device)
train_y = torch.from_numpy(train_out).long().to(device)
test_x = torch.from_numpy(test_in).long().to(device)
test_y = torch.from_numpy(test_out).long().to(device)

# Full batch. The dataset fits in memory many times over, and a fixed batch
# removes sampling noise from an axis where a long flat line is the evidence.
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=env.cfg["learningRate"],
    weight_decay=env.cfg["weightDecay"],
    betas=(0.9, 0.98),
)
criterion = nn.CrossEntropyLoss()


def accuracy(x, y) -> float:
    model.eval()
    with torch.no_grad():
        return float((model(x).argmax(dim=-1) == y).float().mean())


steps_log: list[int] = []
train_acc_log: list[float] = []
test_acc_log: list[float] = []
train_loss_log: list[float] = []

# The two crossings, recorded as they happen rather than reconstructed later:
# the first step at which the model has memorised, and the first at which it
# has generalised. Both stay None if the crossing never occurs, which is a
# distinct state from "crossed at step 0" and must not be flattened into it.
memorise_step = None
generalise_step = None

started = time.time()
model.train()
for step in range(1, env.cfg["steps"] + 1):
    optimizer.zero_grad()
    loss = criterion(model(train_x), train_y)
    loss.backward()
    optimizer.step()
    # .detach() before the scalar conversion: float() on a tensor that still
    # tracks gradients emits a UserWarning, and warnings printed inside a
    # training loop end up on the published page next to the numbers.
    loss_value = loss.detach().item()

    if step == 1 or step % env.cfg["evalEvery"] == 0:
        train_acc = accuracy(train_x, train_y)
        test_acc = accuracy(test_x, test_y)
        model.train()

        steps_log.append(step)
        train_acc_log.append(train_acc)
        test_acc_log.append(test_acc)
        train_loss_log.append(loss_value)

        if memorise_step is None and train_acc >= env.cfg["memoriseAt"]:
            memorise_step = step
        if generalise_step is None and test_acc >= env.cfg["generaliseAt"]:
            generalise_step = step

        if step == 1 or step % env.cfg["logEvery"] == 0:
            elapsed = time.time() - started
            print(
                f"step {step:6d}  train {train_acc:.3f}  test {test_acc:.3f}  "
                f"loss {loss_value:.5f}  {elapsed:5.0f}s"
            )

peak_train_acc = max(train_acc_log)
final_test_acc = test_acc_log[-1]
# The PEAK held-out accuracy, not the last one. At this decay the run keeps
# taking optimization spikes after it has generalised: training accuracy drops
# to around 0.92 for an evaluation or two and the held-out figure falls with
# it. Two measured seeds peaked at 0.937 and 0.917 and both happened to end
# near 0.90, so a check on the final value is really a check on where the
# oscillation was standing when the budget ran out.
peak_test_acc = max(test_acc_log)

env.explain("weight decay")
if env.lang == "ar":
    print(
        f"أعلى دقة تدريب {peak_train_acc:.3f} · أعلى دقة محجوزة {peak_test_acc:.3f} "
        f"· الدقة النهائية {final_test_acc:.3f}"
    )
    print(f"حفظ عند {memorise_step} · عمّم عند {generalise_step}")
else:
    print(
        f"peak train {peak_train_acc:.3f} · peak unseen {peak_test_acc:.3f} "
        f"· final unseen {final_test_acc:.3f}"
    )
    print(f"memorised at {memorise_step} · generalised at {generalise_step}")
# --8<-- [end:train]


# --8<-- [start:curves]
import matplotlib.pyplot as plt

# A gap of 0 is what an incomplete run produces: either crossing missing means
# there is nothing to measure. Reporting 0 rather than skipping the number
# makes the `delay` check FAIL, which is the honest outcome — a run that never
# grokked should not certify a delay.
if memorise_step is not None and generalise_step is not None:
    grok_gap = generalise_step - memorise_step
else:
    grok_gap = 0

fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 5.4), sharex=True)

top.plot(steps_log, train_acc_log, label="train", color="#2a9d8f")
top.plot(steps_log, test_acc_log, label="held out", color="#e76f51")
top.axhline(chance, color="#8d99ae", linestyle=":", linewidth=1, label="chance")
# Zero-based: the long flat stretch is the argument, and an autoscaled axis
# would turn the noise on it into a trend.
top.set_ylim(0, 1.02)
top.set_ylabel("accuracy")
top.legend(loc="center left")
top.spines[["top", "right"]].set_visible(False)

bottom.plot(steps_log, train_loss_log, color="#264653")
bottom.set_yscale("log")
bottom.set_ylabel("train loss")
bottom.set_xlabel("optimizer step")
bottom.spines[["top", "right"]].set_visible(False)

for axis in (top, bottom):
    # Logarithmic in the step axis because the delay spans orders of
    # magnitude; on a linear axis the whole first plateau is one pixel wide.
    axis.set_xscale("log")
    if memorise_step is not None:
        axis.axvline(memorise_step, color="#2a9d8f", linewidth=1, alpha=0.5)
    if generalise_step is not None:
        axis.axvline(generalise_step, color="#e76f51", linewidth=1, alpha=0.5)

fig.tight_layout()
plt.show()

env.explain("grokking")
if env.lang == "ar":
    print(f"الفجوة {grok_gap} خطوة بين الحفظ والتعميم")
else:
    print(f"gap of {grok_gap} steps between memorising and generalising")
# --8<-- [end:curves]


# --8<-- [start:choose_stopping_rule]
# YOUR TURN.
#
# The ordinary rule: watch held-out accuracy, stop when it has not improved
# for PATIENCE consecutive evaluations. Nothing retrains — this replays the
# curve already logged above.
#
# Patience is counted in EVALUATIONS. Multiply by the eval interval to get the
# number of optimizer steps it actually buys you, which is usually smaller
# than it sounds.
PATIENCE = 20

best_acc = -1.0
stale = 0
stopped_at = steps_log[-1]
for step, acc in zip(steps_log, test_acc_log):
    if acc > best_acc:
        best_acc, stale = acc, 0
    else:
        stale += 1
        if stale >= PATIENCE:
            stopped_at = step
            break

patience_steps = PATIENCE * env.cfg["evalEvery"]
reached = generalise_step if generalise_step is not None else steps_log[-1]
discarded = max(0, reached - stopped_at)

if env.lang == "ar":
    print(f"صبر {PATIENCE} تقييماً = {patience_steps} خطوة")
    print(f"كانت القاعدة لتتوقف عند الخطوة {stopped_at} بدقة محجوزة {best_acc:.3f}")
    print(f"خطوات مهدورة قبل القفزة: {discarded}")
else:
    print(f"patience of {PATIENCE} evaluations = {patience_steps} steps")
    print(f"the rule would have stopped at step {stopped_at}, held-out {best_acc:.3f}")
    print(f"steps discarded before the jump: {discarded}")
# --8<-- [end:choose_stopping_rule]


# --8<-- [start:verify]
# The control goes first. Without memorisation there is no plateau, and a
# delay measured on a run that never fit the training set would be a number
# about nothing.
memorises_ok = env.check("memorises", peak_train_acc)
generalises_ok = env.check("generalises", peak_test_acc)
delay_ok = env.check("delay", grok_gap)
# --8<-- [end:verify]


# --8<-- [start:finish]
receipt = env.receipt()
# --8<-- [end:finish]
