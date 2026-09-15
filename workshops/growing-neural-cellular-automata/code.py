# workshops/growing-neural-cellular-automata/code.py
#
# The executable spine. One file, runnable start to finish, cut into regions
# that the notebook builder lifts into cells. A `cell` or `exercise` block in
# workshop.yaml names one with `ref:`.
#
# TWO RULES ARE TRAINED HERE, and the entire argument rests on them differing
# in exactly one respect. Same architecture, same initial weights, same
# optimizer, same step budget, same loss, same target. The only difference is
# WHICH STATES each batch starts from. If any other difference creeps in, the
# comparison stops being about the sampling scheme and the workshop is
# measuring something it does not name.
#
# Everything the run is sized by comes from env.cfg. Identifiers are ASCII in
# both language builds; only comments and printed strings are localized, at
# runtime, via env.lang.

# --8<-- [start:setup]
import azimuth_nb as azimuth

env = azimuth.setup(SLUG, lang=LANG, profile=PROFILE)
# --8<-- [end:setup]


# --8<-- [start:target]
import matplotlib.pyplot as plt
import numpy as np
import torch
import arabic_reshaper
from bidi.algorithm import get_display

GRID = env.cfg["grid"]

# The target is DRAWN, not downloaded. That keeps the maintenance tier stable:
# there is no URL to rot, no licence to track, and no asset to pin. It also
# means the shape can be resized with the profile instead of being resampled.


def make_target(grid: int) -> np.ndarray:
    """An asymmetric flower, as premultiplied RGBA in (4, grid, grid).

    Asymmetric ON PURPOSE. A radially symmetric target can be rebuilt from any
    surviving wedge, so a damage run against one would look like regeneration
    while only testing symmetry. Here the leaf sits left of the stem and
    nowhere else, so erasing the left half removes information that has to be
    reconstructed rather than copied from across an axis.
    """
    rgba = np.zeros((4, grid, grid), dtype=np.float32)
    yy, xx = np.mgrid[0:grid, 0:grid].astype(np.float32)
    cx, cy = (grid - 1) / 2.0, grid * 0.36

    def disc(px, py, r):
        return ((xx - px) ** 2 + (yy - py) ** 2) <= r * r

    petal_r, ring_r = grid * 0.145, grid * 0.20
    petals = np.zeros((grid, grid), dtype=bool)
    for k in range(6):
        angle = 2 * np.pi * k / 6
        petals |= disc(cx + ring_r * np.cos(angle), cy + ring_r * np.sin(angle), petal_r)
    centre = disc(cx, cy, grid * 0.115)

    stem = (np.abs(xx - cx) <= grid * 0.035) & (yy >= cy + grid * 0.10) & (yy <= grid * 0.86)
    lx, ly = cx - grid * 0.17, grid * 0.68
    leaf = (((xx - lx) / (grid * 0.14)) ** 2 + ((yy - ly) / (grid * 0.062)) ** 2) <= 1.0

    # Painted back to front, so the stem never cuts through the yellow centre.
    for mask, colour in (
        (stem, (0.20, 0.56, 0.28)),
        (leaf, (0.24, 0.66, 0.32)),
        (petals & ~centre, (0.86, 0.24, 0.36)),
        (centre, (0.98, 0.78, 0.18)),
    ):
        for channel in range(3):
            rgba[channel][mask] = colour[channel]
        rgba[3][mask] = 1.0

    rgba[:3] *= rgba[3]  # premultiplied: RGB is only meaningful where alpha is
    return rgba


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target = torch.from_numpy(make_target(GRID)).to(device)

# THE DENOMINATOR OF EVERY SCORE BELOW, and the reason the scores mean
# anything. An empty grid is not a neutral prediction: the target is mostly
# transparent background, so a rule that dies immediately scores well on plain
# pixel error. Dividing by the error an EMPTY grid makes pins the empty run at
# exactly zero and a perfect run at one.
empty_error = float((target**2).mean())


def match(state: torch.Tensor) -> float:
    """How much of the target a state accounts for. Empty grid 0, perfect 1.

    Deliberately not clipped below zero. A rule that has overshot into noise is
    WORSE than one that never grew, and flattening that to zero would hide the
    failure this workshop is built to show.
    """
    error = float(((state[:, :4] - target) ** 2).mean())
    return 1.0 - error / empty_error


def to_rgb(state: torch.Tensor) -> np.ndarray:
    """Composite premultiplied RGBA over white, for display."""
    rgba = state[0, :4].detach().cpu().numpy() if state.dim() == 4 else state[:4].cpu().numpy()
    rgb, alpha = rgba[:3], np.clip(rgba[3], 0, 1)
    return np.clip(1.0 - alpha + rgb, 0, 1).transpose(1, 2, 0)


alive_fraction = float((target[3] > 0.1).float().mean())

fig, axis = plt.subplots(figsize=(2.6, 2.6))
axis.imshow(to_rgb(target), interpolation="nearest")
axis.axis("off")
fig.tight_layout()
plt.show()

env.explain("morphogenesis")
if env.lang == "ar":
    print(f"الهدف {GRID}×{GRID} · {alive_fraction:.2f} من الخلايا حيّة")
    print(f"خطأ الشبكة الفارغة {empty_error:.4f} · عليه تُقسم كل درجات التطابق التالية")
else:
    print(f"target {GRID}x{GRID} · {alive_fraction:.2f} of cells alive")
    print(f"empty-grid error {empty_error:.4f} — the denominator of every match score below")
# --8<-- [end:target]


# --8<-- [start:rule]
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = env.cfg["channels"]
ALIVE = env.cfg["aliveThreshold"]

# Channels 0-3 are RGBA and are the only ones the loss ever sees. The rest are
# unconstrained scratch space: nothing tells the rule what to put there, and
# whatever it stores is the closest thing this system has to a signal passed
# between neighbours.
HIDDEN_CHANNELS = CHANNELS - 4

_IDENTITY = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
_PERCEPTION = torch.stack([_IDENTITY, _SOBEL_X, _SOBEL_X.t()])


class CellRule(nn.Module):
    """The whole model. One rule, shared by every cell on the grid.

    There is no global state, no coordinate input and no step counter. A cell
    sees its own channels and a 3x3 neighbourhood, and that is all. Everything
    the grid does over time has to be produced by this one function applied
    everywhere at once.
    """

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        # Fixed, not learned: identity plus two Sobel gradients per channel.
        # Registered as a buffer so it moves with .to(device) but never
        # receives a gradient.
        self.register_buffer("filters", _PERCEPTION.repeat(channels, 1, 1).unsqueeze(1))
        self.channels = channels
        self.dense1 = nn.Conv2d(channels * 3, hidden, 1)
        self.dense2 = nn.Conv2d(hidden, channels, 1, bias=False)
        # ZERO INIT on the output layer, so the rule starts as the identity
        # map: every cell initially does nothing. Growth then has to be learned
        # rather than unlearned from random noise, and the first few hundred
        # steps are not spent recovering from an explosion.
        nn.init.zeros_(self.dense2.weight)

    def alive(self, x: torch.Tensor) -> torch.Tensor:
        """A cell is alive if it or a neighbour has meaningful alpha."""
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > ALIVE

    def forward(self, x: torch.Tensor, fire_rate: float) -> torch.Tensor:
        pre_alive = self.alive(x)
        perception = F.conv2d(x, self.filters, padding=1, groups=self.channels)
        dx = self.dense2(F.relu(self.dense1(perception)))
        # Stochastic update: each cell acts independently with probability
        # fire_rate. Without it the grid updates in lockstep, which is a global
        # clock — exactly the kind of coordination this model is supposed to do
        # without.
        fire = (torch.rand_like(x[:, :1]) <= fire_rate).float()
        x = x + dx * fire
        return x * (pre_alive & self.alive(x)).float()


def make_seed(n: int) -> torch.Tensor:
    """n grids, empty except for one cell at the centre with every channel set."""
    seed = torch.zeros(n, CHANNELS, GRID, GRID, device=device)
    seed[:, 3:, GRID // 2, GRID // 2] = 1.0
    return seed


torch.manual_seed(env.cfg["seed"])
reference = CellRule(CHANNELS, env.cfg["hidden"]).to(device)
param_count = sum(w.numel() for w in reference.parameters())

env.explain("update rule")
if env.lang == "ar":
    # Arabic number agreement changes between 3-10 and 11+, and both profiles
    # are in play here, so the counts are phrased to sidestep it entirely
    # rather than being correct on the free tier and wrong on the paper one.
    print(f"{param_count:,} معامل · قاعدة واحدة تشترك فيها كل خلية")
    print(f"لكل خلية {CHANNELS} قناة: أربع منها RGBA، وبقيتها حرّة")
else:
    print(f"{param_count:,} parameters — one rule, shared by every cell")
    print(f"{CHANNELS} channels per cell: 4 are RGBA, {HIDDEN_CHANNELS} are unconstrained")
# --8<-- [end:rule]


# --8<-- [start:train]
import time

POOL_SIZE = env.cfg["poolSize"]
BATCH = env.cfg["batch"]


def erase_disc(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Zero a random disc. Used inside training, and only on the pool rule."""
    radius = env.cfg["damageRadius"] * GRID
    cy, cx = rng.uniform(0.2, 0.8, size=2) * GRID
    yy, xx = torch.meshgrid(
        torch.arange(GRID, device=device, dtype=torch.float32),
        torch.arange(GRID, device=device, dtype=torch.float32),
        indexing="ij",
    )
    keep = (((xx - cx) ** 2 + (yy - cy) ** 2) > radius**2).float()
    return x * keep


def train_rule(use_pool: bool, tag: str) -> tuple[CellRule, list[float]]:
    """Train one rule. `use_pool` is THE ONLY DIFFERENCE between the two runs.

    Both see the same loss, the same architecture, the same initial weights
    (manual_seed is reset here), the same optimizer and the same step budget.
    With use_pool False, every batch starts from a fresh seed and the rule is
    never asked about any state it has not been walked to from scratch. With
    use_pool True, batches start from states the rule itself produced on
    earlier training steps — including damaged ones.
    """
    torch.manual_seed(env.cfg["seed"])
    model = CellRule(CHANNELS, env.cfg["hidden"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=env.cfg["learningRate"])
    rng = np.random.default_rng(env.cfg["seed"])
    pool = make_seed(POOL_SIZE)
    losses: list[float] = []

    for step in range(1, env.cfg["trainSteps"] + 1):
        if use_pool:
            index = torch.from_numpy(rng.integers(0, POOL_SIZE, BATCH)).to(device)
            batch = pool[index].clone()
            # Rank by how wrong each sampled state already is. The worst is
            # replaced with a fresh seed, so the rule never stops being asked
            # to grow from nothing; the best few are damaged, so it is also
            # asked about states no growth trajectory would ever visit.
            with torch.no_grad():
                per_sample = ((batch[:, :4] - target) ** 2).mean(dim=(1, 2, 3))
            order = torch.argsort(per_sample, descending=True)
            batch[order[0]] = make_seed(1)[0]
            for slot in order[-env.cfg["damagePerBatch"] :]:
                batch[slot] = erase_disc(batch[slot : slot + 1], rng)[0]
        else:
            batch = make_seed(BATCH)

        steps = int(rng.integers(env.cfg["minSteps"], env.cfg["maxSteps"] + 1))
        for _ in range(steps):
            batch = model(batch, env.cfg["fireRate"])

        loss = ((batch[:, :4] - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        # Gradient NORMALIZATION, not clipping. The loss is taken through a few
        # dozen applications of the same rule, so gradient magnitude varies
        # over orders of magnitude between steps; normalizing each parameter's
        # gradient to unit length makes the step size depend on the optimizer
        # instead of on how long this particular rollout happened to be.
        for param in model.parameters():
            if param.grad is not None:
                param.grad /= param.grad.norm() + 1e-8
        optimizer.step()

        if use_pool:
            pool[index] = batch.detach()

        losses.append(float(loss.detach()))
        if step == 1 or step % env.cfg["logEvery"] == 0:
            # Localized rather than left in English. This is the longest-running
            # cell in the workshop, so its log is most of what an Arabic reader
            # actually watches while the run is in progress.
            elapsed = time.time() - started
            if env.lang == "ar":
                print(f"{tag} · خطوة {step:5d} · خسارة {losses[-1]:.5f} · {elapsed:5.0f}ث")
            else:
                print(f"{tag}  step {step:5d}  loss {losses[-1]:.5f}  {elapsed:5.0f}s")

    return model, losses


started = time.time()
# Peak allocation is measured rather than guessed. A profile's vramGb should
# be what the run actually uses, never the capacity of whichever card was
# attached — preflight is a hard refusal, so an over-declared requirement
# locks out every smaller card for no reason.
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

# The tag is what distinguishes the two runs in a log that scrolls for
# minutes, so it is localized like everything else the reader looks at.
if env.lang == "ar":
    NAIVE_TAG, POOL_TAG = "من البذرة", "من المجمّع"
else:
    NAIVE_TAG, POOL_TAG = "seed-only", "pool     "

# The naive rule first, so its log lines cannot be mistaken for the other's.
naive_rule, naive_losses = train_rule(use_pool=False, tag=NAIVE_TAG)
pool_rule, pool_losses = train_rule(use_pool=True, tag=POOL_TAG)

naive_final_loss = float(np.mean(naive_losses[-20:]))
pool_final_loss = float(np.mean(pool_losses[-20:]))
train_seconds = round(time.time() - started, 1)
peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1) if device.type == "cuda" else 0.0

env.explain("sample pool")
if env.lang == "ar":
    print(
        f"\nالخسارة النهائية · من البذرة {naive_final_loss:.5f} · من المجمّع {pool_final_loss:.5f}"
    )
    print(f"استغرق تدريب القاعدتين معاً {train_seconds:.0f} ثانية")
    if peak_vram_mb:
        print(f"ذروة ذاكرة المعالج الرسومي {peak_vram_mb:.0f} ميغابايت")
else:
    print(f"\nfinal loss — seed-only {naive_final_loss:.5f} · pool {pool_final_loss:.5f}")
    print(f"{train_seconds:.0f}s to train both rules")
    if peak_vram_mb:
        print(f"peak VRAM {peak_vram_mb:.0f} MB")
# --8<-- [end:train]


# --8<-- [start:growth]
GROW_STEPS = env.cfg["growSteps"]
FRAMES = env.cfg["stripFrames"]


@torch.no_grad()
def roll_out(model: CellRule, state: torch.Tensor, steps: int, record=None):
    """Run the rule forward, optionally snapshotting at the given step numbers."""
    captured = {0: state.clone()} if record and 0 in record else {}
    for step in range(1, steps + 1):
        state = model(state, env.cfg["fireRate"])
        if record and step in record:
            captured[step] = state.clone()
    return state, captured


torch.manual_seed(env.cfg["seed"])
marks = sorted({GROW_STEPS * i // (FRAMES - 1) for i in range(FRAMES)})
grown, strip = roll_out(pool_rule, make_seed(1), GROW_STEPS, record=set(marks))
pool_match = match(grown)

fig, axes = plt.subplots(1, len(marks), figsize=(1.25 * len(marks), 1.7))
for axis, step in zip(axes, marks):
    axis.imshow(to_rgb(strip[step]), interpolation="nearest")
    axis.set_title(f"{step}", fontsize=8)
    axis.axis("off")
fig.tight_layout()
plt.show()

if env.lang == "ar":
    print(f"من خلية واحدة إلى تطابق {pool_match:.3f} خلال {GROW_STEPS} خطوة")
else:
    print(f"one cell to a match of {pool_match:.3f} in {GROW_STEPS} steps")
# --8<-- [end:growth]


# --8<-- [start:persistence]
PERSIST_STEPS = env.cfg["persistSteps"]

# Both rules run from the same seed, far past the horizon either was trained
# through. Nothing here is retrained: this is the same two rules, asked a
# question the loss never asked them.
PROBE_POINTS = 40
probe = sorted({PERSIST_STEPS * i // (PROBE_POINTS - 1) for i in range(PROBE_POINTS)})

torch.manual_seed(env.cfg["seed"])
_, naive_frames = roll_out(naive_rule, make_seed(1), PERSIST_STEPS, record=set(probe))
torch.manual_seed(env.cfg["seed"])
_, pool_frames = roll_out(pool_rule, make_seed(1), PERSIST_STEPS, record=set(probe))

naive_curve = [match(naive_frames[s]) for s in probe]
pool_curve = [match(pool_frames[s]) for s in probe]

# Measured at the horizon, where the two rules have diverged as far as they
# are going to. A DIFFERENCE, not a ratio: both scores can go negative when a
# rule overshoots into noise, and a ratio over signed quantities reads as
# failure exactly when the gap is largest.
naive_match_long = naive_curve[-1]
pool_match_long = pool_curve[-1]
persistence_gap = pool_match_long - naive_match_long

# The naive rule at the step count it was TRAINED for. This is the control: if
# it never reached the flower, then it did not overshoot the flower, and the
# comparison below would be between two rules that both failed.
naive_trained_horizon = min(probe, key=lambda s: abs(s - env.cfg["maxSteps"]))
naive_match_trained = match(naive_frames[naive_trained_horizon])

fig, (curves, snaps) = plt.subplots(2, 1, figsize=(7, 4.6), gridspec_kw={"height_ratios": [2, 1]})
curves.plot(probe, naive_curve, color="#e76f51", label="seed-only")
curves.plot(probe, pool_curve, color="#2a9d8f", label="pool")
curves.axhline(0, color="#8d99ae", linestyle=":", linewidth=1)
curves.axvspan(0, env.cfg["maxSteps"], color="#8d99ae", alpha=0.12)
# The y axis is pinned to the empty-grid line at the bottom rather than
# autoscaled: "the pool curve is flat" is the claim, and an autoscaled axis
# would turn its noise into a trend.
curves.set_ylim(-0.05, 1.02)
curves.set_ylabel("match to target")
curves.set_xlabel("cellular automaton step")
curves.legend(loc="lower left")
curves.spines[["top", "right"]].set_visible(False)

snaps.axis("off")
for position, (frames, name) in enumerate(((naive_frames, "seed-only"), (pool_frames, "pool"))):
    for column, step in enumerate(probe[:: max(1, len(probe) // 5)][:5]):
        inset = snaps.inset_axes([column * 0.19 + 0.02, 0.5 - position * 0.5, 0.17, 0.46])
        inset.imshow(to_rgb(frames[step]), interpolation="nearest")
        inset.set_xticks([])
        inset.set_yticks([])
        if column == 0:
            inset.set_ylabel(name, fontsize=7)
fig.tight_layout()
plt.show()

env.explain("cellular automaton")
if env.lang == "ar":
    print(
        f"عند الخطوة {PERSIST_STEPS} · من البذرة {naive_match_long:.3f} "
        f"· من المجمّع {pool_match_long:.3f}"
    )
    print(f"الفارق بينهما {persistence_gap:.3f}")
else:
    print(f"at step {PERSIST_STEPS}: seed-only {naive_match_long:.3f} · pool {pool_match_long:.3f}")
    print(f"gap of {persistence_gap:.3f}")
# --8<-- [end:persistence]


# --8<-- [start:damage]
HEAL_STEPS = env.cfg["healSteps"]

def ar(text: str) -> str:
    """Join Arabic letters into their connected forms, then reorder for LTR drawing."""
    return get_display(arabic_reshaper.reshape(text))

def erase_half(state: torch.Tensor, side: str) -> torch.Tensor:
    """Zero one half of the grid, every channel — hidden ones included.

    Zeroing only RGBA would leave the rule's own signalling channels intact and
    the grid would rebuild from a scaffold that is still standing. Taking every
    channel is what makes this an ablation rather than a repaint.
    """
    cut = state.clone()
    half = GRID // 2
    if side == "left":
        cut[:, :, :, :half] = 0.0
    elif side == "right":
        cut[:, :, :, half:] = 0.0
    elif side == "top":
        cut[:, :, :half, :] = 0.0
    else:
        cut[:, :, half:, :] = 0.0
    return cut


torch.manual_seed(env.cfg["seed"])
stable, _ = roll_out(pool_rule, make_seed(1), GROW_STEPS)
stable_match = match(stable)

wounded = erase_half(stable, "left")
damaged_match = match(wounded)

healed, _ = roll_out(pool_rule, wounded, HEAL_STEPS)
healed_match = match(healed)
recovery = healed_match - damaged_match

fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
for axis, (state, label_en, label_ar) in zip(
    axes,
    (
        (target.unsqueeze(0), "target", "الهدف"),
        (stable, "grown", "بعد النمو"),
        (wounded, "half erased", "بعد المحو"),
        (healed, "healed", "بعد الترميم"),
    ),
):
    axis.imshow(to_rgb(state), interpolation="nearest")
    axis.set_title(label_ar if env.lang == "ar" else label_en, fontsize=9)
    axis.axis("off")
fig.tight_layout()
plt.show()

env.explain("regeneration")
if env.lang == "ar":
    print(
        f"بعد النمو {stable_match:.3f} · بعد المحو {damaged_match:.3f} "
        f"· بعد الترميم {healed_match:.3f}"
    )
    print(f"استعادت {recovery:.3f} خلال {HEAL_STEPS} خطوة")
else:
    print(f"grown {stable_match:.3f} · erased {damaged_match:.3f} · healed {healed_match:.3f}")
    print(f"recovered {recovery:.3f} in {HEAL_STEPS} steps")
# --8<-- [end:damage]


# --8<-- [start:choose_damage]
# YOUR TURN.
#
# Nothing retrains here. The rule is fixed; you are choosing what to do to a
# grown flower and watching the same rule respond.
#
# SIDE     "left" | "right" | "top" | "bottom"
# STEPS    how long to let it run afterwards
#
# The defaults sit away from the graded run on purpose — a knob whose default
# happens to reproduce the checked number teaches nothing about the knob.
#
# Two questions worth more than the number: does the LEAF come back on the
# correct side when you erase the side it was on, and is there a side whose
# removal the rule cannot come back from at all?
SIDE = "bottom"
STEPS = HEAL_STEPS

torch.manual_seed(env.cfg["seed"])
your_stable, _ = roll_out(pool_rule, make_seed(1), GROW_STEPS)
your_wounded = erase_half(your_stable, SIDE)
your_healed, _ = roll_out(pool_rule, your_wounded, STEPS)

fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
for axis, state in zip(axes, (your_stable, your_wounded, your_healed)):
    axis.imshow(to_rgb(state), interpolation="nearest")
    axis.axis("off")
fig.tight_layout()
plt.show()

if env.lang == "ar":
    # SIDE stays an ASCII identifier value, but dropping it raw into Arabic
    # prose reverses the reading order around it. Named here instead.
    sides_ar = {"left": "اليسار", "right": "اليمين", "top": "الأعلى", "bottom": "الأسفل"}
    print(
        f"محو {sides_ar.get(SIDE, SIDE)} · بعد المحو {match(your_wounded):.3f} "
        f"· بعد {STEPS} خطوة {match(your_healed):.3f}"
    )
else:
    print(
        f"erased {SIDE} · after damage {match(your_wounded):.3f} · after {STEPS} steps {match(your_healed):.3f}"
    )
# --8<-- [end:choose_damage]


# --8<-- [start:verify]
# The control goes FIRST and on purpose. If the seed-only rule never reached
# the flower at the horizon it was trained through, then it did not overshoot
# the flower either, and every comparison below would be between two rules that
# both simply failed to learn the task. If this check fails the answer is more
# training steps, never a lower bar.
naive_grows_ok = env.check("naive-grows", naive_match_trained)
target_match_ok = env.check("target-match", pool_match)
# Keyed on the pool rule's OWN match at the horizon, not on the gap between
# the two rules. Across two runs the gap moved 8.45 -> 0.63 while this number
# moved 0.81 -> 0.86: the gap's magnitude is mostly a measure of how badly the
# loser happens to fail on a given seed, which is not what the check is for.
# The gap is still captured and still quoted, as evidence rather than as a bar.
persists_ok = env.check("persists", pool_match_long)
recovers_ok = env.check("recovers", recovery)
# --8<-- [end:verify]


# --8<-- [start:finish]
receipt = env.receipt()
# --8<-- [end:finish]
