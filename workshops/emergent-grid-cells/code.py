# workshops/emergent-grid-cells/code.py
#
# A recurrent network learns to keep track of where it is from velocity alone.
# Two copies are trained on byte-identical trajectories from identical weights;
# they differ in one thing only: the shape of the place-cell code they must
# predict. The spatial firing maps of their hidden units are then scored for
# sixfold symmetry.
#
# Model and trajectory statistics follow Sorscher, Mel, Ganguli & Ocko (2019).

# --8<-- [start:setup]
import azimuth_nb as azimuth

env = azimuth.setup(SLUG, lang=LANG, profile=PROFILE)
# --8<-- [end:setup]


# --8<-- [start:arena]
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

assert torch.__version__, "torch comes with the runtime; it is never installed here"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = env.cfg
BOX = cfg["box_m"]
DT = 0.02  # seconds per step
torch.manual_seed(cfg["seed"])

# One fixed set of place-cell centres, shared by both networks.
centers = (
    torch.rand(cfg["place_cells"], 2, generator=torch.Generator().manual_seed(cfg["seed"])) - 0.5
) * BOX
centers = centers.to(device)


def make_trajectories(rng, batch, steps, box=BOX):
    """Smooth random walks that turn away from walls. Returns positions (batch, steps+1, 2)."""
    sigma_turn = 11.52  # rad/s, rotational velocity spread
    speed_scale = 0.13 * 2 * math.pi  # m/s, Rayleigh scale of forward speed
    border = 0.03
    pos = np.zeros((batch, steps + 1, 2))
    pos[:, 0] = rng.uniform(-box / 2, box / 2, (batch, 2))
    heading = rng.uniform(0, 2 * math.pi, batch)
    turns = rng.normal(0, sigma_turn, (batch, steps))
    speeds = rng.rayleigh(speed_scale, (batch, steps))
    for t in range(steps):
        x, y = pos[:, t, 0], pos[:, t, 1]
        dists = np.stack([box / 2 - x, box / 2 - y, box / 2 + x, box / 2 + y])
        wall_angle = dists.argmin(0) * math.pi / 2
        toward = np.mod(heading - wall_angle + math.pi, 2 * math.pi) - math.pi
        near = (dists.min(0) < border) & (np.abs(toward) < math.pi / 2)
        v = np.where(near, 0.25, 1.0) * speeds[:, t]
        heading = (
            heading
            + np.where(near, np.sign(toward) * (math.pi / 2 - np.abs(toward)), 0.0)
            + DT * turns[:, t]
        )
        pos[:, t + 1] = pos[:, t] + (v * DT)[:, None] * np.stack(
            [np.cos(heading), np.sin(heading)], -1
        )
    return pos


def place_code(pos, surround):
    """Population activity of the place cells at positions (..., 2). Sums to 1 over cells.

    surround=None gives plain Gaussian bumps. surround=s subtracts a wider bump
    (variance scaled by s): a centre that excites, ringed by a zone that inhibits.
    """
    d2 = ((pos[..., None, :] - centers) ** 2).sum(-1)
    width2 = cfg["place_sigma_m"] ** 2
    out = torch.softmax(-d2 / (2 * width2), -1)
    if surround is not None:
        out = out - torch.softmax(-d2 / (2 * surround * width2), -1)
        out = out - out.min(-1, keepdim=True).values
        out = out / out.sum(-1, keepdim=True)
    return out


# Picture the task: a few walks, and one place cell's tuning under each target.
rng = np.random.default_rng(cfg["seed"])
walks = make_trajectories(rng, 6, 200)
fig, (ax_walk, ax_tune) = plt.subplots(1, 2, figsize=(9, 4))
c = centers.cpu().numpy()
ax_walk.scatter(c[:, 0], c[:, 1], s=4, color="0.8")
for w in walks:
    ax_walk.plot(w[:, 0], w[:, 1], lw=1)
ax_walk.set_xlim(-BOX / 2, BOX / 2)
ax_walk.set_ylim(-BOX / 2, BOX / 2)
ax_walk.set_aspect("equal")

nearest = int((centers**2).sum(-1).argmin())  # the place cell closest to the middle
xs = torch.linspace(-BOX / 2, BOX / 2, 400, device=device)
line = torch.stack([xs, torch.full_like(xs, centers[nearest, 1].item())], -1)
for surround, colour in [(cfg["surround_scale"], "tab:orange"), (None, "tab:blue")]:
    tuning = place_code(line, surround)[:, nearest].cpu().numpy()
    ax_tune.plot(xs.cpu().numpy(), tuning / tuning.max(), color=colour, lw=2)
ax_tune.axhline(0, color="0.6", lw=0.8)
ax_tune.set_ylim(-0.3, 1.1)
plt.tight_layout()
plt.show()

if env.lang == "ar":
    hardware = "بطاقة رسوميات" if device.type == "cuda" else "المعالج المركزي"
    print(f"الساحة {BOX} م × {BOX} م · {cfg['place_cells']} خلية مكان · التشغيل على {hardware}")
else:
    print(f"arena {BOX} m × {BOX} m · {cfg['place_cells']} place cells · device: {device}")
# --8<-- [end:arena]


# --8<-- [start:model]
from torch import nn


class PathIntegrator(nn.Module):
    """Velocity in, place-cell prediction out. The hidden layer is never told what to be."""

    def __init__(self, n_place, n_hidden):
        super().__init__()
        self.encoder = nn.Linear(
            n_place, n_hidden, bias=False
        )  # starting place -> first hidden state
        self.rnn = nn.RNN(2, n_hidden, nonlinearity="relu", bias=False, batch_first=True)
        self.decoder = nn.Linear(n_hidden, n_place, bias=False)

    def hidden(self, velocity, start_code):
        states, _ = self.rnn(velocity, self.encoder(start_code)[None])
        return states

    def forward(self, velocity, start_code):
        return self.decoder(self.hidden(velocity, start_code))


def decode(logits):
    """Position estimate: the mean centre of the three most active predicted place cells."""
    return centers[logits.topk(3, dim=-1).indices].mean(-2)


def batch_tensors(pos, surround):
    pos = torch.as_tensor(pos, dtype=torch.float32, device=device)
    velocity = pos[:, 1:] - pos[:, :-1]
    return pos, velocity, place_code(pos[:, 0], surround), place_code(pos[:, 1:], surround)


params = sum(
    p.numel() for p in PathIntegrator(cfg["place_cells"], cfg["hidden_units"]).parameters()
)
if env.lang == "ar":
    print(f"{cfg['hidden_units']} وحدة مخفية · {params / 1e6:.1f} مليون معامل")
else:
    print(f"{cfg['hidden_units']} hidden units · {params / 1e6:.1f}M parameters")
# --8<-- [end:model]


# --8<-- [start:train_dog]
import time


def train(surround):
    torch.manual_seed(cfg["seed"])  # identical initial weights for both networks
    rng = np.random.default_rng(cfg["seed"])  # identical trajectories for both networks
    model = PathIntegrator(cfg["place_cells"], cfg["hidden_units"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    # Half precision for the recurrent arithmetic on a GPU. A T4 in full precision
    # ran 6.4 steps/s here, almost all of it matrix multiplication. The loss stays
    # in full precision, and the scaler keeps its very small gradients from
    # rounding to zero.
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, started = [], time.time()
    for step in range(1, cfg["train_steps"] + 1):
        pos, velocity, start, target = batch_tensors(
            make_trajectories(rng, cfg["batch"], cfg["seq_len"]), surround
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(velocity, start)
        logits = logits.float()
        loss = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()
        loss = loss + cfg["weight_decay"] * (model.rnn.weight_hh_l0**2).sum()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % cfg["log_every"] == 0 or step == cfg["train_steps"]:
            with torch.no_grad():
                err_cm = 100 * (decode(logits) - pos[:, 1:]).norm(dim=-1).mean().item()
            history.append((step, loss.item(), err_cm))
            rate = step / (time.time() - started)
            if env.lang == "ar":
                print(
                    f"خطوة {step:>6} · الخسارة {loss.item():.3f} · خطأ الموضع {err_cm:.1f} سم · {rate:.1f} خطوة/ث"
                )
            else:
                print(
                    f"step {step:>6} · loss {loss.item():.3f} · position error {err_cm:.1f} cm · {rate:.1f} steps/s"
                )
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    return model.eval(), np.array(history), time.time() - started, peak


dog_model, dog_history, dog_seconds, dog_peak_gb = train(cfg["surround_scale"])
train_minutes_dog = round(dog_seconds / 60, 1)
peak_vram_gb = round(dog_peak_gb, 2)
# --8<-- [end:train_dog]


# --8<-- [start:train_control]
control_model, control_history, control_seconds, _ = train(None)
train_minutes_control = round(control_seconds / 60, 1)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(dog_history[:, 0], dog_history[:, 2], color="tab:orange", lw=2)
ax.plot(control_history[:, 0], control_history[:, 2], color="tab:blue", lw=2)
ax.set_ylim(bottom=0)  # zero-based: both curves must visibly reach the floor
ax.set_xlabel("step")
ax.set_ylabel("cm")
plt.tight_layout()
plt.show()
# --8<-- [end:train_control]


# --8<-- [start:ratemaps]
@torch.no_grad()
def survey(model, surround, seq_len, skip=0):
    """Walk the trained network through fresh trajectories.

    Returns (rate maps [units, res, res], stability [units], decoding skill).
    Only steps at index >= skip are binned and scored. Skill = 1 - model error /
    error of a guess that never moves from the starting point, so 0 means "did
    not integrate". Stability correlates the maps built from two halves of the
    walks: a real map agrees with itself, noise does not.
    """
    res = cfg["map_res"]
    rng = np.random.default_rng(cfg["seed"] + 1)  # unseen trajectories, same for both networks
    sums = torch.zeros(2, res * res, cfg["hidden_units"], device=device)
    counts = torch.zeros(2, res * res, device=device)
    model_err, still_err = 0.0, 0.0
    for batch in range(cfg["eval_batches"]):
        pos, velocity, start, _ = batch_tensors(
            make_trajectories(rng, cfg["batch"], seq_len), surround
        )
        states = model.hidden(velocity, start)[:, skip:]
        where = pos[:, 1 + skip :]
        model_err += (decode(model.decoder(states)) - where).norm(dim=-1).mean().item()
        still_err += (pos[:, :1] - where).norm(dim=-1).mean().item()
        cell = ((where + BOX / 2) / BOX * res).long().clamp(0, res - 1)
        index = (cell[..., 0] * res + cell[..., 1]).reshape(-1)
        half = batch % 2
        sums[half].index_add_(0, index, states.reshape(-1, states.shape[-1]))
        counts[half].index_add_(0, index, torch.ones_like(index, dtype=torch.float32))
    maps = (sums.sum(0) / counts.sum(0).clamp(min=1)[:, None]).T.reshape(-1, res, res).cpu().numpy()
    halves = sums / counts.clamp(min=1)[..., None]  # (2, bins, units)
    seen = (counts > 0).all(0)
    a, b = halves[0][seen], halves[1][seen]
    a, b = a - a.mean(0), b - b.mean(0)
    stability = (
        ((a * b).sum(0) / ((a * a).sum(0) * (b * b).sum(0)).sqrt().clamp(min=1e-12)).cpu().numpy()
    )
    return maps, stability, 1 - model_err / still_err


dog_maps, dog_stability, dog_skill = survey(dog_model, cfg["surround_scale"], cfg["seq_len"])
control_maps, control_stability, control_skill = survey(control_model, None, cfg["seq_len"])
dog_skill, control_skill = round(dog_skill, 3), round(control_skill, 3)

if env.lang == "ar":
    print(
        f"مهارة تكامل المسار · هدف المركز والمحيط {dog_skill:.3f} · الهدف الغاوسي {control_skill:.3f}"
    )
else:
    print(
        f"path-integration skill · centre-surround {dog_skill:.3f} · Gaussian {control_skill:.3f}"
    )
# --8<-- [end:ratemaps]


# --8<-- [start:gridscore]
def autocorrelogram(maps):
    """Normalised spatial autocorrelation of each map, shape (units, 2*res-1, 2*res-1)."""
    n, res, _ = maps.shape
    size = 2 * res - 1
    x = np.zeros((n, size, size))
    x[:, :res, :res] = maps - maps.mean(axis=(1, 2), keepdims=True)
    ones = np.zeros((1, size, size))
    ones[:, :res, :res] = 1.0

    def xcorr(a, b):
        return np.fft.fftshift(
            np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b)))), axes=(1, 2)
        )

    overlap = np.round(xcorr(ones, ones))
    s_a, s_b = xcorr(x, ones), xcorr(ones, x)
    s_ab, s_aa, s_bb = xcorr(x, x), xcorr(x**2, ones), xcorr(ones, x**2)
    num = overlap * s_ab - s_a * s_b
    den = np.sqrt(
        np.clip(overlap * s_aa - s_a**2, 0, None) * np.clip(overlap * s_bb - s_b**2, 0, None)
    )
    sac = np.where((den > 1e-12) & (overlap >= 20), num / np.maximum(den, 1e-12), 0.0)
    return sac  # zero lag sits at index res-1 on both axes


def rotate(images, degrees):
    """Bilinear rotation about the centre, keeping the frame."""
    _, h, w = images.shape
    theta = math.radians(degrees)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    src_x = math.cos(theta) * (xx - cx) + math.sin(theta) * (yy - cy) + cx
    src_y = -math.sin(theta) * (xx - cx) + math.cos(theta) * (yy - cy) + cy
    x0, y0 = np.floor(src_x).astype(int), np.floor(src_y).astype(int)
    fx, fy = src_x - x0, src_y - y0
    out = np.zeros_like(images)
    for dy, dx, weight in [
        (0, 0, (1 - fx) * (1 - fy)),
        (0, 1, fx * (1 - fy)),
        (1, 0, (1 - fx) * fy),
        (1, 1, fx * fy),
    ]:
        yi, xi = y0 + dy, x0 + dx
        inside = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
        out += images[:, yi.clip(0, h - 1), xi.clip(0, w - 1)] * (weight * inside)
    return out


def grid_scores(maps):
    """Sixfold symmetry of each map: min(r60, r120) - max(r30, r90, r150), best over ring sizes."""
    sac = autocorrelogram(maps)
    n, size, _ = sac.shape
    res = maps.shape[1]
    r = np.hypot(*(np.mgrid[0:size, 0:size] - (size - 1) / 2))
    rotated = {a: rotate(sac, a).reshape(n, -1) for a in (30, 60, 90, 120, 150)}
    flat = sac.reshape(n, -1)
    best = np.full(n, -np.inf)
    # Rings stop at ring_max of the map width. Beyond it two copies of the map
    # barely overlap, the autocorrelogram is noise, and in the first T4 run that
    # noise gave single blobs scores above 1.1. The cap keeps full sensitivity to
    # lattices up to 0.7 of the box apart.
    for outer in np.linspace(0.4, cfg["ring_max"], 10):
        ring = ((r >= 0.2 * res) & (r <= outer * res)).reshape(-1)
        a = flat[:, ring] - flat[:, ring].mean(1, keepdims=True)
        corr = {}
        for angle, rot in rotated.items():
            b = rot[:, ring] - rot[:, ring].mean(1, keepdims=True)
            corr[angle] = (a * b).sum(1) / np.sqrt((a**2).sum(1) * (b**2).sum(1) + 1e-12)
        score = np.minimum(corr[60], corr[120]) - np.maximum(
            np.maximum(corr[30], corr[90]), corr[150]
        )
        best = np.maximum(best, score)
    return best, sac


def show_maps(maps, scores, count, cmap):
    order = np.argsort(-scores)[:count]  # callers pass unstable units as -inf
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    _, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.95 * rows))
    for ax, unit in zip(axes.ravel()[: len(order)], order, strict=True):
        ax.imshow(maps[unit].T, origin="lower", cmap=cmap, interpolation="gaussian")
        ax.set_title(f"{scores[unit]:.2f}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    return order


def grid_units(maps, scores, stability):
    """A grid unit scores above the cut AND draws the same map from both halves of the walks.

    Returns (mask over units, percent of active units)."""
    active = maps.std(axis=(1, 2)) > 1e-6  # silent units have no map to score
    mask = active & (scores > cfg["grid_score_cut"]) & (stability > cfg["stability_min"])
    return mask, round(100 * float(mask[active].mean()), 1)


# The null. In the same network before training, maps are speckle, and speckle
# can score above 1 by pure chance: on T4 runs about 15% of untrained units
# cleared a score of 0.3, as many as in the trained network. Speckle cannot
# repeat itself across two halves of the data, so the stability test removes it.
torch.manual_seed(cfg["seed"])
untrained = PathIntegrator(cfg["place_cells"], cfg["hidden_units"]).to(device).eval()
untrained_maps, untrained_stability, _ = survey(untrained, cfg["surround_scale"], cfg["seq_len"])
_, grid_percent_untrained = grid_units(
    untrained_maps, grid_scores(untrained_maps)[0], untrained_stability
)

dog_scores, dog_sac = grid_scores(dog_maps)
control_scores, _ = grid_scores(control_maps)
dog_grid, grid_percent_dog = grid_units(dog_maps, dog_scores, dog_stability)
control_grid, grid_percent_control = grid_units(control_maps, control_scores, control_stability)
active_dog = dog_maps.std(axis=(1, 2)) > 1e-6
active_control = control_maps.std(axis=(1, 2)) > 1e-6
grid_advantage_points = round(grid_percent_dog - grid_percent_control, 1)
cut = cfg["grid_score_cut"]
best_grid_score = round(float(dog_scores.max()), 2)

stable_dog = np.where(dog_stability > cfg["stability_min"], dog_scores, -np.inf)
top_dog_units = show_maps(dog_maps, stable_dog, cfg["units_shown"], "inferno")

if env.lang == "ar":
    print(
        f"وحدات سداسية مستقرة · {grid_percent_dog}% بعد التدريب · {grid_percent_untrained}% بالأوزان نفسها قبله"
    )
else:
    print(
        f"stable grid units · trained {grid_percent_dog}% · same weights before training {grid_percent_untrained}%"
    )
# --8<-- [end:gridscore]


# --8<-- [start:control_maps]
stable_control = np.where(control_stability > cfg["stability_min"], control_scores, -np.inf)
top_control_units = show_maps(control_maps, stable_control, cfg["units_shown"], "viridis")

fig, (ax_hist, ax_sac) = plt.subplots(1, 2, figsize=(9, 3.8))
bins = np.linspace(-1.0, 1.8, 57)
ax_hist.hist(control_scores[active_control], bins=bins, color="tab:blue", alpha=0.55, density=True)
ax_hist.hist(dog_scores[active_dog], bins=bins, color="tab:orange", alpha=0.55, density=True)
ax_hist.axvline(cut, color="0.2", ls="--", lw=1)
# The exemplar is the best unit that fires over a real part of the floor: a unit
# with two or three tiny spots can score well and still draw an unreadable picture.
coverage = (dog_maps > 0.5 * dog_maps.max(axis=(1, 2), keepdims=True)).mean(axis=(1, 2))
exemplar = next((u for u in top_dog_units if coverage[u] >= 0.15), top_dog_units[0])
ax_sac.imshow(dog_sac[exemplar].T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
ax_sac.axis("off")
plt.tight_layout()
plt.show()

if env.lang == "ar":
    print(
        f"وحدات سداسية مستقرة · هدف المركز والمحيط {grid_percent_dog}% · الهدف الغاوسي {grid_percent_control}%"
    )
else:
    print(
        f"stable grid units · centre-surround {grid_percent_dog}% · Gaussian {grid_percent_control}%"
    )
# --8<-- [end:control_maps]


# --8<-- [start:exercise]
# YOUR TURN.
# The network only ever saw walks of cfg["seq_len"] steps. Run it for longer and
# score only the steps it was never trained on. Try 1, then 3, 5, 10.
LENGTH_MULTIPLE = 5

long_len = cfg["seq_len"] * LENGTH_MULTIPLE
long_maps, _, long_skill = survey(
    dog_model, cfg["surround_scale"], long_len, skip=cfg["seq_len"] if LENGTH_MULTIPLE > 1 else 0
)
long_scores, _ = grid_scores(long_maps)
kept = [float(long_scores[u]) for u in top_dog_units]

fig, axes = plt.subplots(2, 8, figsize=(14, 3.8))
for i, unit in enumerate(top_dog_units[:8]):
    axes[0, i].imshow(dog_maps[unit].T, origin="lower", cmap="inferno", interpolation="gaussian")
    axes[1, i].imshow(long_maps[unit].T, origin="lower", cmap="inferno", interpolation="gaussian")
    axes[0, i].set_title(f"{dog_scores[unit]:.2f}", fontsize=9)
    axes[1, i].set_title(f"{long_scores[unit]:.2f}", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()

if env.lang == "ar":
    print(
        f"مسارات أطول بـ {LENGTH_MULTIPLE} مرات · مهارة التكامل {long_skill:.3f} (كانت {dog_skill:.3f})"
    )
    print(
        f"متوسط درجة الوحدات نفسها {np.mean(kept):.2f} (كان {np.mean(dog_scores[top_dog_units]):.2f})"
    )
else:
    print(
        f"walks {LENGTH_MULTIPLE}× longer · integration skill {long_skill:.3f} (was {dog_skill:.3f})"
    )
    print(
        f"mean score of the same units {np.mean(kept):.2f} (was {np.mean(dog_scores[top_dog_units]):.2f})"
    )
# --8<-- [end:exercise]


# --8<-- [start:verify]
# Control first: a grid comparison between networks that cannot find their way
# would be a comparison between two kinds of noise. A grid unit must both score
# above the cut and reproduce its map from independent halves of the walks.
integrates_ok = env.check("both-integrate", min(dog_skill, control_skill))
grids_ok = env.check("grid-units", grid_percent_dog)
advantage_ok = env.check("surround-advantage", grid_advantage_points)
# --8<-- [end:verify]


# --8<-- [start:finish]
receipt = env.receipt()
# --8<-- [end:finish]
