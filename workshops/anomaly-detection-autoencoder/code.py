# workshops/anomaly-detection-autoencoder/code.py
#
# THE EXECUTABLE SPINE. One file, runnable start to finish as a script, cut
# into regions that the notebook builder lifts into cells.
#
# Why a real .py and not cells in the YAML:
#   * ruff and black run on it, so formatting is not a review conversation
#   * it can be executed directly — `python code.py` — which is how a bug gets
#     found in ten seconds instead of through a notebook kernel
#   * a diff on the code is a diff on code, not on YAML string escaping
#
# Regions are marked with 8<-- start/end pairs naming the region, and a `cell` block
# in workshop.yaml names one with `ref:`. Every region must be referenced and
# every ref must resolve — build-workshops.mjs asserts both, in both
# directions, so a region cannot be silently orphaned by an edit to the prose.
#
# Identifiers are ASCII in both language builds. Only comments and printed
# strings are localized, and they are localized by `env.lang` at runtime rather
# than by generating two different programs — one program, two voices, so the
# Arabic notebook can never drift into being a different workshop.

# --8<-- [start:setup]
import azimuth_nb as azimuth

env = azimuth.setup(SLUG, lang=LANG, profile=PROFILE)
# --8<-- [end:setup]


# --8<-- [start:load_data]
import numpy as np

raw = np.loadtxt(env.assets["ecg5000.csv"], delimiter=",")
traces, labels = raw[:, :-1].astype(np.float32), raw[:, -1].astype(int)

# Min-max to [0, 1] using TRAINING statistics only. Fitting the scaler on
# everything would leak the abnormal range into the normal model — a quiet
# mistake that inflates every number downstream.
rng = np.random.default_rng(env.cfg["seed"])
order = rng.permutation(len(traces))
traces, labels = traces[order], labels[order]

split = int(0.8 * len(traces))
train_all, test_x = traces[:split], traces[split:]
train_labels, test_y = labels[:split], labels[split:]

# THE METHOD, IN ONE LINE: the model only ever sees normal traces.
train_x = train_all[train_labels == 1]

lo, hi = train_x.min(), train_x.max()
train_x = (train_x - lo) / (hi - lo)
test_x = (test_x - lo) / (hi - lo)

n_normal = int((test_y == 1).sum())
n_abnormal = int((test_y == 0).sum())
class_balance = {
    "trainNormal": len(train_x),
    "testNormal": n_normal,
    "testAbnormal": n_abnormal,
}

if env.lang == "ar":
    print(f"التدريب: {len(train_x)} أثراً طبيعياً فقط")
    print(f"الاختبار: {n_normal} طبيعي · {n_abnormal} شاذ")
else:
    print(f"train: {len(train_x)} normal traces only")
    print(f"test:  {n_normal} normal · {n_abnormal} abnormal")
# --8<-- [end:load_data]


# --8<-- [start:model]
import torch
import torch.nn as nn

torch.manual_seed(env.cfg["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_features = train_x.shape[1]


class Autoencoder(nn.Module):
    """Symmetric encoder/decoder around a deliberately narrow layer.

    The bottleneck width is the entire experiment. Widen it and the model
    learns the identity function and reconstructs abnormal traces just as well
    as normal ones — at which point there is no detector left, only a copier.
    """

    def __init__(self, n_features: int, hidden: list[int], latent: int):
        super().__init__()
        widths = [n_features, *hidden]

        encoder: list[nn.Module] = []
        for a, b in zip(widths[:-1], widths[1:]):
            encoder += [nn.Linear(a, b), nn.ReLU()]
        encoder += [nn.Linear(widths[-1], latent), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = []
        rev = [latent, *hidden[::-1]]
        for a, b in zip(rev[:-1], rev[1:]):
            decoder += [nn.Linear(a, b), nn.ReLU()]
        # Sigmoid because the inputs were scaled to [0, 1]: the output range
        # should be able to reach the input range and no further.
        decoder += [nn.Linear(rev[-1], n_features), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(self.encoder(x))


model = Autoencoder(n_features, list(env.cfg["hidden"]), env.cfg["latentDim"]).to(device)
param_count = sum(p.numel() for p in model.parameters())

env.explain("bottleneck")
if env.lang == "ar":
    print(f"عنق الزجاجة: {env.cfg['latentDim']} من أصل {n_features} بُعداً · {param_count:,} معامل")
else:
    print(f"bottleneck: {env.cfg['latentDim']} of {n_features} dims · {param_count:,} parameters")
# --8<-- [end:model]


# --8<-- [start:train]
from torch.utils.data import DataLoader, TensorDataset

train_t = torch.from_numpy(train_x).to(device)
loader = DataLoader(
    TensorDataset(train_t, train_t),
    batch_size=env.cfg["batchSize"],
    shuffle=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=env.cfg["learningRate"])
criterion = nn.MSELoss()

loss_curve = []
model.train()
for epoch in range(env.cfg["epochs"]):
    total = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch_x)
    epoch_loss = total / len(train_t)
    loss_curve.append(epoch_loss)
    if epoch % 10 == 0 or epoch == env.cfg["epochs"] - 1:
        print(f"epoch {epoch:3d}  loss {epoch_loss:.5f}")

final_loss = loss_curve[-1]
# --8<-- [end:train]


# --8<-- [start:error_distributions]
import matplotlib.pyplot as plt


def reconstruction_error(x: np.ndarray) -> np.ndarray:
    """Mean absolute error per trace — one number for each recording.

    L1 rather than L2 on purpose: squared error lets a single badly-missed
    sample dominate a trace's score, which makes the threshold sensitive to
    noise rather than to shape.
    """
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(x).to(device)
        return torch.mean(torch.abs(model(t) - t), dim=1).cpu().numpy()


train_errors = reconstruction_error(train_x)
test_errors = reconstruction_error(test_x)

normal_errors = test_errors[test_y == 1]
abnormal_errors = test_errors[test_y == 0]
separation = float(abnormal_errors.mean() / normal_errors.mean())

env.explain("reconstruction error")
fig, ax = plt.subplots(figsize=(7, 3.6))
bins = np.linspace(0, float(np.percentile(test_errors, 99.5)), 60)
ax.hist(normal_errors, bins=bins, alpha=0.75, label="normal", color="#2a9d8f")
ax.hist(abnormal_errors, bins=bins, alpha=0.75, label="abnormal", color="#e76f51")
ax.set_xlabel("reconstruction error (MAE)")
ax.set_ylabel("traces")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

print(f"normal mean   {normal_errors.mean():.4f}")
print(f"abnormal mean {abnormal_errors.mean():.4f}")
separation_ok = env.check("separation", separation)
# --8<-- [end:error_distributions]


# --8<-- [start:choose_threshold]
# YOUR TURN.
#
# Pick a percentile of `train_errors` — the errors on traces the model was
# trained on. Anything you can compute from this array is available on the day
# you deploy, with no abnormal examples in hand. Anything you compute from
# `abnormal_errors` is not.
#
# Start here and change the number, with a reason:
PERCENTILE = 95

threshold = float(np.percentile(train_errors, PERCENTILE))
print(f"threshold = {threshold:.4f}  (p{PERCENTILE} of training error)")
# --8<-- [end:choose_threshold]


# --8<-- [start:evaluate]
predicted_abnormal = test_errors > threshold
actually_abnormal = test_y == 0

tp = int((predicted_abnormal & actually_abnormal).sum())
fp = int((predicted_abnormal & ~actually_abnormal).sum())
fn = int((~predicted_abnormal & actually_abnormal).sum())
tn = int((~predicted_abnormal & ~actually_abnormal).sum())

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
confusion = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

if env.lang == "ar":
    print(f"الدقة {precision:.3f} · الاستدعاء {recall:.3f} · F1 {f1:.3f}")
    print(f"أخطأ في {fn} حالة شاذة، وأنذر زوراً في {fp} حالة طبيعية")
else:
    print(f"precision {precision:.3f} · recall {recall:.3f} · F1 {f1:.3f}")
    print(f"missed {fn} abnormal · false-alarmed on {fp} normal")

f1_ok = env.check("f1", f1)
# --8<-- [end:evaluate]


# --8<-- [start:baseline]
from sklearn.ensemble import IsolationForest

forest = IsolationForest(
    n_estimators=100,
    contamination=n_abnormal / len(test_y),
    random_state=env.cfg["seed"],
)
forest.fit(train_x)
forest_abnormal = forest.predict(test_x) == -1

b_tp = int((forest_abnormal & actually_abnormal).sum())
b_fp = int((forest_abnormal & ~actually_abnormal).sum())
b_fn = int((~forest_abnormal & actually_abnormal).sum())
b_precision = b_tp / (b_tp + b_fp) if (b_tp + b_fp) else 0.0
b_recall = b_tp / (b_tp + b_fn) if (b_tp + b_fn) else 0.0
baseline_f1 = (
    2 * b_precision * b_recall / (b_precision + b_recall) if (b_precision + b_recall) else 0.0
)

print(f"isolation forest F1 {baseline_f1:.3f}   ·   autoencoder F1 {f1:.3f}")
# --8<-- [end:baseline]


# --8<-- [start:finish]
receipt = env.receipt()
# --8<-- [end:finish]
