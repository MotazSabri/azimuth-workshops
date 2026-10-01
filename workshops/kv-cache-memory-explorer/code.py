# workshops/kv-cache-memory-explorer/code.py
#
# Three memory bills for one decoder: weights, activations, KV cache.
# Every size comes from env.cfg. Weights are random: memory and time do not
# depend on what a model has learned, only on its shapes.


# --8<-- [start:setup]
import azimuth_nb as azimuth

env = azimuth.setup(SLUG, lang=LANG, profile=PROFILE)
# --8<-- [end:setup]


# --8<-- [start:model]
import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "This workshop measures GPU memory: switch the runtime to a GPU."
_tv = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
assert _tv >= (2, 1), (
    f"torch {torch.__version__} is too old for fused attention; use Colab's default runtime."
)

DEVICE = "cuda"
DTYPE = torch.float16
GIB = 1024**3

cfg = env.cfg
D_MODEL = cfg["d_model"]
N_LAYERS = cfg["n_layers"]
N_HEADS = cfg["n_heads"]
D_HEAD = D_MODEL // N_HEADS
FFN = cfg["ffn"]
VOCAB = cfg["vocab"]


def say(en, ar):
    print(ar if env.lang == "ar" else en)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def linear(n_in, n_out):
    return nn.Linear(n_in, n_out, bias=False, device=DEVICE, dtype=DTYPE)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=DEVICE, dtype=DTYPE))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return xf.to(x.dtype) * self.weight


class Attention(nn.Module):
    # n_kv_heads == N_HEADS is multi-head attention (MHA),
    # 1 < n_kv_heads < N_HEADS is grouped-query (GQA), n_kv_heads == 1 is multi-query (MQA).
    # Rotary position embeddings are left out: they change the values in K, not its size.
    def __init__(self, n_kv_heads):
        super().__init__()
        self.n_kv = n_kv_heads
        self.group = N_HEADS // n_kv_heads
        self.q = linear(D_MODEL, N_HEADS * D_HEAD)
        self.k = linear(D_MODEL, n_kv_heads * D_HEAD)
        self.v = linear(D_MODEL, n_kv_heads * D_HEAD)
        self.o = linear(N_HEADS * D_HEAD, D_MODEL)

    def prefill(self, x, naive):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, D_HEAD).transpose(1, 2).contiguous()
        v = self.v(x).view(B, T, self.n_kv, D_HEAD).transpose(1, 2).contiguous()
        # The cache keeps k and v with n_kv heads. The expanded copy below is
        # only borrowed for this one computation and is freed with it.
        kk, vv = k, v
        if self.group > 1:
            kk = k.repeat_interleave(self.group, dim=1)
            vv = v.repeat_interleave(self.group, dim=1)
        if naive:
            # Textbook attention: the full T x T score matrix exists in memory.
            scores = (q @ kk.transpose(-2, -1)) / math.sqrt(D_HEAD)
            mask = torch.ones(T, T, dtype=torch.bool, device=DEVICE).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
            out = scores.softmax(dim=-1) @ vv
        else:
            # Fused kernel (FlashAttention or memory-efficient, whichever the GPU
            # supports): the score matrix is computed in tiles and never stored.
            out = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, N_HEADS * D_HEAD)
        return self.o(out), (k, v)

    def decode(self, x, kv, pos):
        # One new token per sequence, attending over a cache of fixed length.
        # A fixed length stands in for a growing one: the step reads every
        # cached position either way.
        B = x.shape[0]
        k_cache, v_cache = kv
        q = self.q(x).view(B, self.n_kv, self.group, D_HEAD)
        slot = pos % k_cache.shape[2]
        k_cache[:, :, slot] = self.k(x).view(B, self.n_kv, D_HEAD)
        v_cache[:, :, slot] = self.v(x).view(B, self.n_kv, D_HEAD)
        # Grouped form: each cached head is read once and shared by its group
        # of query heads. Nothing is expanded.
        scores = (q @ k_cache.transpose(-2, -1)) / math.sqrt(D_HEAD)
        out = scores.softmax(dim=-1) @ v_cache
        return self.o(out.reshape(B, 1, N_HEADS * D_HEAD))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = linear(D_MODEL, FFN)
        self.up = linear(D_MODEL, FFN)
        self.down = linear(FFN, D_MODEL)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, n_kv_heads):
        super().__init__()
        self.norm1 = RMSNorm(D_MODEL)
        self.attn = Attention(n_kv_heads)
        self.norm2 = RMSNorm(D_MODEL)
        self.mlp = MLP()

    def prefill(self, x, naive):
        a, kv = self.attn.prefill(self.norm1(x), naive)
        x = x + a
        return x + self.mlp(self.norm2(x)), kv

    def decode(self, x, kv, pos):
        x = x + self.attn.decode(self.norm1(x), kv, pos)
        return x + self.mlp(self.norm2(x))


class Decoder(nn.Module):
    def __init__(self, n_kv_heads):
        super().__init__()
        self.n_kv = n_kv_heads
        self.embed = nn.Embedding(VOCAB, D_MODEL, device=DEVICE, dtype=DTYPE)
        self.blocks = nn.ModuleList(Block(n_kv_heads) for _ in range(N_LAYERS))
        self.norm = RMSNorm(D_MODEL)
        self.head = linear(D_MODEL, VOCAB)

    def prefill(self, ids, naive=False):
        x = self.embed(ids)
        cache = []
        for block in self.blocks:
            x, kv = block.prefill(x, naive)
            cache.append(kv)
        # Logits for the last position only: that is all generation needs.
        return cache, self.head(self.norm(x[:, -1:]))

    def decode(self, ids, cache, pos):
        x = self.embed(ids)
        for block, kv in zip(self.blocks, cache):
            x = block.decode(x, kv, pos)
        return self.head(self.norm(x))

    def filled_cache(self, batch, length):
        # A cache of the right shape for timing; its contents do not matter.
        shape = (batch, self.n_kv, length, D_HEAD)
        return [
            (
                torch.randn(shape, device=DEVICE, dtype=DTYPE),
                torch.randn(shape, device=DEVICE, dtype=DTYPE),
            )
            for _ in range(N_LAYERS)
        ]


def build(n_kv_heads):
    torch.manual_seed(cfg["seed"])
    model = Decoder(n_kv_heads)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0.0, 0.02)
    return model.eval().requires_grad_(False)


def weight_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def predicted_cache_bytes(n_layers, n_kv, d_head, batch, ctx, bytes_per_value=2):
    # 2 tensors (K and V) x layers x kv heads x head size x bytes, per token.
    return 2 * n_layers * n_kv * d_head * bytes_per_value * batch * ctx


free()
_before = torch.cuda.memory_allocated()
model = build(N_HEADS)
weights_measured = torch.cuda.memory_allocated() - _before
weights_gib = round(weight_bytes(model) / GIB, 2)
n_params_m = round(sum(p.numel() for p in model.parameters()) / 1e6)
kv_kib_per_token = round(predicted_cache_bytes(N_LAYERS, N_HEADS, D_HEAD, 1, 1) / 1024)

say(
    f"{n_params_m:,.0f}M parameters, {N_LAYERS} layers, {N_HEADS} heads of size {D_HEAD}\n"
    f"weights in fp16: {weights_gib:.2f} GiB (allocator saw {weights_measured / GIB:.2f} GiB)\n"
    f"cache per token, by the formula: {kv_kib_per_token:.0f} KiB",
    f"عدد المعاملات: {n_params_m:,.0f} مليون. الطبقات: {N_LAYERS}. رؤوس الانتباه: {N_HEADS}، وحجم كلٍّ منها {D_HEAD}\n"
    f"الأوزان بدقة fp16: {weights_gib:.2f} GiB (رصدها المُخصِّص {weights_measured / GIB:.2f} GiB)\n"
    f"الذاكرة المؤقتة لكل رمز، حسب المعادلة: {kv_kib_per_token:.0f} KiB",
)
# --8<-- [end:model]


# --8<-- [start:measure]
import pandas as pd


def measure_prefill(model, batch, ctx, naive=False):
    """Run one prefill and split memory into what stays and what passed through."""
    gen = torch.Generator(device=DEVICE).manual_seed(cfg["seed"])
    ids = torch.randint(0, VOCAB, (batch, ctx), device=DEVICE, generator=gen)
    free()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        cache, logits = model.prefill(ids, naive=naive)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    result = {
        # What the forward pass leaves behind, minus the last-token logits.
        "cache": after - before - logits.untyped_storage().nbytes(),
        # What existed only while the pass ran.
        "act": peak - after,
    }
    del cache, logits, ids
    free()
    return result


BATCH = cfg["batch"]
PROBE_CTX = cfg["probe_ctx"]
probe = measure_prefill(model, BATCH, PROBE_CTX)
probe_weights_gib = weights_gib
probe_cache_gib = round(probe["cache"] / GIB, 2)
probe_act_gib = round(probe["act"] / GIB, 2)

say(
    f"{BATCH} sequences x {PROBE_CTX} tokens, fp16",
    f"الدفعة: {BATCH} × {PROBE_CTX} (تسلسلات × رموز)، بدقة fp16",
)
if env.lang == "ar":
    probe_table = pd.DataFrame(
        {
            "الفاتورة": ["الأوزان", "التفعيلات (ذروة عابرة)", "ذاكرة المفاتيح والقيم"],
            "GiB": [probe_weights_gib, probe_act_gib, probe_cache_gib],
        }
    )
else:
    probe_table = pd.DataFrame(
        {
            "bill": ["weights", "activations (transient peak)", "KV cache"],
            "GiB": [probe_weights_gib, probe_act_gib, probe_cache_gib],
        }
    )
probe_table.round(2)
# --8<-- [end:measure]


# --8<-- [start:sweep]
import glob

import matplotlib.pyplot as plt
from matplotlib import font_manager

if env.lang == "ar":
    # matplotlib draws Arabic as isolated letters in logical order unless the
    # text is reshaped and reordered, and its default font has no Arabic glyphs.
    import arabic_reshaper
    from bidi.algorithm import get_display

    _fonts = sorted(glob.glob("/usr/share/fonts/**/NotoSansArabic-Regular.ttf", recursive=True))
    if _fonts:
        font_manager.fontManager.addfont(_fonts[0])
        _family = font_manager.FontProperties(fname=_fonts[0]).get_name()
        plt.rcParams["font.family"] = [_family, "DejaVu Sans"]
    else:
        print("تحذير: لم يُعثر على خط عربي؛ ستظهر النصوص العربية في الرسوم مربعات فارغة.")


def ar(en, ar_text):
    if env.lang == "ar":
        return get_display(arabic_reshaper.reshape(ar_text))
    return en


rows = []
for ctx in cfg["contexts"]:
    m = measure_prefill(model, BATCH, ctx)
    rows.append({"ctx": ctx, "cache": m["cache"], "act": m["act"]})

wb = weight_bytes(model)
sweep_max_ctx = rows[-1]["ctx"]
cache_gib_max = round(rows[-1]["cache"] / GIB, 2)
act_gib_max = round(rows[-1]["act"] / GIB, 2)
cache_over_weights = round(rows[-1]["cache"] / wb, 2)
# Measured bytes per token in flight, from the largest point.
cache_bytes_per_token = rows[-1]["cache"] / (BATCH * sweep_max_ctx)
crossover_tokens = round(wb / cache_bytes_per_token)
crossover_ctx = round(crossover_tokens / BATCH)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
xs = range(len(rows))
w = [wb / GIB] * len(rows)
c = [r["cache"] / GIB for r in rows]
a = [r["act"] / GIB for r in rows]
ax.bar(xs, w, color="#8a94a6", label=ar("weights", "الأوزان"))
ax.bar(xs, c, bottom=w, color="#d9822b", label=ar("KV cache", "ذاكرة المفاتيح والقيم"))
ax.bar(
    xs,
    a,
    bottom=[wi + ci for wi, ci in zip(w, c)],
    color="#5b8def",
    alpha=0.55,
    label=ar("activations (transient)", "التفعيلات (عابرة)"),
)
ax.axhline(2 * wb / GIB, color="#333", lw=0.8, ls=":")
ax.text(
    len(rows) - 0.5,
    2 * wb / GIB,
    ar(" cache = weights", " الذاكرة المؤقتة = الأوزان"),
    va="bottom",
    ha="right",
    fontsize=8,
)
ax.set_xticks(list(xs), [str(r["ctx"]) for r in rows])
ax.set_xlabel(ar(f"context length (batch of {BATCH})", f"طول السياق (دفعة من {BATCH})"))
ax.set_ylabel("GiB")
ax.set_ylim(bottom=0)
ax.set_title(ar("Where the memory goes during prefill", "أين تذهب الذاكرة أثناء معالجة الموجّه"))
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
plt.show()

say(
    f"At {sweep_max_ctx} tokens the cache is {cache_over_weights:.2f}x the weights.\n"
    f"It matches the weights at {crossover_tokens:,.0f} tokens in flight "
    f"({crossover_ctx:,.0f} tokens each across {BATCH} sequences).",
    f"عند سياق طوله {sweep_max_ctx}، حجم الذاكرة المؤقتة = {cache_over_weights:.2f} × حجم الأوزان.\n"
    f"نقطة التعادل مع الأوزان: {crossover_tokens:,.0f} رمزاً قيد المعالجة معاً، "
    f"أي سياق طوله {crossover_ctx:,.0f} لكل تسلسل في دفعة من {BATCH}.",
)
# --8<-- [end:sweep]


# --8<-- [start:formula]
predicted = [predicted_cache_bytes(N_LAYERS, N_HEADS, D_HEAD, BATCH, r["ctx"]) for r in rows]
formula_ratio = round(sum(r["cache"] for r in rows) / sum(predicted), 4)
say(
    f"measured cache / formula, summed over the sweep: {formula_ratio:.4f}",
    f"الذاكرة المقيسة ÷ المعادلة، مجموعةً على كل نقاط المسح: {formula_ratio:.4f}",
)

# The formula now has a measurement behind it, so it can be pointed at models
# this GPU could never load. Shapes are from the Llama 2 paper.
LLAMA2 = [
    # English name, Arabic name, parameters, layers, kv heads, head size
    ("Llama-2-7B (MHA)", "Llama-2-7B (MHA)", 6.74e9, 32, 32, 128),
    ("Llama-2-13B (MHA)", "Llama-2-13B (MHA)", 13.0e9, 40, 40, 128),
    ("Llama-2-70B (GQA, 8 kv heads)", "Llama-2-70B (GQA، 8 رؤوس KV)", 69.0e9, 80, 8, 128),
    ("Llama-2-70B if it were MHA", "Llama-2-70B لو كان MHA", 69.0e9, 80, 64, 128),
]
LLAMA_CTX = cfg["project_ctx"]
LLAMA_BATCH = cfg["project_batch"]
proj = []
for name_en, name_ar, params, layers, kv, dh in LLAMA2:
    name = name_ar if env.lang == "ar" else name_en
    per_token = predicted_cache_bytes(layers, kv, dh, 1, 1)
    weights = params * 2
    proj.append(
        {
            "name": name,
            "kib_per_token": per_token / 1024,
            "weights_gib": weights / GIB,
            "crossover_tokens": weights / per_token,
            "cache_gib_b8": per_token * LLAMA_BATCH * LLAMA_CTX / GIB,
        }
    )

llama7b_crossover_tokens = round(proj[0]["crossover_tokens"])
llama7b_cache_over_weights_b8 = round(proj[0]["cache_gib_b8"] / proj[0]["weights_gib"], 2)
llama70b_crossover_tokens = round(proj[2]["crossover_tokens"])
llama70b_mha_crossover_tokens = round(proj[3]["crossover_tokens"])

if env.lang == "ar":
    cols = {
        "name": "النموذج",
        "kib_per_token": "KiB لكل رمز",
        "weights_gib": "الأوزان GiB",
        "crossover_tokens": "رموز التعادل",
        "cache_gib_b8": f"الذاكرة المؤقتة GiB ({LLAMA_BATCH} × {LLAMA_CTX})",
    }
else:
    cols = {
        "name": "model",
        "kib_per_token": "KiB per token",
        "weights_gib": "weights GiB",
        "crossover_tokens": "tokens to match weights",
        "cache_gib_b8": f"cache GiB at {LLAMA_BATCH} x {LLAMA_CTX}",
    }
proj_table = pd.DataFrame(proj).rename(columns=cols).round(1)
proj_table
# --8<-- [end:formula]


# --8<-- [start:flash]
FB, FC = cfg["flash_batch"], cfg["flash_ctx"]
naive = measure_prefill(model, FB, FC, naive=True)
fused = measure_prefill(model, FB, FC, naive=False)

naive_act_gib = round(naive["act"] / GIB, 2)
fused_act_gib = round(fused["act"] / GIB, 2)
act_reduction = round(naive["act"] / fused["act"], 1)
cache_naive_gib = round(naive["cache"] / GIB, 2)
cache_fused_gib = round(fused["cache"] / GIB, 2)
score_matrix_gib = round(FB * N_HEADS * FC * FC * 2 / GIB, 2)

if env.lang == "ar":
    flash_table = pd.DataFrame(
        {
            "الانتباه": ["تقليدي (مصفوفة كاملة)", "مدمج، على شكل كتل"],
            "التفعيلات GiB": [naive_act_gib, fused_act_gib],
            "الذاكرة المؤقتة GiB": [cache_naive_gib, cache_fused_gib],
        }
    )
else:
    flash_table = pd.DataFrame(
        {
            "attention": ["naive (full matrix)", "fused, tiled kernel"],
            "activations GiB": [naive_act_gib, fused_act_gib],
            "KV cache GiB": [cache_naive_gib, cache_fused_gib],
        }
    )
say(
    f"{FB} x {FC} tokens. One layer's score matrix alone: {score_matrix_gib:.2f} GiB",
    f"الدفعة: {FB} × {FC} (تسلسلات × رموز). مصفوفة الدرجات لطبقة واحدة وحدها: {score_matrix_gib:.2f} GiB",
)
flash_table.round(2)
# --8<-- [end:flash]


# --8<-- [start:gqa]
del model
free()

variants = []
for kv in cfg["kv_variants"]:
    m_kv = build(kv)
    res = measure_prefill(m_kv, BATCH, PROBE_CTX)
    variants.append({"kv": kv, "weights": weight_bytes(m_kv), "cache": res["cache"]})
    del m_kv
    free()

by_kv = {v["kv"]: v for v in variants}
mha = by_kv[N_HEADS]
gqa_ratio = round(mha["cache"] / by_kv[cfg["gqa_kv"]]["cache"], 2)
mqa_ratio = round(mha["cache"] / by_kv[1]["cache"], 2)
gqa_weight_change_pct = round(
    100 * (mha["weights"] - by_kv[cfg["gqa_kv"]]["weights"]) / mha["weights"], 1
)

if env.lang == "ar":
    gqa_table = pd.DataFrame(
        {
            "رؤوس KV": [v["kv"] for v in variants],
            "الأوزان GiB": [v["weights"] / GIB for v in variants],
            "الذاكرة المؤقتة GiB": [v["cache"] / GIB for v in variants],
            "تقلّص الذاكرة المؤقتة": [mha["cache"] / v["cache"] for v in variants],
        }
    )
else:
    gqa_table = pd.DataFrame(
        {
            "kv heads": [v["kv"] for v in variants],
            "weights GiB": [v["weights"] / GIB for v in variants],
            "KV cache GiB": [v["cache"] / GIB for v in variants],
            "cache shrink vs MHA": [mha["cache"] / v["cache"] for v in variants],
        }
    )
gqa_table.round(2)
# --8<-- [end:gqa]


# --8<-- [start:int8]
def quantize(x):
    # Symmetric int8, one fp16 scale per (sequence, head, position).
    scale = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-8) / 127
    q = (x.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(DTYPE)


model = build(N_HEADS)
gen = torch.Generator(device=DEVICE).manual_seed(cfg["seed"])
ids = torch.randint(0, VOCAB, (BATCH, PROBE_CTX), device=DEVICE, generator=gen)
free()
base = torch.cuda.memory_allocated()
with torch.inference_mode():
    cache, logits = model.prefill(ids)
    base += logits.untyped_storage().nbytes()
    fp16_bytes = torch.cuda.memory_allocated() - base
    err_sq, ref_sq = 0.0, 0.0
    for i in range(len(cache)):
        k, v = cache[i]
        qk, sk = quantize(k)
        qv, sv = quantize(v)
        for orig, qx, sx in ((k, qk, sk), (v, qv, sv)):
            err_sq += (qx.float() * sx.float() - orig.float()).pow(2).sum().item()
            ref_sq += orig.float().pow(2).sum().item()
        cache[i] = (qk, sk, qv, sv)
        del k, v, qk, sk, qv, sv, orig, qx, sx
    free()
    int8_bytes = torch.cuda.memory_allocated() - base

int8_ratio = round(int8_bytes / fp16_bytes, 3)
int8_err_pct = round(100 * math.sqrt(err_sq / ref_sq), 2)
say(
    f"cache in fp16: {fp16_bytes / GIB:.2f} GiB   in int8 + scales: {int8_bytes / GIB:.2f} GiB   "
    f"ratio {int8_ratio:.3f}\nrelative reconstruction error: {int8_err_pct:.2f}%",
    f"الذاكرة المؤقتة بدقة fp16: {fp16_bytes / GIB:.2f} GiB   وبدقة int8 مع معاملات التحجيم: "
    f"{int8_bytes / GIB:.2f} GiB   النسبة {int8_ratio:.3f}\nخطأ إعادة البناء النسبي: {int8_err_pct:.2f}%",
)
del cache, logits, ids, model
free()
# --8<-- [end:int8]


# --8<-- [start:decode]
def time_decode(model, batch, ctx):
    cache = model.filled_cache(batch, ctx)
    ids = torch.zeros(batch, 1, dtype=torch.long, device=DEVICE)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for step in range(cfg["decode_warmup"]):
            model.decode(ids, cache, step)
        torch.cuda.synchronize()
        start.record()
        for step in range(cfg["decode_steps"]):
            model.decode(ids, cache, step)
        end.record()
        torch.cuda.synchronize()
    ms = start.elapsed_time(end) / cfg["decode_steps"]
    cache_b = sum(k.numel() * k.element_size() * 2 for k, _ in cache)
    del cache
    free()
    return ms, cache_b


DB = cfg["decode_batch"]
timings = {}
for kv in (N_HEADS, cfg["gqa_kv"]):
    m_kv = build(kv)
    wbytes = weight_bytes(m_kv)
    timings[kv] = []
    for ctx in cfg["decode_contexts"]:
        ms, cache_b = time_decode(m_kv, DB, ctx)
        timings[kv].append({"ctx": ctx, "ms": ms, "bytes": wbytes + cache_b})
    del m_kv
    free()

mha_t, gqa_t = timings[N_HEADS], timings[cfg["gqa_kv"]]
decode_ms_short = round(mha_t[0]["ms"], 1)
decode_ms_long = round(mha_t[-1]["ms"], 1)
decode_slowdown = round(mha_t[-1]["ms"] / mha_t[0]["ms"], 1)
decode_gqa_ms_long = round(gqa_t[-1]["ms"], 1)
# Bytes that must cross from memory to the cores per step, over the step time.
decode_gbps = round(mha_t[-1]["bytes"] / (mha_t[-1]["ms"] / 1000) / 1e9)

fig, ax = plt.subplots(figsize=(7.5, 4.0))
ctxs = [t["ctx"] for t in mha_t]
ax.plot(
    ctxs,
    [t["ms"] for t in mha_t],
    "o-",
    color="#d9822b",
    label=ar(f"MHA, {N_HEADS} kv heads", f"MHA، رؤوس KV: {N_HEADS}"),
)
ax.plot(
    ctxs,
    [t["ms"] for t in gqa_t],
    "s-",
    color="#2b9d8f",
    label=ar(f"GQA, {cfg['gqa_kv']} kv heads", f"GQA، رؤوس KV: {cfg['gqa_kv']}"),
)
ax.set_xscale("log", base=2)
ax.set_xticks(ctxs, [str(x) for x in ctxs])
ax.set_ylim(bottom=0)
ax.set_xlabel(
    ar(f"cached tokens per sequence (batch of {DB})", f"الرموز المخزّنة لكل تسلسل (دفعة من {DB})")
)
ax.set_ylabel(ar("ms per generated token", "ميلي ثانية لكل رمز مولَّد"))
ax.set_title(ar("Every new token reads the whole cache", "كل رمز جديد يقرأ الذاكرة المؤقتة كاملة"))
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

say(
    f"MHA step: {decode_ms_short:.1f} ms at {ctxs[0]} tokens, {decode_ms_long:.1f} ms at {ctxs[-1]} "
    f"({decode_slowdown:.1f}x). GQA at {ctxs[-1]}: {decode_gqa_ms_long:.1f} ms.\n"
    f"effective read rate at the longest context: {decode_gbps:.0f} GB/s",
    f"زمن الخطوة مع MHA: {decode_ms_short:.1f} ms عند سياق {ctxs[0]}، و{decode_ms_long:.1f} ms عند سياق {ctxs[-1]} "
    f"(أي {decode_slowdown:.1f} × أبطأ). ومع GQA عند سياق {ctxs[-1]}: {decode_gqa_ms_long:.1f} ms.\n"
    f"معدّل القراءة الفعلي عند أطول سياق: {decode_gbps:.0f} GB/s",
)
# --8<-- [end:decode]


# --8<-- [start:plan]
# YOUR TURN — size a server.
# Predict the number first, then run. Change KV_HEADS to 8, 4, 2 or 1.
KV_HEADS = N_HEADS
CONTEXT = cfg["plan_ctx"]

assert N_HEADS % KV_HEADS == 0, f"KV_HEADS must divide {N_HEADS}"
budget = cfg["budget_gib"] * GIB
headroom = cfg["headroom_gib"] * GIB

free()
base = torch.cuda.memory_allocated()
model = build(KV_HEADS)
per_seq = predicted_cache_bytes(N_LAYERS, KV_HEADS, D_HEAD, 1, CONTEXT)
plan_batch = int((budget - weight_bytes(model) - headroom) // per_seq)

say(
    f"budget {cfg['budget_gib']} GiB, weights {weight_bytes(model) / GIB:.2f} GiB, "
    f"{per_seq / 2**20:.0f} MiB of cache per sequence at {CONTEXT} tokens\n"
    f"-> {plan_batch} sequences at once",
    f"الميزانية {cfg['budget_gib']} GiB، والأوزان {weight_bytes(model) / GIB:.2f} GiB، "
    f"والذاكرة المؤقتة لكل تسلسل بطول {CONTEXT}: {per_seq / 2**20:.0f} MiB\n"
    f"← عدد التسلسلات التي تتسع لها البطاقة معاً: {plan_batch}",
)

if plan_batch >= 1:
    # Prove the plan: hold that many full caches and generate one token.
    torch.cuda.reset_peak_memory_stats()
    cache = model.filled_cache(plan_batch, CONTEXT)
    ids = torch.zeros(plan_batch, 1, dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        model.decode(ids, cache, CONTEXT - 1)
    torch.cuda.synchronize()
    plan_peak_gib = (torch.cuda.max_memory_allocated() - base) / GIB
    say(
        f"measured peak {plan_peak_gib:.2f} GiB of {cfg['budget_gib']} GiB",
        f"الذروة المقيسة {plan_peak_gib:.2f} GiB من أصل {cfg['budget_gib']} GiB",
    )
    del cache, ids
del model
free()
# --8<-- [end:plan]


# --8<-- [start:verify]
formula_ok = env.check("cache-matches-formula", formula_ratio)
overtake_ok = env.check("cache-overtakes-weights", cache_over_weights)
gqa_ok = env.check("gqa-shrinks-cache", gqa_ratio)
# --8<-- [end:verify]


# --8<-- [start:finish]
receipt = env.receipt()
# --8<-- [end:finish]
