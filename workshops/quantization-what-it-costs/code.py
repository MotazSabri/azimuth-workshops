# workshops/quantization-what-it-costs/code.py
#
# Loads the same model three ways and measures three costs each. The point is
# that the three do not move together — and one of them moves the wrong way.

# --8<-- [start:setup]
import azimuth_nb as azimuth

env = azimuth.setup(SLUG, lang=LANG, profile=PROFILE)
# --8<-- [end:setup]


# --8<-- [start:windows]
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = env.cfg["model"]
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# BUILD THE WINDOWS ONCE, BEFORE ANY MODEL LOADS.
#
# Every variant is scored on byte-identical text. Re-sampling per model would
# put sampling noise on the same axis as the effect being measured, and a real
# int4 regression could hide inside it — the numbers would still look precise.
raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(t for t in raw["text"] if t.strip())
all_ids = tokenizer(text, return_tensors="pt").input_ids[0][: env.cfg["evalTokens"]]

WINDOW = 512
windows = [all_ids[i : i + WINDOW] for i in range(0, len(all_ids) - WINDOW, WINDOW)]
n_windows = len(windows)

if env.lang == "ar":
    print(f"النموذج: {MODEL}")
    print(f"نوافذ التقييم: {n_windows} نافذة × {WINDOW} رمز")
else:
    print(f"model: {MODEL}")
    print(f"eval windows: {n_windows} × {WINDOW} tokens")
# --8<-- [end:windows]


# --8<-- [start:measure]
import gc
import time

from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def load(variant):
    """One model, three ways. Only the dtype/quantization config differs."""
    if variant == "fp16":
        kwargs = {"torch_dtype": torch.float16}
    elif variant == "int8":
        kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    else:
        kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                # NF4 rather than plain int4: GPTQ's lesson is that WHERE the
                # levels sit matters as much as how many there are.
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        }
    return AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0", **kwargs)


def perplexity(model):
    """Mean NLL over the shared windows, exponentiated."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for window in windows:
            ids = window.unsqueeze(0).to("cuda:0")
            loss = model(ids, labels=ids).loss
            total += loss.item() * ids.numel()
            count += ids.numel()
    return float(torch.exp(torch.tensor(total / count)))


def footprint(model):
    """Megabytes of parameters as actually stored, not as declared.

    Reading nelement * element_size per parameter is the honest measure: a
    4-bit tensor reports element_size 1 with two values packed per byte, so
    counting declared dtypes would overstate int4 by exactly the factor the
    workshop is trying to measure.
    """
    return sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024**2


def latency(model, runs, new_tokens):
    """Median milliseconds per generated token, single sequence."""
    prompt = windows[0][:64].unsqueeze(0).to("cuda:0")
    with torch.no_grad():  # warm the kernels before timing anything
        model.generate(prompt, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=new_tokens, do_sample=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000 / new_tokens)
    times.sort()
    return times[len(times) // 2]


results = {}
for variant in ("fp16", "int8", "int4"):
    model = load(variant)
    results[variant] = {
        "ppl": perplexity(model),
        "mb": footprint(model),
        "ms": latency(model, env.cfg["latencyRuns"], env.cfg["newTokens"]),
    }
    print(
        f"  {variant:5} ppl {results[variant]['ppl']:7.3f}"
        f"  {results[variant]['mb']:8.1f} MB"
        f"  {results[variant]['ms']:6.1f} ms/token"
    )
    # Freed between variants so the memory figure is the model, not leftovers.
    del model
    gc.collect()
    torch.cuda.empty_cache()

fp16_ppl, int8_ppl, int4_ppl = (results[v]["ppl"] for v in ("fp16", "int8", "int4"))
fp16_mb, int8_mb, int4_mb = (results[v]["mb"] for v in ("fp16", "int8", "int4"))
# --8<-- [end:measure]


# --8<-- [start:curve]
import matplotlib.pyplot as plt

memory_ratio = fp16_mb / int4_mb
ppl_ratio_int8 = int8_ppl / fp16_ppl
ppl_ratio_int4 = int4_ppl / fp16_ppl

variants = ["fp16", "int8", "int4"]
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
for ax, key, title in zip(
    axes,
    ["ppl", "mb", "ms"],
    ["perplexity", "memory (MB)", "ms / token"],
):
    values = [results[v][key] for v in variants]
    ax.plot(variants, values, marker="o", color="#457b9d")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    # Zero-based so a flat line LOOKS flat. Autoscaled axes turn a 1% change
    # into a dramatic slope, which is how a chart lies without a wrong number
    # anywhere in it.
    ax.set_ylim(0, max(values) * 1.2)
fig.tight_layout()
plt.show()

if env.lang == "ar":
    print(f"الذاكرة: النصفية أكبر بـ {memory_ratio:.2f}× من الرباعية")
    print(f"الحيرة: ثمانية {ppl_ratio_int8:.3f}× · رباعية {ppl_ratio_int4:.3f}×")
else:
    print(f"memory: fp16 is {memory_ratio:.2f}× larger than int4")
    print(f"perplexity: int8 {ppl_ratio_int8:.3f}× · int4 {ppl_ratio_int4:.3f}×")
# --8<-- [end:curve]


# --8<-- [start:latency]
fp16_ms, int8_ms, int4_ms = (results[v]["ms"] for v in ("fp16", "int8", "int4"))

# Report the DIRECTION, because the direction is the surprise. Quantized
# weights are smaller but every matmul now pays a dequantize step, so at batch
# size 1 the smaller model is often the slower one.
if env.lang == "ar":
    print(
        f"زمن الاستجابة: نصفية {fp16_ms:.1f} · ثمانية {int8_ms:.1f} · رباعية {int4_ms:.1f} م.ث/رمز"
    )
    verdict = "أبطأ" if int4_ms > fp16_ms else "أسرع"
    print(
        f"الرباعية {verdict} من النصفية بعامل {max(int4_ms, fp16_ms) / min(int4_ms, fp16_ms):.2f}"
    )
else:
    print(f"latency: fp16 {fp16_ms:.1f} · int8 {int8_ms:.1f} · int4 {int4_ms:.1f} ms/token")
    verdict = "SLOWER" if int4_ms > fp16_ms else "faster"
    print(f"int4 is {verdict} than fp16 by {max(int4_ms, fp16_ms) / min(int4_ms, fp16_ms):.2f}×")
# --8<-- [end:latency]


# --8<-- [start:pick_a_deployment]
# YOUR TURN.
#
# State the constraint BEFORE reading the table. Then let the table choose.
BUDGET_MB = 400
MAX_MS_PER_TOKEN = 60
MAX_PPL_RATIO = 1.05

viable = [
    v
    for v in variants
    if results[v]["mb"] <= BUDGET_MB
    and results[v]["ms"] <= MAX_MS_PER_TOKEN
    and results[v]["ppl"] / fp16_ppl <= MAX_PPL_RATIO
]
print(f"viable under your constraints: {viable or 'none — loosen one, and say which'}")
# --8<-- [end:pick_a_deployment]


# --8<-- [start:verify]
memory_ok = env.check("memory-falls", memory_ratio)
quality_ok = env.check("quality-survives-int8", ppl_ratio_int8)
# --8<-- [end:verify]


# --8<-- [start:finish]
receipt = env.receipt()
# --8<-- [end:finish]
