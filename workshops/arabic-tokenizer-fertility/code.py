# workshops/arabic-tokenizer-fertility/code.py
#
# Measures how many tokens the same meaning costs in Arabic vs English.
#
# Identifiers stay ASCII in both language builds; only comments and printed
# strings are localized, and they are localized at runtime by env.lang rather
# than by generating two different programs.

# --8<-- [start:setup]
import azimuth_nb as azimuth

env = azimuth.setup(SLUG, lang=LANG, profile=PROFILE)
# --8<-- [end:setup]


# --8<-- [start:load_corpus]
import random

pairs_raw = []
with open(env.assets["parallel.tsv"], encoding="utf-8") as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            pairs_raw.append((parts[0].strip(), parts[1].strip()))

# A fixed seed so the sample — and therefore every number below — is the same
# on your machine, on CI, and on the page.
random.seed(env.cfg["seed"])
limit = env.cfg["sampleSentences"]
pairs = random.sample(pairs_raw, min(limit, len(pairs_raw))) if limit else pairs_raw

# WHITESPACE, deliberately. "Word" has to mean the same operation in both
# languages or the ratio compares nothing. Whitespace undercounts Arabic
# morphology — one written word often carries article, stem and pronoun — so
# this makes the result a LOWER BOUND on the real gap rather than a flattering
# one. Overstating the case would be the easiest way to lose the argument.
ar_words = sum(len(ar.split()) for ar, _ in pairs)
en_words = sum(len(en.split()) for _, en in pairs)

if env.lang == "ar":
    print(f"أزواج: {len(pairs):,} من أصل {len(pairs_raw):,}")
    print(f"كلمات: {ar_words:,} عربية · {en_words:,} إنجليزية")
else:
    print(f"pairs: {len(pairs):,} of {len(pairs_raw):,}")
    print(f"words: {ar_words:,} Arabic · {en_words:,} English")
# --8<-- [end:load_corpus]


# --8<-- [start:tokenizers]
from tokenizers import Tokenizer

# Three different bets about which languages deserve vocabulary budget.
# Loaded from the Hub by name — small JSON files, no model weights.
SPECS = [
    ("gpt2", "openai-community/gpt2", "50k pieces, almost all English"),
    ("bloom", "bigscience/bloom", "250k pieces shared across 46 languages"),
    ("xlm-roberta", "FacebookAI/xlm-roberta-base", "250k pieces, 100 languages"),
]

tokenizers = {}
for name, repo, note in SPECS:
    try:
        tokenizers[name] = Tokenizer.from_pretrained(repo)
        print(f"  · {name:12} {note}")
    except Exception as exc:
        # A tokenizer that will not load is reported and skipped, not fatal:
        # the comparison still means something with two, and a dead Hub should
        # not cost you the whole workshop.
        print(f"  ! {name:12} unavailable ({type(exc).__name__}) — skipped")

n_tokenizers = len(tokenizers)
# --8<-- [end:tokenizers]


# --8<-- [start:fertility]
def count_tokens(tok, texts):
    """Total pieces across a list of strings, no special tokens."""
    return sum(len(tok.encode(t, add_special_tokens=False).ids) for t in texts)


ar_texts = [ar for ar, _ in pairs]
en_texts = [en for _, en in pairs]

table = []
for name, tok in tokenizers.items():
    ar_tokens = count_tokens(tok, ar_texts)
    en_tokens = count_tokens(tok, en_texts)
    ar_f = ar_tokens / ar_words
    en_f = en_tokens / en_words
    table.append(
        {
            "tokenizer": name,
            "ar_fertility": round(ar_f, 3),
            "en_fertility": round(en_f, 3),
            "ratio": round(ar_f / en_f, 3),
        }
    )

header = f"{'tokenizer':14}{'ar/word':>10}{'en/word':>10}{'ratio':>9}"
print("\n" + header)
print("-" * len(header))
for row in table:
    print(
        f"{row['tokenizer']:14}{row['ar_fertility']:>10.2f}"
        f"{row['en_fertility']:>10.2f}{row['ratio']:>9.2f}"
    )

# The headline numbers come from the WORST tokenizer for Arabic, because that
# is the one an Arabic reader is most likely to be paying for.
worst = max(table, key=lambda r: r["ratio"])
ar_fertility = worst["ar_fertility"]
en_fertility = worst["en_fertility"]
ratio = worst["ratio"]
# --8<-- [end:fertility]


# --8<-- [start:pieces]
env.explain("subword")

# One sentence, cut both ways. The average is an argument; this is the
# evidence for it.
sample_ar, sample_en = pairs[0]
name, tok = next(iter(tokenizers.items()))
for label, text in (
    (("العربية" if env.lang == "ar" else "Arabic"), sample_ar),
    (("الإنجليزية" if env.lang == "ar" else "English"), sample_en),
):
    ids = tok.encode(text, add_special_tokens=False)
    print(f"\n{label} — {len(ids.ids)} tokens, {len(text.split())} words")
    print("  " + " | ".join(ids.tokens))
# --8<-- [end:pieces]


# --8<-- [start:the_bill]
env.explain("context window")

price = env.cfg["pricePerMillionTokens"]
window = env.cfg["contextWindow"]

# What the ratio means once it leaves the spreadsheet.
ar_cost_multiple = round(ratio, 3)
effective_context_ar = int(window / ar_fertility)
effective_context_en = int(window / en_fertility)

ar_million_words_cost = (ar_fertility * 1_000_000 / 1_000_000) * price
en_million_words_cost = (en_fertility * 1_000_000 / 1_000_000) * price

if env.lang == "ar":
    print(
        f"لكل مليون كلمة: {ar_million_words_cost:.2f}$ بالعربية · {en_million_words_cost:.2f}$ بالإنجليزية"
    )
    print(f"القارئ العربي يدفع {ar_cost_multiple:.2f}× للمحتوى ذاته")
    print(
        f"نافذة {window:,} رمز تسع {effective_context_ar:,} كلمة عربية · {effective_context_en:,} كلمة إنجليزية"
    )
else:
    print(
        f"per million words: ${ar_million_words_cost:.2f} Arabic · ${en_million_words_cost:.2f} English"
    )
    print(f"an Arabic reader pays {ar_cost_multiple:.2f}× for the same content")
    print(
        f"a {window:,}-token window holds {effective_context_ar:,} Arabic words · {effective_context_en:,} English"
    )
# --8<-- [end:the_bill]


# --8<-- [start:reduce_the_tax]
# YOUR TURN.
#
# Each of these is one line, and each changes how the vocabulary matches.
# Turn them on, re-measure, and report BOTH what you saved and what you lost.
import re

STRIP_DIACRITICS = False  # harmless on modern prose, destructive on Qur'anic text
NORMALIZE_ALEF = False  # أ إ آ -> ا ; ى -> ي
STRIP_TATWEEL = False  # the ـ elongation character


def normalize(text):
    if STRIP_DIACRITICS:
        text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    if NORMALIZE_ALEF:
        text = re.sub(r"[أإآ]", "ا", text).replace("ى", "ي")
    if STRIP_TATWEEL:
        text = text.replace("\u0640", "")
    return text


normalized = [normalize(t) for t in ar_texts]
after = count_tokens(tok, normalized) / ar_words
saved = (ar_fertility - after) / ar_fertility * 100 if ar_fertility else 0.0
print(f"{name}: {ar_fertility:.3f} -> {after:.3f} tokens/word  ({saved:+.1f}%)")
# --8<-- [end:reduce_the_tax]


# --8<-- [start:verify]
fertility_ratio_final = ratio
ratio_ok = env.check("fertility-measured", fertility_ratio_final)
count_ok = env.check("tokenizers-compared", n_tokenizers)
# --8<-- [end:verify]


# --8<-- [start:finish]
receipt = env.receipt()
# --8<-- [end:finish]
