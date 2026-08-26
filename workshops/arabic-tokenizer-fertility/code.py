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
#
# DECODE EACH PIECE, never print `.tokens` directly. GPT-2 is a BYTE-level
# BPE: its raw token strings are the bytes re-encoded as printable Latin-1, so
# an Arabic word shows up as `ĠØ§ÙĦ | Øª | Ø¹` — mojibake that looks like a
# bug in the notebook rather than the finding. Decoding each id individually
# puts the actual fragments back on screen, and where a token is half a UTF-8
# character it shows as `�` — which IS the finding: the tokenizer is cutting
# below the level of a letter.
sample_ar, sample_en = pairs[0]
worst_name = worst["tokenizer"]
tok = tokenizers[worst_name]


def pieces_of(tokenizer, text):
    """Decoded pieces, plus how many were not even whole characters.

    A byte-level BPE splits UTF-8, and an Arabic letter is two bytes. So a
    single token id can be HALF A LETTER, and decoding it alone yields no
    character at all — U+FFFD, the replacement character.

    That is not a corpus problem and not a rendering problem. It is the
    measurement: the same file decodes perfectly through BLOOM two blocks
    below. Counting the fragments turns the confusing symbol into the number
    it was always standing for.
    """
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    parts, fragments = [], 0
    for i in ids:
        piece = tokenizer.decode([i])
        if not piece or "\ufffd" in piece:
            fragments += 1
            parts.append("\ufffd")
        else:
            parts.append(piece)
    return parts, fragments


def show(tokenizer, label, text, name):
    parts, fragments = pieces_of(tokenizer, text)
    print(f"\n{label} — {len(parts)} tokens, {len(text.split())} words  [{name}]")
    print("  " + " | ".join(parts))
    if fragments:
        share = fragments / len(parts) * 100
        if env.lang == "ar":
            print(
                f"  ← {fragments} من {len(parts)} رمزاً ({share:.0f}%) ليست حروفاً كاملة."
                " رمز \ufffd يعني قطعة أصغر من الحرف الواحد — وهذا هو القياس لا خطأ عرض."
            )
        else:
            print(
                f"  ← {fragments} of {len(parts)} tokens ({share:.0f}%) are not whole"
                " characters. A \ufffd is a piece SMALLER than one letter — that is the"
                " measurement, not a rendering fault."
            )


sample_ar, sample_en = pairs[0]
worst_name = worst["tokenizer"]
tok = tokenizers[worst_name]

show(tok, "العربية" if env.lang == "ar" else "Arabic", sample_ar, worst_name)
show(tok, "الإنجليزية" if env.lang == "ar" else "English", sample_en, worst_name)

# The same Arabic sentence through a tokenizer that bought vocabulary for it.
# This block is what proves the file is fine and the fragmentation is a choice.
best_name = min(table, key=lambda r: r["ratio"])["tokenizer"]
if best_name != worst_name:
    show(
        tokenizers[best_name],
        "العربية" if env.lang == "ar" else "Arabic",
        sample_ar,
        best_name,
    )
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


enabled = [
    name
    for name, on in (
        ("diacritics", STRIP_DIACRITICS),
        ("alef", NORMALIZE_ALEF),
        ("tatweel", STRIP_TATWEEL),
    )
    if on
]

if not enabled:
    # Saying so beats printing "6.018 -> 6.018 (-0.0%)", which reads like the
    # normalization does not work rather than like it was never switched on.
    if env.lang == "ar":
        print("لم تُفعَّل أي معالجة بعد — أعد أحد الثوابت أعلاه إلى True ثم شغّل الخلية.")
    else:
        print("No normalization enabled yet — set one of the flags above to True and re-run.")
else:
    # Measured on the WORST tokenizer, the one the headline number came from.
    # Normalizing against a tokenizer that already handles Arabic well would
    # show almost no gain and teach the opposite of the truth.
    normalized = [normalize(t) for t in ar_texts]
    after = count_tokens(tok, normalized) / ar_words
    saved = (ar_fertility - after) / ar_fertility * 100 if ar_fertility else 0.0
    print(f"{worst_name}: {ar_fertility:.3f} -> {after:.3f} tokens/word  ({-saved:+.1f}%)")
    print(f"  enabled: {', '.join(enabled)}")
# --8<-- [end:reduce_the_tax]


# --8<-- [start:verify]
fertility_ratio_final = ratio
ratio_ok = env.check("fertility-measured", fertility_ratio_final)
count_ok = env.check("tokenizers-compared", n_tokenizers)
# --8<-- [end:verify]


# --8<-- [start:finish]
receipt = env.receipt()
# --8<-- [end:finish]
