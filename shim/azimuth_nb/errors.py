"""Bilingual failure, with a code you can search for.

A learner who hits a wall in a Colab notebook has three bad options: read an
English stack trace they may not read comfortably, paste it into a search
engine that will return answers about a different library, or give up. The
third is the common one.

So every failure this package raises carries:

  * an ``AZ-Exxx`` code, stable across versions, greppable, and the thing to
    quote when asking for help
  * both languages, always — not "English with a translation available"
  * a *fix*, not just a diagnosis. "No GPU attached" is a diagnosis.
    "Runtime → Change runtime type → T4 GPU" is a fix.

The codes are grouped so the first digit tells you whose problem it is:

  AZ-E1xx  the runtime         — no GPU, too little VRAM, too little RAM
  AZ-E2xx  the environment     — torch missing, torch too old
  AZ-E3xx  the assets          — a dataset that would not download
  AZ-E4xx  the workshop itself — a bad slug, a malformed manifest

That last group is ours, not the learner's, and says so in the message.
"""

from __future__ import annotations

# Arabic is written RTL; the AZ code and any Latin identifier inside an Arabic
# sentence are wrapped in U+2066..U+2069 (LTR isolate / pop) so a terminal or
# a notebook output cell renders them the right way round. Without the
# isolates, "AZ-E101" in Arabic prose reads as "101E-ZA".
_LRI = "\u2066"
_PDI = "\u2069"


def ltr(text: str) -> str:
    """Wrap a Latin fragment so it survives inside Arabic prose."""
    return f"{_LRI}{text}{_PDI}"


class AzimuthError(RuntimeError):
    """A failure a learner is meant to read, not debug.

    Carries both languages. ``__str__`` renders the language the notebook was
    built in, because a notebook is monolingual even though this package is
    not.
    """

    def __init__(
        self, code: str, en: str, ar: str, fix_en: str = "", fix_ar: str = "", lang: str = "en"
    ):
        self.code = code
        self.en = en
        self.ar = ar
        self.fix_en = fix_en
        self.fix_ar = fix_ar
        self.lang = lang if lang in ("en", "ar") else "en"
        super().__init__(self.render())

    def render(self, lang: str | None = None) -> str:
        lang = lang or self.lang
        if lang == "ar":
            body = f"{ltr(self.code)} — {self.ar}"
            if self.fix_ar:
                body += f"\n\nالحل: {self.fix_ar}"
        else:
            body = f"{self.code} — {self.en}"
            if self.fix_en:
                body += f"\n\nFix: {self.fix_en}"
        return body


# ── the catalogue ────────────────────────────────────────────────────────────
#
# Defined as data rather than as raise-sites, so the full set can be listed,
# translated, and checked for gaps without reading the code that raises it.

CATALOGUE: dict[str, dict[str, str]] = {
    # ── AZ-E1xx: the runtime ────────────────────────────────────────────────
    "AZ-E101": {
        "en": "This workshop needs a GPU, and this runtime does not have one.",
        "ar": "تحتاج هذه الورشة إلى معالج رسوميات، وهذه البيئة لا تملك واحداً.",
        "fix_en": "Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The free T4 is enough for this workshop.",
        "fix_ar": "من قائمة Runtime اختر Change runtime type ثم T4 GPU، ثم Runtime ← Run all. المعالج المجاني T4 يكفي لهذه الورشة.",
    },
    "AZ-E102": {
        "en": "The attached GPU has less memory than this workshop's profile asks for.",
        "ar": "ذاكرة معالج الرسوميات المتصل أقل مما يطلبه ملف إعدادات هذه الورشة.",
        "fix_en": "Set PROFILE = 'free' in the setup cell and run again. That profile is sized for a 16 GB T4.",
        "fix_ar": "اضبط PROFILE = 'free' في خلية الإعداد وأعد التشغيل. هذا الملف مُعدّ لمعالج T4 بسعة 16 غيغابايت.",
    },
    "AZ-E103": {
        "en": "This runtime has less system RAM than the workshop needs to hold the dataset.",
        "ar": "ذاكرة النظام في هذه البيئة أقل مما تحتاجه الورشة لتحميل البيانات.",
        "fix_en": "Runtime → Disconnect and delete runtime, then reconnect. A fresh runtime usually clears it.",
        "fix_ar": "من قائمة Runtime اختر Disconnect and delete runtime ثم أعد الاتصال. البيئة الجديدة تحلّ المشكلة عادةً.",
    },
    "AZ-E104": {
        "en": "This workshop declares a platform it can run on, and this is not one of them.",
        "ar": "تعلن هذه الورشة المنصّات التي تعمل عليها، وهذه ليست منها.",
        # Some libraries have no working build for every OS — bitsandbytes has
        # no Windows CUDA extension, and the failure it produces is a bare
        # "[WinError 127] The specified procedure could not be found" thrown
        # from deep inside a model load, minutes in, naming nothing useful.
        # Refusing at setup with the reason is worth a schema field.
        "fix_en": "Open this workshop in Colab instead — the badge on its page runs it on Linux, where the libraries it needs have working builds.",
        "fix_ar": "افتح هذه الورشة في Colab بدلاً من ذلك — فالشارة في صفحتها تشغّلها على لينكس، حيث تتوفر للمكتبات التي تحتاجها نسخ عاملة.",
    },
    # ── AZ-E2xx: the environment ────────────────────────────────────────────
    "AZ-E201": {
        "en": "PyTorch is not installed in this runtime.",
        "ar": "مكتبة PyTorch غير مثبّتة في هذه البيئة.",
        # We ASSERT torch, never install it. Installing a second torch on top
        # of Colab's is the single most common way to break a runtime: the
        # new wheel and the driver disagree, and every later cell fails with
        # a CUDA error that names none of this.
        "fix_en": "Colab ships PyTorch already, so this usually means a custom runtime. Runtime → Disconnect and delete runtime, then reconnect to a standard Colab GPU runtime. Do not pip install torch — a second torch will not match the driver.",
        "fix_ar": "تأتي Colab بـ PyTorch مثبّتة مسبقاً، لذا يعني هذا عادةً أنك في بيئة مخصّصة. من Runtime اختر Disconnect and delete runtime ثم أعد الاتصال ببيئة Colab قياسية. لا تُثبّت torch عبر pip — نسخة ثانية لن تطابق مشغّل العتاد.",
    },
    "AZ-E202": {
        "en": "The installed PyTorch is older than this workshop was written against.",
        "ar": "نسخة PyTorch المثبّتة أقدم مما كُتبت هذه الورشة عليه.",
        "fix_en": "Runtime → Disconnect and delete runtime, then reconnect. Colab's default image is kept current.",
        "fix_ar": "من Runtime اختر Disconnect and delete runtime ثم أعد الاتصال. صورة Colab الافتراضية محدَّثة باستمرار.",
    },
    # ── AZ-E3xx: the assets ─────────────────────────────────────────────────
    "AZ-E301": {
        "en": "The dataset could not be downloaded from any of its sources.",
        "ar": "تعذّر تنزيل مجموعة البيانات من أيٍّ من مصادرها.",
        # The shim already retried three times with backoff before raising
        # this, so "try again" is advice it has taken on your behalf. Say what
        # is left to try instead.
        "fix_en": "The download was retried three times and failed each time. Check your connection, then run the cell again. If it keeps failing the source may be down — open an issue on the workshops repo and we will re-point it.",
        "fix_ar": "أُعيدت محاولة التنزيل ثلاث مرات وأخفقت في كلٍّ منها. تحقّق من اتصالك ثم أعد تشغيل الخلية. وإن استمر الفشل فقد يكون المصدر معطّلاً؛ افتح مشكلة في مستودع الورش وسنعيد توجيهه.",
    },
    "AZ-E302": {
        "en": "The dataset downloaded, but its contents are not what this workshop expects.",
        "ar": "نُزّلت البيانات لكن محتواها ليس ما تتوقعه هذه الورشة.",
        "fix_en": "The file changed at the source. Open an issue on the workshops repo — this needs a pin update, not a retry.",
        "fix_ar": "تغيّر الملف عند المصدر. افتح مشكلة في مستودع الورش — يحتاج الأمر إلى تحديث التثبيت لا إلى إعادة المحاولة.",
    },
    # ── AZ-E4xx: our fault, and it says so ──────────────────────────────────
    "AZ-E401": {
        "en": "This notebook asked for a workshop that does not exist in the repository.",
        "ar": "طلب هذا الدفتر ورشة غير موجودة في المستودع.",
        "fix_en": "This is a bug on our side, not yours. Please open an issue quoting this code.",
        "fix_ar": "هذا خلل من جانبنا لا من جانبك. من فضلك افتح مشكلة مع ذكر هذا الرمز.",
    },
    "AZ-E402": {
        "en": "This notebook asked for a scale profile the workshop does not define.",
        "ar": "طلب هذا الدفتر ملف إعدادات حجم غير معرَّف في الورشة.",
        "fix_en": "Set PROFILE back to 'free' in the setup cell.",
        "fix_ar": "أعد ضبط PROFILE إلى 'free' في خلية الإعداد.",
    },
}


def fail(code: str, lang: str = "en", **extra: str) -> AzimuthError:
    """Build the catalogued error. Unknown codes still produce something
    readable rather than a KeyError inside an error path — failing while
    failing is how a learner ends up with no message at all."""
    entry = CATALOGUE.get(code)
    if entry is None:
        return AzimuthError(
            code,
            f"Unexpected failure ({code}).",
            f"عطل غير متوقع ({ltr(code)}).",
            lang=lang,
        )
    en = entry["en"]
    ar = entry["ar"]
    if extra:
        detail_en = " ".join(f"{k}: {v}." for k, v in extra.items())
        en = f"{en} {detail_en}"
        ar = f"{ar} {ltr(detail_en)}"
    return AzimuthError(code, en, ar, entry.get("fix_en", ""), entry.get("fix_ar", ""), lang=lang)
