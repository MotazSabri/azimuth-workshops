# Azimuth Workshops

**English** · [العربية](#ورش-أزيموث)

Multi-paper prototypes that run on a free Colab GPU, in English and Arabic.

This is the fourth tier of [Azimuth](https://azimuth.blog). The papers explain
why an idea exists. The demos make it tangible. The Lab has you assemble an
architecture without a GPU. **A workshop is where it runs on real hardware and
produces a real number.**

---

## For learners

Open a workshop from its page on Azimuth, or straight from here:

| workshop | English | العربية |
|---|---|---|
| Anomaly detection with an autoencoder | [Colab](https://colab.research.google.com/github/azimuth/azimuth-workshops/blob/stable/generated/notebooks/anomaly-detection-autoencoder.en.ipynb) | [Colab](https://colab.research.google.com/github/azimuth/azimuth-workshops/blob/stable/generated/notebooks/anomaly-detection-autoencoder.ar.ipynb) |

Three things worth knowing before you start:

- **Save a copy to Drive first** (File → Save a copy in Drive). Edits to the
  original are not saved anywhere.
- **The badges point at `stable`,** which only ever moves to a commit whose
  notebooks were executed end to end by CI on a real GPU. `main` may contain
  work that has not run yet.
- **The free tier is the default.** `PROFILE = "free"` at the top of every
  notebook is sized for Colab's free T4. Change it only if you have better
  hardware — there is no commented-out code to hunt for.

Nothing is uploaded. Checks run inside your runtime against a threshold written
in the workshop file, which you can read. The completion code at the end is
derived from which checks passed, and pasting it on the site is the only thing
that leaves the notebook.

---

## For contributors

### Layout

```
workshops/<slug>/
  workshop.yaml     prose, structure, cell ids, profiles, checks, hints, terms
  code.py           the executable spine, cut into named regions
generated/
  notebooks/        built by CI — outputs stripped, never hand-edited
  runs/             captured outputs from the last verified execution
shim/azimuth_nb/    the setup package every notebook imports
```

### The two-file rule

A workshop is **`workshop.yaml` + `code.py`**, and nothing else. No per-workshop
Python package, no bespoke checker, no notebook committed by hand. If a workshop
cannot be expressed in those two files, the right move is to extend the schema
once so every future workshop can use the addition too — the same rule the Lab's
challenges are written under, and for the same reason.

### How the two files join

`code.py` is cut into regions:

```python
# --8<-- [start:train]
...
# --8<-- [end:train]
```

and a `cell` block in `workshop.yaml` names one:

```yaml
- type: cell
  id: train            # joins to generated/runs/<slug>/manifest.json
  ref: train           # joins to the region above
  captures: [loss_curve, final_loss]
```

Every region must be referenced and every `ref` must resolve. The build asserts
both directions, so a region cannot be orphaned by an edit to the prose.

`id` is the join key for captured outputs. It is stable across edits: renaming
an id orphans that cell's outputs and the drift assertions will say so.

### Scale is data, never a comment

Every size the code uses comes from `env.cfg`, filled from the active profile.
A workshop must run inside the free tier on its default profile — this is
checked, and it is the one constraint that never gets relaxed.

The alternative, which every notebook repository eventually becomes, is a file
with `# batch_size = 512  # uncomment for A100` scattered through it and two
half-maintained versions of the truth.

### torch is asserted, never installed

Colab ships a PyTorch built against its own driver. Installing another over it
produces CUDA errors that name nothing leading back to the line that caused
them. If torch is missing or old, the fix is a fresh runtime, and `AZ-E201` /
`AZ-E202` say so in both languages.

### Both languages, always

Arabic is not a translation pass that happens later. A missing `ar` is a lint
error, and the Arabic build is executed by CI exactly like the English one —
which is how the first run of this repo caught a graded check leaking English
into Arabic output.

Identifiers stay ASCII in both builds. One program, two voices, so the Arabic
notebook cannot drift into being a different workshop.

### Error codes

`AZ-E1xx` runtime · `AZ-E2xx` environment · `AZ-E3xx` assets · `AZ-E4xx` our bug.
Every one carries both languages and a fix, not just a diagnosis. The catalogue
is data in `shim/azimuth_nb/errors.py` so gaps are visible without reading the
code that raises them.

### Licensing

Code is MIT (`LICENSE-CODE`). Prose is CC BY-SA 4.0 (`LICENSE-CONTENT`). Two
licenses because the two things are used differently: code should lift into your
project with no strings, and teaching should stay open if it is built upon.
Datasets are not ours; each declares its own terms in its `assets:` block.

---

<div dir="rtl">

# ورش أزيموث

نماذج أولية متعددة الأوراق تعمل على معالج رسوميات مجاني في Colab، بالعربية
والإنجليزية.

هذه هي الطبقة الرابعة من [أزيموث](https://azimuth.blog). الأوراق تشرح لماذا
وُجدت الفكرة، والعروض التفاعلية تجعلها ملموسة، والمختبر يجعلك تركّب معمارية دون
معالج رسوميات. **أما الورشة فهي حيث تعمل الفكرة على عتاد حقيقي وتُنتج رقماً
حقيقياً.**

## للمتعلّمين

افتح الورشة من صفحتها على أزيموث، أو مباشرة من الجدول في الأعلى.

ثلاثة أمور قبل أن تبدأ:

- **احفظ نسخة في Drive أولاً** (ملف ← حفظ نسخة في Drive). التعديلات على الأصل
  لا تُحفظ في أي مكان.
- **الشارات تشير إلى فرع `stable`** الذي لا يتحرك إلا إلى إصدار نُفِّذت دفاتره
  كاملةً على معالج رسوميات حقيقي. أما `main` فقد يحوي عملاً لم يُنفَّذ بعد.
- **المستوى المجاني هو الافتراضي.** السطر `PROFILE = "free"` في أعلى كل دفتر
  مُعَدٌّ لمعالج T4 المجاني. لا تغيّره إلا إن كان لديك عتاد أفضل — ولا يوجد كود
  معطَّل عليك البحث عنه وتفعيله.

لا يُرفع شيء إلى أي خادم. تعمل الفحوص داخل بيئتك مقابل عتبة مكتوبة في ملف
الورشة يمكنك قراءتها. ورمز الإتمام في النهاية مشتقّ من الفحوص التي اجتزتها،
ولصقه في الموقع هو الشيء الوحيد الذي يغادر الدفتر.

## للمساهمين

الورشة ملفان اثنان: `workshop.yaml` و`code.py`، ولا شيء غيرهما. لا حزمة بايثون
لكل ورشة، ولا فاحص مخصّص، ولا دفتر مكتوب باليد. وإن تعذّر التعبير عن ورشة
بهذين الملفين فالصواب توسيع المخطط مرة واحدة لتستفيد منه كل ورشة لاحقة — وهي
القاعدة ذاتها التي كُتبت بها تحديات المختبر، وللسبب ذاته.

العربية ليست مرحلة ترجمة لاحقة: غياب `ar` خطأ يوقف البناء، وتُنفَّذ النسخة
العربية في التكامل المستمر تماماً كالإنجليزية — وهكذا اكتُشف في أول تشغيل لهذا
المستودع أن أحد الفحوص كان يسرّب الإنجليزية إلى المخرجات العربية.

الأحجام كلها تأتي من `env.cfg` بحسب الملف الفعّال، ويجب أن تعمل الورشة داخل
حدود المستوى المجاني بإعداداتها الافتراضية. هذا القيد لا يُخفَّف أبداً.

</div>
