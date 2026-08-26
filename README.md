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

---

## Adding a workshop, end to end

Nine steps. Everything else — notebooks, manifests, indexes, search entries,
paper invites — is generated.

```
 1. pick papers      →  2. write 2 files  →  3. build   →  4. run locally
                                                              ↓
 9. it's on the site ←  8. bump submodule ←  7. promote ←  5-6. push & CI runs
```

### 1 · Pick the papers, and the constellation

Two or three `core` papers the workshop genuinely teaches from, plus any
`supporting` ones. They may span constellations — the first one draws on three.
Then choose the constellation the workshop **teaches from**, and check
`WORKSHOP.md`'s coverage sketch so you know where it lands.

### 2 · Write the two files

```
workshops/<slug>/workshop.yaml     prose, cells, profiles, checks, hints, terms
workshops/<slug>/code.py           the executable spine, region-marked
```

That is the whole authoring surface. Copy an existing pair and replace.
Non-negotiables, each enforced by `tools/validate.py`:

- **Both languages everywhere.** A missing `ar` is an error, not a TODO.
- **Every size comes from `env.cfg`.** No constants in the code, no
  commented-out alternatives.
- **The default profile fits the free tier.** ≤16 GB VRAM, ≤12 GB RAM.
- **`sha256: null`** on assets for now — step 4 fills it in.
- **`status: draft`** until it is ready; drafts are reachable by URL and absent
  from the index and search.

```bash
python tools/validate.py          # 0 errors before you go on
```

### 3 · Build the notebooks

```bash
python tools/build_notebooks.py
```

Writes `generated/notebooks/<slug>.{en,ar}.ipynb` — outputs stripped, stable
cell ids, the bootstrap cell pointed at whatever `config.yaml` says. Never edit
these by hand; they are regenerated and your edit will vanish.

### 4 · Run it yourself, and pin the dataset

```bash
python tools/fetch-assets.py <slug> --from-mirror   # verify the pin, offline
python tools/pin-asset.py <slug>                    # downloads, hashes, writes sha256
python tools/execute.py <slug>                      # runs both languages, captures outputs
```

**When a dataset changes** — a corrected encoding, a trimmed corpus — the order
matters, because a stale pin fails silently rather than loudly:

```bash
# 1. put the new file under mirror/ and push it to main
git add mirror/<file> && git commit && git push origin main
# 2. re-pin: `pin-asset` refuses to overwrite an existing hash without --force
python tools/pin-asset.py <slug> --force
# 3. refresh the local cache and confirm the two agree
python tools/fetch-assets.py <slug> --force
```

Asset URLs point at `main`, never `stable`. Pinning them to `stable` deadlocks
this: the corrected file is invisible until promotion, promotion should not
happen before it is tested, and the old hash still matches the old file — so
the run reports "verified" while measuring stale data.

**Then read what came back.** This is where the workshop actually gets written:
on the first one, three separate statements in the prose turned out to be wrong
about the run — an epoch count, a caption describing the loss curve, and a
threshold. None were catchable by review. Expect to rewrite around the numbers,
not merely to check them.

Open a notebook on Colab too, at least once. CI is not Colab.

### 5 · Push to `main`

```bash
git add workshops/<slug> generated/ && git commit && git push origin main
```

`validate` runs on every push: lint, schema, and that the committed notebooks
still match their source.

### 6 · Let CI execute it

`execute` fires automatically when `code.py` or `workshop.yaml` changes. It
runs both languages, captures outputs to
`generated/runs/<slug>/{en,ar}.manifest.json`, and commits them. A failed run
still publishes — with its error visible — but will not promote.

You can also trigger it by hand from the Actions tab.

### 7 · Promote to `stable`

`promote` runs after a green `execute` and fast-forwards `stable`. It refuses
unless every **`status: stable`** workshop has a verified manifest in both
languages. Drafts are skipped, so an unfinished workshop never freezes the
branch.

To publish, set in `workshop.yaml`:

```yaml
status: stable
publishedAt: "2026-08-25"     # required once stable; the newsletter window is a date
```

then push and let `execute` → `promote` run again.

### 8 · Bump the submodule in the site repo

```bash
cd azimuth
git -C vendor/azimuth-workshops fetch && git -C vendor/azimuth-workshops checkout origin/stable
npm run content            # build-content → lab → build-workshops
git add vendor/azimuth-workshops && git commit -m "workshops: add <slug>" && git push
```

The pin is a reviewable commit on purpose — the site's content should not
change without one.

### 9 · What appears, with no further work

- `/workshop` — card under its constellation, thumbnail from the captured plot
- `/workshop/<slug>` — full page, both languages, outputs, verification footer
- **Every `core` paper's page** — a WorkshopInvite, via the reverse index
- **Search** — under Workshops, findable by its own name or a linked paper's
- **Newsletter** — a candidate in the next issue, using `goal` as its line
- **Analytics** — views and finish rate under `workshop:<slug>`
- **Badges** — counts toward its constellation

### Changing one afterwards

| you changed | what to do |
|---|---|
| prose only | build notebooks, push — no re-run needed |
| `code.py` | push; `execute` re-runs automatically (the codeHash moved) |
| a cell `id` | expect "outputs pending re-run" until CI re-executes |
| a dataset URL | re-pin, add a mirror, re-run |

The one rule: **never hand-edit anything under `generated/`.** It is all
rebuilt, and `tools/build_notebooks.py --check` fails CI when it drifts.
