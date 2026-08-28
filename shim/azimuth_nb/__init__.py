"""azimuth_nb — the one import a workshop notebook makes.

    import azimuth_nb as azimuth
    env = azimuth.setup("anomaly-detection-autoencoder", lang="en", profile=PROFILE)

That single call is the whole contract between a notebook and the repository.
It reads the workshop's own ``workshop.yaml``, checks the runtime against the
profile the learner selected, downloads and verifies the declared assets, and
hands back an ``Env``: the scale knobs, the glossary, the hints, and the
self-check.

Eight clauses, and the reasoning for each:

1.  **``setup(slug, lang, profile)`` is idempotent.** Colab users re-run cells
    constantly, and a setup that is expensive or stateful the second time
    teaches them not to. Re-running returns the same Env and re-downloads
    nothing.

2.  **torch is asserted, never installed.** See preflight.py. Installing a
    second torch over Colab's is the most reliable way to break a runtime.

3.  **Preflight refuses early, in both languages, with a fix.** Nineteen
    minutes into a run is the wrong time to learn there is no GPU.

4.  **``env.cfg`` comes from the active profile.** Scale lives in data, so the
    difference between the free tier and an A100 is one constant at the top of
    the notebook — never a commented-out line the learner has to find and
    uncomment, which is how notebooks rot into two half-maintained versions.

5.  **``env.term(key)`` reaches the site's glossary.** A workshop is the same
    corpus as the papers; a term should not mean something new because it
    appeared in a notebook.

6.  **``env.hint(n)`` is progressive.** Hints are earned one at a time and
    printed on demand, mirroring the Lab's hint pill exactly, so the two tiers
    feel like one product.

7.  **``env.check(id, result)`` verifies locally.** No answer key ships in the
    notebook and nothing is uploaded anywhere: the check holds the learner's
    own number against the threshold the workshop declared.

8.  **A receipt is written on success.** It is what the completion code is
    derived from, what the site pastes back into progress, and — with the same
    fields the CI run captures — what proves a page's outputs came from the
    code it is showing.

The package has NO third-party dependencies beyond PyYAML, which Colab already
has. Everything else it needs (torch, numpy, matplotlib) belongs to the
workshop, not to the harness.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .assets import fetch
from .errors import AzimuthError, fail, ltr
from .preflight import Runtime, check_profile, inspect_runtime

__all__ = ["REPO_ROOT", "AzimuthError", "Env", "__version__", "setup"]

__version__ = "0.1.0"

#: Repo root, found by walking up from this file. Works whether the notebook
#: cloned the repo, mounted it from Drive, or is running inside CI.
REPO_ROOT = Path(__file__).resolve().parents[2]

_SESSIONS: dict[tuple[str, str, str], Env] = {}


# ── the environment handed to the notebook ──────────────────────────────────


@dataclass
class Env:
    """Everything the notebook body is allowed to know about its context."""

    slug: str
    lang: str
    profile: str
    cfg: dict[str, Any]
    runtime: Runtime
    assets: dict[str, Path]
    spec: dict[str, Any] = field(repr=False, default_factory=dict)
    started: float = field(default_factory=time.time, repr=False)
    _hints_used: int = field(default=0, repr=False)
    _checks: dict[str, bool] = field(default_factory=dict, repr=False)

    # ── clause 5: the glossary, shared with the site ────────────────────────
    def term(self, key: str) -> str:
        """The workshop's own gloss for a term, in the notebook's language.

        Terms live in the workshop YAML rather than being fetched from the
        site at runtime: a notebook must work with no network once its assets
        are down, and a term that silently fails to resolve is worse than one
        that was written into the file.
        """
        entry = (self.spec.get("terms") or {}).get(key)
        if not entry:
            return key
        return entry.get(self.lang) or entry.get("en") or key

    def explain(self, key: str) -> None:
        """Print a term and its gloss — the notebook equivalent of the site's
        <Term> tag, which is the affordance Arabic readers rely on most."""
        gloss = self.term(key)
        if self.lang == "ar":
            print(f"{ltr(key)} — {gloss}")
        else:
            print(f"{key} — {gloss}")

    # ── clause 6: progressive hints ─────────────────────────────────────────
    def hint(self, n: int | None = None) -> None:
        """Print the next hint, or a specific one. Mirrors the Lab's pill."""
        hints = self.spec.get("hints") or []
        if not hints:
            print(
                "No hints for this workshop."
                if self.lang == "en"
                else "لا توجد تلميحات لهذه الورشة."
            )
            return
        index = self._hints_used if n is None else n - 1
        index = max(0, min(index, len(hints) - 1))
        self._hints_used = max(self._hints_used, index + 1)
        text = hints[index].get(self.lang) or hints[index].get("en", "")
        label = (
            f"تلميح {index + 1} من {len(hints)}"
            if self.lang == "ar"
            else f"Hint {index + 1} of {len(hints)}"
        )
        print(f"{label}\n{text}")

    # ── clause 7: self-verification ─────────────────────────────────────────
    def check(self, check_id: str, result: float | bool) -> bool:
        """Hold a measured result against the workshop's declared threshold.

        Nothing leaves the runtime. The threshold is in the YAML, so the
        learner can read it — this is a check, not an exam, and hiding the bar
        would only teach them to guess at it.
        """
        checks = {c["id"]: c for c in (self.spec.get("checks") or [])}
        spec = checks.get(check_id)
        if spec is None:
            print(f"Unknown check: {check_id}")
            return False

        if isinstance(result, bool):
            passed = result
            readout = str(result)
        else:
            value = float(result)
            lo = spec.get("min")
            hi = spec.get("max")
            passed = (lo is None or value >= float(lo)) and (hi is None or value <= float(hi))
            # The readout is graded feedback, not a log line, so it is
            # localized. The Arabic build caught this leaking English on its
            # first run — the only reason to build both languages early.
            bounds = []
            if lo is not None:
                bounds.append(f"≥ {lo}")
            if hi is not None:
                bounds.append(f"≤ {hi}")
            joiner = " و" if self.lang == "ar" else " and "
            needs = "المطلوب" if self.lang == "ar" else "needs"
            readout = f"{value:.4g} ({needs} {joiner.join(bounds)})"

        self._checks[check_id] = passed
        label = spec.get("label", {}).get(self.lang) or spec.get("label", {}).get("en") or check_id
        mark = "✓" if passed else "✗"
        print(f"{mark} {label}: {readout}")
        return passed

    # ── clause 8: the receipt ───────────────────────────────────────────────
    def receipt(self, write: bool = True) -> dict[str, Any]:
        """Emit proof of a completed run.

        The completion code is a short digest over (slug, code hash, the set of
        checks that passed). It is deliberately NOT a secret — anyone who wants
        to skip the workshop can already just not do it. What it is, is
        specific: a code from a run where the checks failed does not match a
        code from a run where they passed, so pasting it into the site records
        something true.
        """
        required = [c["id"] for c in (self.spec.get("checks") or []) if c.get("required", True)]
        passed = [cid for cid in required if self._checks.get(cid)]
        # A check the workshop DECLARED but the code never CALLED is a
        # different problem from a check that ran and failed, and it has a
        # different fix: the notebook is out of step with its workshop.yaml,
        # usually because only half a change was applied. Reporting both as
        # "still failing" sends the reader to tune a threshold when what they
        # need is to rebuild.
        never_ran = [cid for cid in required if cid not in self._checks]
        complete = len(required) > 0 and len(passed) == len(required)

        payload = {
            "slug": self.slug,
            "lang": self.lang,
            "profile": self.profile,
            "codeHash": self.spec.get("codeHash", ""),
            "checks": {cid: bool(self._checks.get(cid)) for cid in required},
            "complete": complete,
            "hintsUsed": self._hints_used,
            "elapsedSec": round(time.time() - self.started, 1),
            "device": self.runtime.device_name,
            "torch": self.runtime.torch_version,
            "python": platform.python_version(),
            "shim": __version__,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        seed = f"{self.slug}|{payload['codeHash']}|{'|'.join(sorted(passed))}"
        payload["code"] = "AZ-" + hashlib.sha256(seed.encode()).hexdigest()[:10].upper()

        if write:
            out = Path("azimuth-receipt.json")
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if complete:
            if self.lang == "ar":
                print("اكتملت الورشة.\n")
                print(f"رمز الإتمام: {ltr(payload['code'])}")
                print("الصقه في صفحة الورشة على أزيموث لتسجيل إتمامها.")
            else:
                print("Workshop complete.\n")
                print(f"Completion code: {payload['code']}")
                print("Paste it on the workshop's page on Azimuth to record it.")
        else:
            failed = [c for c in required if c in self._checks and not self._checks[c]]
            if self.lang == "ar":
                if failed:
                    print(f"لم تكتمل بعد — أخفق: {ltr(', '.join(failed))}")
                if never_ran:
                    print(f"لم تُستدعَ قط: {ltr(', '.join(never_ran))}")
                    print(
                        "  هذه الفحوص معرَّفة في workshop.yaml ولا تستدعيها code.py — "
                        "الدفتر غير متوافق مع مصدره. أعد البناء: "
                        + ltr("python tools/build_notebooks.py")
                    )
            else:
                if failed:
                    print(f"Not complete yet — failed: {', '.join(failed)}")
                if never_ran:
                    print(f"Never called: {', '.join(never_ran)}")
                    print(
                        "  These checks are declared in workshop.yaml but code.py does "
                        "not call them — the notebook is out of step with its source. "
                        "Rebuild: python tools/build_notebooks.py"
                    )
        return payload


# ── clause 1: setup ─────────────────────────────────────────────────────────


def _load_spec(slug: str, lang: str) -> dict[str, Any]:
    path = REPO_ROOT / "workshops" / slug / "workshop.yaml"
    if not path.exists():
        raise fail("AZ-E401", lang=lang, slug=slug)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    code_path = REPO_ROOT / "workshops" / slug / "code.py"
    if code_path.exists():
        # Normalized, so the hash a learner's receipt carries matches the one
        # CI and the site computed — a Windows checkout would otherwise never
        # produce a matching completion code.
        code_bytes = code_path.read_bytes().replace(b"\r\n", b"\n")
        spec["codeHash"] = hashlib.sha256(code_bytes).hexdigest()[:16]
    return spec


def _resolve_profile(spec: dict[str, Any], profile: str, lang: str) -> dict[str, Any]:
    profiles = {p["id"]: p for p in (spec.get("profiles") or [])}
    if profile not in profiles:
        raise fail("AZ-E402", lang=lang, asked=profile, known=", ".join(sorted(profiles)))
    return profiles[profile]


def setup(
    slug: str,
    lang: str = "en",
    profile: str = "free",
    quiet: bool = False,
) -> Env:
    """Prepare the runtime for a workshop and return its Env.

    Idempotent (clause 1): calling it twice with the same arguments returns the
    same Env and re-downloads nothing.
    """
    lang = lang if lang in ("en", "ar") else "en"
    key = (slug, lang, profile)
    if key in _SESSIONS:
        return _SESSIONS[key]

    spec = _load_spec(slug, lang)
    prof = _resolve_profile(spec, profile, lang)

    runtime = inspect_runtime(lang=lang)  # clause 2: asserts torch
    check_profile(runtime, prof, lang=lang)  # clause 3: refuses early

    if not quiet:
        header = spec.get("title", {}).get(lang) or spec.get("title", {}).get("en") or slug
        print(header)
        print(runtime.summary(lang))
        print(f"profile: {profile}" if lang == "en" else f"الملف: {ltr(profile)}")

    assets: dict[str, Path] = {}
    declared = spec.get("assets") or []
    if declared and not quiet:
        print("assets:" if lang == "en" else "البيانات:")
    data_dir = Path(os.environ.get("AZIMUTH_DATA_DIR", "data"))
    for asset in declared:
        assets[asset["name"]] = fetch(asset, data_dir, lang=lang, quiet=quiet)

    env = Env(
        slug=slug,
        lang=lang,
        profile=profile,
        cfg=dict(prof.get("scale") or {}),  # clause 4: scale is data
        runtime=runtime,
        assets=assets,
        spec=spec,
    )
    _SESSIONS[key] = env

    if not quiet:
        knobs = ", ".join(f"{k}={v}" for k, v in env.cfg.items())
        # The codeHash, printed. It is the only thing in the output that says
        # WHICH VERSION of the workshop just ran, and a notebook opened from
        # the `stable` badge can easily be older than the source someone is
        # editing. Three separate rounds of debugging were spent discovering
        # from the numbers that an old notebook had been run; the hash says it
        # in the first cell, before anything is measured.
        code_hash = spec.get("codeHash", "?")
        # The notebook stamps the hash it was BUILT from; `code_hash` is what
        # is on disk now. Equal is the only healthy state.
        built_from = os.environ.get("AZIMUTH_NOTEBOOK_CODEHASH")
        if built_from and built_from != code_hash:
            if lang == "ar":
                print(
                    f"\n⚠ هذا الدفتر بُني من الشيفرة {ltr(built_from)} بينما "
                    f"code.py على القرص {ltr(code_hash)}.\n"
                    "  الخلايا أقدم من مصدرها — أعد البناء قبل الوثوق بأي رقم:\n"
                    "  " + ltr("python tools/build_notebooks.py")
                )
            else:
                print(
                    f"\n⚠ this notebook was built from code {built_from}, but "
                    f"code.py on disk is {code_hash}.\n"
                    "  The cells are older than their source — rebuild before "
                    "trusting any number below:\n"
                    "  python tools/build_notebooks.py"
                )
        print(f"ready · {knobs}" if lang == "en" else f"جاهز · {ltr(knobs)}")
        print(f"code · {code_hash}" if lang == "en" else f"الشيفرة · {ltr(code_hash)}")
        sys.stdout.flush()
    return env
