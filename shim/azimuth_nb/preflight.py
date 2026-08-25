"""Check the runtime before the learner spends twenty minutes finding out.

The failure this exists to prevent: someone opens the notebook, runs every
cell, waits through a download and eight epochs, and the training cell dies on
an out-of-memory error at minute nineteen. Everything needed to predict that
was knowable in the first two seconds.

So preflight runs once, at setup, and refuses early and legibly.

TORCH IS ASSERTED, NEVER INSTALLED. Colab ships a PyTorch built against its
own driver; `pip install torch` on top of it fetches a wheel that may not
match, and the resulting CUDA errors name nothing that would lead a learner
back to the pip line that caused them. If torch is missing or too old, the
right answer is a fresh runtime, and AZ-E201/AZ-E202 say exactly that.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from .errors import fail

#: Oldest PyTorch these workshops are written against. Below this, `torch.compile`
#: and several `nn` defaults differ enough that the prose would be wrong.
MIN_TORCH = (2, 0)


@dataclass
class Runtime:
    """What we found, in the units the profile is written in."""

    accelerator: str  # 'cuda' | 'mps' | 'cpu'
    device_name: str
    vram_gb: float  # 0.0 when there is no accelerator
    ram_gb: float
    disk_gb: float
    torch_version: str

    def summary(self, lang: str = "en") -> str:
        if lang == "ar":
            gpu = f"{self.device_name} · {self.vram_gb:.1f} غ.ب" if self.accelerator == "cuda" else "بدون معالج رسوميات"
            return f"{gpu} · ذاكرة {self.ram_gb:.1f} غ.ب · PyTorch {self.torch_version}"
        gpu = f"{self.device_name} · {self.vram_gb:.1f} GB" if self.accelerator == "cuda" else "no GPU"
        return f"{gpu} · {self.ram_gb:.1f} GB RAM · PyTorch {self.torch_version}"


def _ram_gb() -> float:
    """Total system RAM. os.sysconf is present on Linux, which is every Colab
    runtime; anything else reports 0 and the RAM check is skipped rather than
    guessed at."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except (ValueError, OSError, AttributeError):
        return 0.0


def inspect_runtime(lang: str = "en") -> Runtime:
    """Look at the machine. Raises AZ-E201/E202 if torch is unusable."""
    try:
        import torch
    except ImportError as exc:
        raise fail("AZ-E201", lang=lang) from exc

    parts = torch.__version__.split(".")
    try:
        major_minor = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        major_minor = (0, 0)
    if major_minor < MIN_TORCH:
        raise fail(
            "AZ-E202",
            lang=lang,
            found=torch.__version__,
            needed=f"{MIN_TORCH[0]}.{MIN_TORCH[1]}+",
        )

    if torch.cuda.is_available():
        accelerator = "cuda"
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # Someone running this locally on a Mac. Not a target, but no reason
        # to refuse it when the workshop is small enough to fit.
        accelerator, device_name, vram_gb = "mps", "Apple GPU", 0.0
    else:
        accelerator, device_name, vram_gb = "cpu", "CPU", 0.0

    return Runtime(
        accelerator=accelerator,
        device_name=device_name,
        vram_gb=vram_gb,
        ram_gb=_ram_gb(),
        disk_gb=shutil.disk_usage("/").free / (1024**3),
        torch_version=torch.__version__,
    )


def check_profile(runtime: Runtime, profile: dict, lang: str = "en") -> None:
    """Hold the runtime against what the active profile declared it needs.

    `requiresGpu: false` is honoured: a workshop that runs fine on CPU should
    not refuse a CPU. That matters more than it sounds — it is what lets the
    site's own CI execute a workshop without a GPU runner when the workshop
    genuinely does not need one.
    """
    needs = profile.get("requires", {}) or {}

    if needs.get("gpu", True) and runtime.accelerator == "cpu":
        raise fail("AZ-E101", lang=lang)

    min_vram = float(needs.get("vramGb", 0) or 0)
    if min_vram and runtime.accelerator == "cuda" and runtime.vram_gb < min_vram:
        raise fail(
            "AZ-E102",
            lang=lang,
            needs=f"{min_vram:.0f} GB",
            found=f"{runtime.vram_gb:.1f} GB",
        )

    min_ram = float(needs.get("ramGb", 0) or 0)
    if min_ram and runtime.ram_gb and runtime.ram_gb < min_ram:
        raise fail(
            "AZ-E103",
            lang=lang,
            needs=f"{min_ram:.0f} GB",
            found=f"{runtime.ram_gb:.1f} GB",
        )
