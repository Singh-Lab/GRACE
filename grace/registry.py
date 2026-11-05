"""Task registry: configuration, starting sequences, and model-file resolution.

A task is described entirely by `configs/tasks/<key>.yaml`. Nothing in this
package hard-codes a tolerance, a learning rate or an oracle path; change the
YAML and the change reaches the CLI, the Python API and the reproduction
scripts at once.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs" / "tasks"
STARTS_DIR = ROOT / "data" / "starts"

#: Where the trained oracles live. Override with GRACE_MODELS to keep a large
#: model directory outside the repository (see models/README.md).
MODEL_DIR = Path(os.environ.get("GRACE_MODELS", ROOT / "models"))

#: Runners that need weights or packages beyond the default bundle.
EXTERNAL_RUNNERS = {
    "esm": "ESM-2 fine-tuned oracles (Zenodo record, `esm/`, 3.8 GB)",
    "pangolin": "the Pangolin package and its released weights",
}

#: Runners that go through the generic core but prefix their oracle names.
#: `parm:induction` resolves to <GRACE_MODELS>/parm/induction.pt.
PREFIXED_RUNNERS = {"parm"}


class TaskError(RuntimeError):
    """Raised when a task is misconfigured or its models are missing."""


@dataclass
class Constraint:
    name: str
    oracle: str
    eps: float
    units: str = ""

    @property
    def is_builtin(self) -> bool:
        return self.oracle.startswith("builtin:")


@dataclass
class Task:
    key: str
    title: str
    domain: str
    runner: str
    alphabet: str
    target_sign: float
    length: Any
    K: int
    steps: int
    lr: float
    rho: float
    dual_lr: float
    active_set_tol: float
    target: dict
    constraints: list[Constraint] = field(default_factory=list)

    # ---------------------------------------------------------------- paths
    @property
    def model_dir(self) -> Path:
        if self.runner in PREFIXED_RUNNERS:
            return MODEL_DIR / self.runner
        return MODEL_DIR / self.key

    def resolve_oracle(self, spec: str) -> str:
        """Turn a config oracle string into what the core expects.

        `builtin:*` passes through; `parm:induction` becomes
        `parm:<models>/parm/induction.pt`; anything else is a file in the
        task's own model directory.
        """
        if spec.startswith("builtin:"):
            return spec
        for pref in PREFIXED_RUNNERS:
            if spec.startswith(pref + ":"):
                return f"{pref}:{MODEL_DIR / pref / (spec.split(':', 1)[1] + '.pt')}"
        return str(self.model_dir / spec)

    @property
    def starts_file(self) -> Path:
        return STARTS_DIR / f"{self.key}.txt"

    def starts(self) -> list[str]:
        """The benchmark starting sequences, in their published order."""
        if not self.starts_file.exists():
            raise TaskError(
                f"No starting sequences shipped for {self.key}. Its starts are "
                f"genomic intervals rather than free sequences; docs/TASKS.md "
                f"gives the extraction recipe.")
        return [s for s in self.starts_file.read_text().split() if s]

    def start(self, index: int) -> str:
        """Benchmark start `index`, numbered from 1 as in the paper."""
        s = self.starts()
        if not 1 <= index <= len(s):
            raise TaskError(f"{self.key} has {len(s)} starts; asked for #{index}.")
        return s[index - 1]

    # ----------------------------------------------------------- readiness
    def missing_models(self) -> list[str]:
        """Model files this task needs that are not on disk."""
        if self.runner in PREFIXED_RUNNERS:
            want = {o.split(":", 1)[1] for o in
                    [self.target["oracle"]] + [c.oracle for c in self.constraints]
                    if o.startswith(self.runner + ":")}
            return [f"{self.runner}/{w}.pt" for w in sorted(want)
                    if not (MODEL_DIR / self.runner / f"{w}.pt").exists()]
        if self.runner == "generic":
            want = [self.target["oracle"]] + [c.oracle for c in self.constraints
                                              if not c.is_builtin]
            return [w for w in want if not (self.model_dir / w).exists()]
        if self.runner == "dna":
            cells = {o.split(":")[1] for o in
                     [self.target["oracle"]] + [c.oracle for c in self.constraints]
                     if o.startswith("dna_cnn:")}
            return [f"dna_cnn/{c}/best_model.pt" for c in sorted(cells)
                    if not (MODEL_DIR / "dna_cnn" / c / "best_model.pt").exists()]
        return [] if self._external_ready() else [EXTERNAL_RUNNERS[self.runner]]

    def _external_ready(self) -> bool:
        if self.runner == "pangolin":
            try:
                import pangolin  # noqa: F401
                return True
            except ImportError:
                return False
        return (MODEL_DIR / "esm").is_dir()

    def describe(self) -> str:
        cons = "\n".join(
            f"    - {c.name}: within +/-{c.eps} {c.units}".rstrip()
            for c in self.constraints)
        return (f"{self.key}  ({self.domain}, length {self.length})\n"
                f"  {self.title}\n"
                f"  maximise: {self.target['name']} [{self.target['units']}]\n"
                f"  subject to:\n{cons}\n"
                f"  published settings: K={self.K} steps={self.steps} "
                f"lr={self.lr} rho={self.rho}")


def _load(path: Path) -> Task:
    raw = yaml.safe_load(path.read_text())
    cons = [Constraint(**c) for c in raw.pop("constraints", [])]
    return Task(constraints=cons, **raw)


def list_tasks() -> list[str]:
    """Every task key, sorted by domain then name."""
    ts = [_load(p) for p in sorted(CONFIG_DIR.glob("*.yaml"))]
    order = {"rna": 0, "dna": 1, "protein": 2}
    return [t.key for t in sorted(ts, key=lambda t: (order.get(t.domain, 9), t.key))]


def load_task(key: str) -> Task:
    p = CONFIG_DIR / f"{key}.yaml"
    if not p.exists():
        raise TaskError(f"Unknown task {key!r}. Known tasks: {', '.join(list_tasks())}")
    return _load(p)
