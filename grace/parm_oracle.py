"""PARM promoter-activity oracles, for the heat-shock task.

A PARM model is an ensemble of five folds (PARM_predict averages them), takes a
600 bp one-hot window with the fragment CENTERED and zero padding either side,
and returns a Log2RPM promoter activity. Two kinds are exposed, both plain
nn.Modules with the signature the rest of the stack expects --
(B, 4, L_core) relaxed one-hot -> (B,) score -- so they drop straight into
`_core.py` alongside the CNN oracles:

  parm       : a single condition, e.g. f_control
  parm_diff  : f_a - f_b, used as the inducibility target f_heatshock - f_control

Only the L_core central positions receive gradient; the padding stays fixed, so
the optimiser engineers the promoter fragment rather than its flanks.

The *weights* are ours and ship in the Zenodo bundle under `parm/`. The *model
class* is PARM's own and is not redistributed here, so point PARM_REPO at a
checkout of https://github.com/vansteensellab/PARM (see docs/TASKS.md).

This file is deliberately named without a leading underscore: `_core.py` is
vendored unchanged from the paper's runner and does `from parm_oracle import
PARMOracle`, which resolves here because the core runs as a script inside this
directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

#: Where to find a checkout of the PARM repository.
PARM_REPO = os.environ.get(
    "PARM_REPO", str(Path(__file__).resolve().parent.parent / "third_party" / "PARM"))


def _import_parm():
    if PARM_REPO not in sys.path:
        sys.path.insert(0, PARM_REPO)
    try:
        from PARM.PARM_utils_load_model import ResNet_Attentionpool
    except ImportError as e:
        raise ImportError(
            "The heat-shock task needs PARM's model class, which we do not "
            "redistribute. Clone it and point PARM_REPO at the checkout:\n"
            "    git clone https://github.com/vansteensellab/PARM "
            "third_party/PARM\n"
            "    export PARM_REPO=$PWD/third_party/PARM\n"
            f"(tried {PARM_REPO})") from e
    return ResNet_Attentionpool


def _build(state, L_max=600, n_block=5, filter_size=125):
    ResNet_Attentionpool = _import_parm()
    m = ResNet_Attentionpool(L_max=L_max, n_block=n_block, filter_size=filter_size,
                             type_loss="poisson", validation=True,
                             index_interested_output=False, maxglobalpool=True,
                             use_AttentionPool=True)
    m.load_state_dict(state)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class _Ensemble(nn.Module):
    """Mean over the five PARM folds, matching PARM_predict."""

    def __init__(self, states, L_core=None, L_max=600):
        super().__init__()
        self.models = nn.ModuleList([_build(s, L_max=L_max) for s in states])
        self.L_core, self.L_max = L_core, L_max

    def forward(self, x):                      # x: (B, 4, L) for any L <= L_max
        if x.dim() == 2:
            x = x.unsqueeze(0)
        L = x.shape[-1]
        if L == self.L_max:
            buf = x
        else:
            buf = x.new_zeros(x.shape[0], 4, self.L_max)
            off = (self.L_max - L) // 2        # centered, matching PARM inference
            buf[:, :, off:off + L] = x
        return torch.stack([m(buf).squeeze(-1) for m in self.models]).mean(0)


class PARMOracle(nn.Module):
    """kind='parm' -> one condition.  kind='parm_diff' -> f_a - f_b."""

    def __init__(self, spec_path, device="cpu"):
        super().__init__()
        ck = torch.load(spec_path, map_location="cpu", weights_only=False)
        self.kind = ck["kind"]
        self.L_core = ck.get("L_core")
        self.name = ck.get("name", self.kind)
        L_max = ck.get("L_max", 600)
        if self.kind == "parm":
            self.a = _Ensemble(ck["weights"], self.L_core, L_max)
            self.b = None
        elif self.kind == "parm_diff":
            self.a = _Ensemble(ck["weights_a"], self.L_core, L_max)
            self.b = _Ensemble(ck["weights_b"], self.L_core, L_max)
        else:
            raise ValueError(f"unknown PARM oracle kind {self.kind}")
        self.to(device)
        self.eval()

    def forward(self, x):
        return self.a(x) if self.b is None else self.a(x) - self.b(x)

    def __call__(self, x):      # the core calls oracles as plain callables
        return self.forward(x)
