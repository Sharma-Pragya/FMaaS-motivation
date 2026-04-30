#!/usr/bin/env python3
"""Generate base deployment plans for the workload-adaptation experiment.

Wraps the cluster_sharing_benefit generator so we don't duplicate plan-builder
logic. Selects which case config (case1 / case2) to load via CLI arg or
WORKLOAD_ADAPT_CASE env var.

Usage:
    python -m experiments.workload_adaptation.deployments.generate --case case1
    python -m experiments.workload_adaptation.deployments.generate --case case2
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

# The cluster_sharing_benefit generator imports `user_config` by package path.
# We import it as a module, then monkeypatch its `uc` attribute to point at
# the case-specific config. This is the cleanest way to reuse builders
# without copy-pasting the file.
from experiments.cluster_sharing_benefit.deployments import generate as csb_gen

OUT_DIR = Path(__file__).resolve().parent


def _load_case_config(case: str):
    mod = importlib.import_module(f"experiments.workload_adaptation.configs.{case}")
    return mod


def generate_case(case: str) -> Path:
    cfg = _load_case_config(case)

    # Point csb_gen at our config and our output dir for the duration of
    # this call. csb_gen.generate_all() reads OUT_DIR module-level constant
    # to decide where to write plans.
    saved_uc = csb_gen.uc
    saved_out = csb_gen.OUT_DIR
    csb_gen.uc = cfg
    csb_gen.OUT_DIR = OUT_DIR / case
    try:
        csb_gen.OUT_DIR.mkdir(parents=True, exist_ok=True)
        csb_gen.generate_all()
    finally:
        csb_gen.uc = saved_uc
        csb_gen.OUT_DIR = saved_out

    n = cfg.n_apps_list[0]
    plan_path = OUT_DIR / case / f"N{n}" / "sharing.json"
    if not plan_path.is_file():
        raise RuntimeError(f"generator did not produce {plan_path}")
    print(f"[workload_adaptation] base plan -> {plan_path}")
    return plan_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                    choices=["case1", "case2", "case3", "case4"])
    args = ap.parse_args()
    generate_case(args.case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
