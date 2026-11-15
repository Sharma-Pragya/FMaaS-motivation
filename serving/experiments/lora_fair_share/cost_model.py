"""Profiled per-token step costs from the lora_cost_attribution sweep.

Kept free of vLLM/torch imports so plot.py can use it on any machine.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

PROFILE_DIR = Path(__file__).resolve().parent.parent / "lora_cost_attribution"
DEFAULT_PROFILE = PROFILE_DIR / "results_t4_sweep" / "raw.jsonl"

# Which tenant's requests carry the rank-64 adapter. Tenant ids stay P and A
# in every case; in "both_plain" A has no adapter, in "both_adapter" P has one.
CASES = {
    "mixed":        {"P": False, "A": True},
    "both_adapter": {"P": True,  "A": True},
    "both_plain":   {"P": False, "A": False},
}


class Profile:
    """base(b) = config B step time; adapter(b) = (F1 - B) / b per request.

    `has_adapter` (one of CASES) decides which tenants pay the adapter terms.
    """

    def __init__(self, path: Path, prompt_len: int, has_adapter: dict | None = None):
        self.has_adapter = has_adapter or CASES["mixed"]
        rows = {}
        for line in open(path):
            r = json.loads(line)
            key = r["config"] + (str(r["k"]) if r["k"] else "")
            rows[(r["batch_size"], key)] = r
        self.bs = sorted(b for b, k in rows if k == "B" and (b, "F1") in rows)
        self.base_ms = [rows[(b, "B")]["tpot_ms_median"] for b in self.bs]
        self.adapter_ms = [(rows[(b, "F1")]["tpot_ms_median"]
                            - rows[(b, "B")]["tpot_ms_median"]) / b for b in self.bs]

        def ttft(key):
            return statistics.median(x["ttft_ms_median"]
                                     for x in rows[(self.bs[0], key)]["reps"])
        self.prefill_tok_ms = {t: ttft("F1" if self.has_adapter[t] else "B") / prompt_len
                               for t in ("P", "A")}

    def base(self, b: float) -> float:
        return float(np.interp(b, self.bs, self.base_ms))

    def adapter(self, b: float) -> float:
        return float(np.interp(b, self.bs, self.adapter_ms))

    def decode_tok_ms(self, tenant: str, b: float) -> float:
        return self.base(b) / b + (self.adapter(b) if self.has_adapter[tenant] else 0.0)

    def m_A(self, b: float) -> float:
        """A:P per-token decode cost ratio at batch size b (tuned-VTC multiplier)."""
        return self.decode_tok_ms("A", b) / self.decode_tok_ms("P", b)
