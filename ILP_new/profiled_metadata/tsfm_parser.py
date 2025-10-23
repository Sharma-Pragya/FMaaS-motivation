# TSFMParser.py
# Usage: python TSFMParser.py
# Reads TSFM.csv and writes two PURE-DATA files:
#   - config.py   (DEVICES, DEVICE_MEM_MB, TASKS, BACKBONES, DECODERS, BH_COMBOS)
#   - profiler.py (CAN_SERVE, LATENCY_MS, THROUGHPUT_QPS, ACCURACY, INFERENCE_PEAK_MEM_MB,
#                  BACKBONE_MEM_MB, DECODER_MEM_MB, FEASIBLE)
#
# Changes per your request:
# - Latency is taken from "inference time" in **milliseconds** (no x1000 error).
#   Only if header explicitly indicates seconds do we convert to ms.
# - Throughput recomputed from corrected latency if missing.
# - ACCURACY uses only CSV "result" (device-agnostic); aggregates by mean if duplicates.
# - Adds INFERENCE_PEAK_MEM_MB (device-agnostic); aggregates by max if duplicates.
# - config.py has a blank line after TASKS list to improve readability.

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import csv, os, re, math, pprint

CSV_NAME = "TSFM.csv"

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("_", " ").split())

ALIASES = {
    "backbone": {"backbone", "model", "base", "encoder", "bb"},
    "decoder": {"decoder", "head", "mlp", "classifier"},
    "task_name": {"task", "task name", "target", "objective"},
    "dataset_name": {"dataset", "dataset name"},
    "device": {"device", "gpu", "accelerator"},
    # residency memory
    "backbone_mem": {"backbone memory", "backbone vram", "encoder memory", "bb memory"},
    "decoder_mem": {"decoder memory", "head memory", "mlp memory"},
    # runtime
    "latency": {"inference time", "latency", "latency ms", "time per request"},
    "throughput": {"throughput", "qps", "req/s", "requests per second"},
    # accuracy (you asked to use only this)
    "result": {"result", "final result", "score"},
    # peak runtime memory
    "infer_peak_mem": {"inference peak memory", "peak inference memory", "inference mem peak"},
}

ERROR_METRICS = {"mae", "mse", "rmse", "mape", "smape"}  # (kept for potential future use)

_num_re = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

def _parse_number(s: Any) -> Optional[float]:
    if s is None: return None
    if isinstance(s, (int, float)): return float(s)
    txt = str(s).strip()
    if not txt: return None
    m = _num_re.search(txt)
    return float(m.group(0)) if m else None

def _header_has_ms(header_label: str) -> bool:
    hl = _norm(header_label)
    return " ms" in (" " + hl + " ") or "(ms)" in hl or hl.endswith("ms")

def _header_has_seconds(header_label: str) -> bool:
    hl = _norm(header_label)
    return " s " in (" " + hl + " ") or "(s)" in hl or " second" in hl or hl.endswith("s")

def _mem_in_mb(val: Optional[float], header_label: str) -> Optional[float]:
    if val is None: return None
    hl = _norm(header_label)
    if "gb" in hl or "gib" in hl or "gigabyte" in hl:
        return float(val) * 1024.0
    return float(val)  # assume MB otherwise

def _latency_in_ms_from_csv(val: Optional[float], header_label: str) -> Optional[float]:
    """Treat CSV inference time as **milliseconds** unless header explicitly marks seconds."""
    if val is None: return None
    if _header_has_seconds(header_label):
        return float(val) * 1000.0
    # default: the file uses ms already
    return float(val)

def _normalize_device(dev: str) -> str:
    d = _norm(dev)
    if "a100" in d or "cuda:" in d or d == "cuda":
        return "A100"
    return dev.strip() or "A100"

def _build_colmap(header: List[str]) -> Dict[str, int]:
    idxmap: Dict[str, int] = {}
    hn = [_norm(h) for h in header]

    def find_one(keys: set[str]) -> Optional[int]:
        for i, h in enumerate(hn):
            if h in keys: return i
        return None

    def bind(name: str, aliases: set[str]):
        i = find_one(aliases)
        if i is not None:
            idxmap[name] = i

    bind("backbone", ALIASES["backbone"])
    bind("decoder", ALIASES["decoder"])
    bind("task_name", ALIASES["task_name"])
    bind("dataset_name", ALIASES["dataset_name"])
    bind("device", ALIASES["device"])
    bind("backbone_mem", ALIASES["backbone_mem"])
    bind("decoder_mem", ALIASES["decoder_mem"])
    bind("latency", ALIASES["latency"])
    bind("throughput", ALIASES["throughput"])
    bind("result", ALIASES["result"])
    bind("infer_peak_mem", ALIASES["infer_peak_mem"])
    return idxmap

def main():
    if not os.path.exists(CSV_NAME):
        raise FileNotFoundError(f"Could not find {CSV_NAME} in {os.getcwd()}")

    with open(CSV_NAME, "r", encoding="utf-8-sig", errors="replace") as f:
        rdr = csv.reader(f)
        try:
            header = next(rdr)
        except StopIteration:
            raise ValueError("TSFM.csv appears to be empty.")
        rows = [r for r in rdr if r and len(r) == len(header)]

    col = _build_colmap(header)

    for req in ("backbone", "decoder", "device"):
        if req not in col:
            raise ValueError(f"TSFM.csv missing required column matching {req!r}")

    DEVICES: set[str] = set()
    TASKS: set[str] = set()
    BACKBONES: set[str] = set()
    DECODERS: set[str] = set()
    BH_COMBOS: set[Tuple[str, str]] = set()

    CAN_SERVE: Dict[str, set[Tuple[str, str]]] = {}

    LATENCY_MS: Dict[Tuple[str, str, str, str], float] = {}
    THROUGHPUT_QPS: Dict[Tuple[str, str, str, str], float] = {}

    # device-agnostic
    ACCURACY: Dict[Tuple[str, str, str], float] = {}
    _acc_count: Dict[Tuple[str, str, str], int] = {}  # for mean aggregation
    INFERENCE_PEAK_MEM_MB: Dict[Tuple[str, str, str], float] = {}

    BACKBONE_MEM_MB: Dict[str, float] = {}
    DECODER_MEM_MB: Dict[str, float] = {}

    FEASIBLE: Dict[Tuple[str, str, str, str], int] = {}

    for r in rows:
        backbone = r[col["backbone"]].strip()
        decoder = r[col["decoder"]].strip()

        # prefer task_name else dataset_name; no dataset prefix in IDs
        task = ""
        if "task_name" in col:
            task = (r[col["task_name"]] or "").strip()
        if not task and "dataset_name" in col:
            task = (r[col["dataset_name"]] or "").strip()
        if not task:
            task = "unknown_task"

        device = _normalize_device(r[col["device"]])

        DEVICES.add(device)
        TASKS.add(task)
        BACKBONES.add(backbone)
        DECODERS.add(decoder)
        BH_COMBOS.add((backbone, decoder))
        CAN_SERVE.setdefault(task, set()).add((backbone, decoder))
        FEASIBLE[(task, backbone, decoder, device)] = 1

        # residency memory (max across rows)
        if "backbone_mem" in col:
            v = _parse_number(r[col["backbone_mem"]])
            v = _mem_in_mb(v, header[col["backbone_mem"]]) if v is not None else None
            if v is not None:
                BACKBONE_MEM_MB[backbone] = v if backbone not in BACKBONE_MEM_MB else max(BACKBONE_MEM_MB[backbone], v)
        if "decoder_mem" in col:
            v = _parse_number(r[col["decoder_mem"]])
            v = _mem_in_mb(v, header[col["decoder_mem"]]) if v is not None else None
            if v is not None:
                DECODER_MEM_MB[decoder] = v if decoder not in DECODER_MEM_MB else max(DECODER_MEM_MB[decoder], v)

        # latency (ms) & throughput (qps)
        lat_ms: Optional[float] = None
        if "latency" in col:
            lv = _parse_number(r[col["latency"]])
            if lv is not None:
                lat_ms = _latency_in_ms_from_csv(lv, header[col["latency"]])
        qps: Optional[float] = None
        if "throughput" in col:
            tv = _parse_number(r[col["throughput"]])
            if tv is not None:
                qps = float(tv)

        # derive the missing side if possible
        if lat_ms is None and qps is not None and qps > 0:
            lat_ms = 1000.0 / qps
        if qps is None and lat_ms is not None and lat_ms > 0:
            qps = 1000.0 / lat_ms

        k_tbhd = (task, backbone, decoder, device)
        if lat_ms is not None:
            LATENCY_MS[k_tbhd] = min(LATENCY_MS[k_tbhd], float(lat_ms)) if k_tbhd in LATENCY_MS else float(lat_ms)
        if qps is not None:
            THROUGHPUT_QPS[k_tbhd] = max(THROUGHPUT_QPS[k_tbhd], float(qps)) if k_tbhd in THROUGHPUT_QPS else float(qps)

        # accuracy: only 'result', device-agnostic; aggregate by mean
        if "result" in col:
            rv = _parse_number(r[col["result"]])
            if rv is not None:
                k_tbh = (task, backbone, decoder)
                if k_tbh in ACCURACY:
                    ACCURACY[k_tbh] = (ACCURACY[k_tbh] * _acc_count[k_tbh] + rv) / (_acc_count[k_tbh] + 1)
                    _acc_count[k_tbh] += 1
                else:
                    ACCURACY[k_tbh] = rv
                    _acc_count[k_tbh] = 1

        # inference peak memory: device-agnostic; aggregate by max
        if "infer_peak_mem" in col:
            pv = _parse_number(r[col["infer_peak_mem"]])
            if pv is not None:
                k_tbh = (task, backbone, decoder)
                if k_tbh in INFERENCE_PEAK_MEM_MB:
                    INFERENCE_PEAK_MEM_MB[k_tbh] = max(INFERENCE_PEAK_MEM_MB[k_tbh], pv)
                else:
                    INFERENCE_PEAK_MEM_MB[k_tbh] = pv

    # sorted, deterministic
    devices_sorted = sorted(DEVICES)
    tasks_sorted = sorted(TASKS)
    backbones_sorted = sorted(BACKBONES)
    decoders_sorted = sorted(DECODERS)
    bh_sorted = sorted(BH_COMBOS)

    # --------- write config.py ----------
    device_mem = {}
    for d in devices_sorted:
        device_mem[d] = 40960.0 if d == "A100" else 0.0  # A100 = 40GB cap as requested

    pp = pprint.PrettyPrinter(indent=2, width=100, sort_dicts=True)

    cfg_lines: List[str] = []
    cfg_lines.append("# THIS FILE IS AUTO-GENERATED by TSFMParser.py — DO NOT EDIT\n")
    cfg_lines.append("# Pure data: devices/memory and workload sets.\n\n")
    cfg_lines.append(f"DEVICES = {pp.pformat(devices_sorted)}\n")
    cfg_lines.append(f"DEVICE_MEM_MB = {pp.pformat(device_mem)}\n\n")
    cfg_lines.append(f"TASKS = {pp.pformat(tasks_sorted)}\n")
    cfg_lines.append("\n")  # <<< extra blank line after TASKS (your readability request)
    cfg_lines.append(f"BACKBONES = {pp.pformat(backbones_sorted)}\n")
    cfg_lines.append(f"DECODERS = {pp.pformat(decoders_sorted)}\n")
    cfg_lines.append(f"BH_COMBOS = {pp.pformat(bh_sorted)}\n")

    with open("config.py", "w", encoding="utf-8") as f:
        f.write("".join(cfg_lines))

    # --------- write profiler.py ----------
    can_serve_out = {t: sorted(list(v)) for t, v in CAN_SERVE.items()}

    prof_lines: List[str] = []
    prof_lines.append("# THIS FILE IS AUTO-GENERATED by TSFMParser.py — DO NOT EDIT\n")
    prof_lines.append("# Pure data: feasibility and performance profiles for shared-backbone TSFMs.\n\n")

    prof_lines.append("CAN_SERVE = ")
    prof_lines.append(pp.pformat(can_serve_out))
    prof_lines.append("\n\n")

    prof_lines.append("# Keys: (task, backbone, decoder, device) -> latency in milliseconds\n")
    prof_lines.append("LATENCY_MS = ")
    prof_lines.append(pp.pformat({k: float(v) for k, v in LATENCY_MS.items()}))
    prof_lines.append("\n\n")

    prof_lines.append("# Keys: (task, backbone, decoder, device) -> throughput in QPS\n")
    prof_lines.append("THROUGHPUT_QPS = ")
    prof_lines.append(pp.pformat({k: float(v) for k, v in THROUGHPUT_QPS.items()}))
    prof_lines.append("\n\n")

    prof_lines.append("# Device-agnostic accuracy: (task, backbone, decoder) -> result\n")
    prof_lines.append("ACCURACY = ")
    prof_lines.append(pp.pformat({k: float(v) for k, v in ACCURACY.items()}))
    prof_lines.append("\n\n")

    prof_lines.append("# Device-agnostic peak inference memory (if present): (task, backbone, decoder) -> MB\n")
    prof_lines.append("INFERENCE_PEAK_MEM_MB = ")
    prof_lines.append(pp.pformat({k: float(v) for k, v in INFERENCE_PEAK_MEM_MB.items()}))
    prof_lines.append("\n\n")

    prof_lines.append("# Residency VRAM in MB for backbones/decoders when loaded\n")
    prof_lines.append("BACKBONE_MEM_MB = ")
    prof_lines.append(pp.pformat({k: float(v) for k, v in BACKBONE_MEM_MB.items()}))
    prof_lines.append("\n")
    prof_lines.append("DECODER_MEM_MB = ")
    prof_lines.append(pp.pformat({k: float(v) for k, v in DECODER_MEM_MB.items()}))
    prof_lines.append("\n\n")

    prof_lines.append("# Explicit feasibility: (task, backbone, decoder, device) -> 1\n")
    prof_lines.append("FEASIBLE = ")
    prof_lines.append(pp.pformat(FEASIBLE))
    prof_lines.append("\n")

    with open("profiler.py", "w", encoding="utf-8") as f:
        f.write("".join(prof_lines))

    print("[TSFMParser] Wrote profiler.py and config.py")
    print(f"  devices: {devices_sorted}")
    print(f"  counts: tasks={len(tasks_sorted)}, backbones={len(backbones_sorted)}, decoders={len(decoders_sorted)}, combos={len(bh_sorted)}")
    print(f"  entries: latency={len(LATENCY_MS)}, throughput={len(THROUGHPUT_QPS)}, accuracy_triples={len(ACCURACY)}, peak_mem_triples={len(INFERENCE_PEAK_MEM_MB)}")
    print(f"  residency: backbone_mem={len(BACKBONE_MEM_MB)}, decoder_mem={len(DECODER_MEM_MB)}")

if __name__ == "__main__":
    main()