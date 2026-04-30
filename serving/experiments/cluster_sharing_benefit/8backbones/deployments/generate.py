#!/usr/bin/env python3
"""Generate deployment JSON plans for the cluster sharing-benefit sweep.

Reads user_config.py and emits, for each N in n_apps_list:
    deployments/N{N}/<condition>.json
    deployments/N{N}/assignment.json
    deployments/N{N}/task_meta.json   {synthetic_task: base_task}

Usage:
    python -m experiments.cluster_sharing_benefit.deployments.generate
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from experiments.cluster_sharing_benefit import user_config as uc


OUT_DIR = Path(__file__).resolve().parent
BASE_PORT = 8000
PORT_STEP = 10


# ── Selection & assignment ─────────────────────────────────────────

def _synthetic_task_name(task: str, app_idx: int, seen: set[str]) -> str:
    """Give repeated logical apps a unique routing key.

    The first occurrence keeps the original task name so existing single-app
    configs stay readable. Later repeats become e.g. "ecgclass__app10".
    """
    if task not in seen:
        seen.add(task)
        return task
    synth = f"{task}__app{app_idx}"
    seen.add(synth)
    return synth

def select_apps(n: int) -> List[dict]:
    pool = uc.task_pool
    out = []
    seen_tasks: set[str] = set()
    for i in range(n):
        entry = pool[i % len(pool)]
        logical_task = entry["task"]
        task = _synthetic_task_name(logical_task, i, seen_tasks)
        backbone = entry["backbone"]
        base_task = entry.get("base_task", logical_task)
        bb_short = backbone.replace("-patch", "")
        decoder_path = entry.get("decoder_path", f"{base_task}_{bb_short}_mlp")
        out.append({
            "task": task,
            "logical_task": logical_task,
            "backbone": backbone,
            "base_task": base_task,
            "decoder_path": decoder_path,
            "app_idx": i,
        })
    return out


def assign_to_gpus(apps: List[dict], n: int) -> List[List[dict]]:
    gpus = uc.gpus
    groups: List[List[dict]] = [[] for _ in gpus]
    if uc.assignment == "custom":
        layout = uc.custom_assignment.get(n)
        if layout is None:
            raise ValueError(f"assignment=custom but no layout for N={n}")
        if len(layout) != len(gpus):
            raise ValueError(f"custom layout for N={n} has {len(layout)} groups, expected {len(gpus)}")
        for gi, idxs in enumerate(layout):
            for ai in idxs:
                groups[gi].append(apps[ai])
    else:
        for i, app in enumerate(apps):
            groups[i % len(gpus)].append(app)
    return groups


# ── Device helpers ─────────────────────────────────────────────────

def _dev_meta(device_name: str) -> dict:
    if device_name not in uc.devices:
        raise KeyError(f"device {device_name!r} not in user_config.devices")
    return uc.devices[device_name]


def _decoder(app: dict) -> dict:
    base = app["base_task"]
    return {
        "task": app["task"],
        "base_task": base,
        "type": uc.tasks[base]["type"],
        "path": app["decoder_path"],
    }


def _task_entry(app: dict, rps: float) -> dict:
    return {"type": uc.tasks[app["base_task"]]["type"],
            "request_per_sec": float(rps)}


def _base_deployment(device_name: str, port: int, backbone: str,
                     scheduler: str) -> dict:
    meta = _dev_meta(device_name)
    return {
        "device": f"{meta['ip']}:{port}",
        "device_name": device_name,
        "device_type": meta["type"],
        "backbone": backbone,
        "cuda": meta["cuda"],
        "scheduler_policy": scheduler,
        "worker_mode": "inline",
        "decoders": [],
        "tasks": {},
    }


def _wrap_sites(deployments: List[dict]) -> dict:
    by_site: Dict[str, List[dict]] = {}
    for d in deployments:
        sid = _dev_meta(d["device_name"])["site_manager"]
        by_site.setdefault(sid, []).append(d)
    return {"sites": [{"id": sid, "deployments": deps} for sid, deps in by_site.items()]}


def _consecutive(start: int, count: int, total: int) -> List[int]:
    return [(start + k) % total for k in range(count)]


# ── Condition builders ─────────────────────────────────────────────

def build_no_sharing(groups: List[List[dict]]) -> dict:
    deps: List[dict] = []
    for gi, apps in enumerate(groups):
        device_name = uc.gpus[gi]
        for slot, app in enumerate(apps):
            port = BASE_PORT + slot * PORT_STEP
            d = _base_deployment(device_name, port, app["backbone"], "fifo")
            d["decoders"].append(_decoder(app))
            d["tasks"][app["task"]] = _task_entry(app, uc.per_app_rps)
            deps.append(d)
    return _wrap_sites(deps)


def build_no_sharing_tpc(groups: List[List[dict]], n: int) -> dict:
    split_map = getattr(uc, "tpc_per_app_split", {}).get(n, {})
    if n not in uc.tpc_per_app and not split_map:
        raise KeyError(f"tpc_per_app missing entry for N={n}")
    default_k = int(uc.tpc_per_app[n]) if n in uc.tpc_per_app else None
    deps: List[dict] = []
    for gi, apps in enumerate(groups):
        device_name = uc.gpus[gi]
        total = int(_dev_meta(device_name)["tpcs"])
        per_app_ks: List[int]
        if device_name in split_map:
            split = split_map[device_name]
            if len(split) != len(apps):
                raise ValueError(
                    f"tpc_per_app_split[{n}][{device_name!r}] has {len(split)} "
                    f"entries, expected {len(apps)}"
                )
            per_app_ks = [int(x) for x in split]
        else:
            if default_k is None:
                raise KeyError(
                    f"tpc_per_app[{n}] missing and no tpc_per_app_split for "
                    f"{device_name!r}"
                )
            per_app_ks = [default_k] * len(apps)

        if sum(per_app_ks) > total:
            raise ValueError(
                f"no_sharing_tpc N={n} {device_name!r}: requested "
                f"{sum(per_app_ks)} TPCs ({per_app_ks}) exceeds available {total}"
            )
        cursor = 0
        for slot, (app, k) in enumerate(zip(apps, per_app_ks)):
            port = BASE_PORT + slot * PORT_STEP
            d = _base_deployment(device_name, port, app["backbone"], "fifo")
            d["decoders"].append(_decoder(app))
            d["tasks"][app["task"]] = _task_entry(app, uc.per_app_rps)
            d["tpc_mode"] = "libsmctrl"
            d["tpc_partition"] = list(range(cursor, cursor + k))
            cursor += k
            deps.append(d)
    return _wrap_sites(deps)


def build_sharing(groups: List[List[dict]], n: int) -> dict:
    deps: List[dict] = []
    split_map = uc.sharing_tpc_split.get(n, {}) if uc.sharing_tpc_split else {}

    for gi, apps in enumerate(groups):
        device_name = uc.gpus[gi]
        total = int(_dev_meta(device_name)["tpcs"])
        if not apps:
            continue

        order: List[str] = []
        buckets: Dict[str, List[dict]] = {}
        for app in apps:
            bb = app["backbone"]
            if bb not in buckets:
                buckets[bb] = []
                order.append(bb)
            buckets[bb].append(app)

        multi = len(order) > 1
        if multi:
            if device_name not in split_map:
                raise KeyError(
                    f"sharing_tpc_split[{n}][{device_name!r}] required: "
                    f"{len(order)} backbones co-locate ({order})"
                )
            split = split_map[device_name]
            if len(split) != len(order):
                raise ValueError(
                    f"sharing_tpc_split[{n}][{device_name!r}] has {len(split)} "
                    f"entries, expected {len(order)} (backbones: {order})"
                )

        cursor = 0
        for bi, backbone in enumerate(order):
            group_apps = buckets[backbone]
            port = BASE_PORT + bi * PORT_STEP
            d = _base_deployment(device_name, port, backbone, "stfq")
            d["worker_mode"] = "threaded"
            for app in group_apps:
                d["decoders"].append(_decoder(app))
                d["tasks"][app["task"]] = _task_entry(app, uc.per_app_rps)
            if multi:
                k = int(split[bi])
                d["tpc_mode"] = "libsmctrl"
                d["tpc_partition"] = _consecutive(cursor, k, total)
                cursor += k
            deps.append(d)
    return _wrap_sites(deps)


BUILDERS = {
    "sharing":         lambda groups, n: build_sharing(groups, n),
    "no_sharing":      lambda groups, n: build_no_sharing(groups),
    "no_sharing_tpc":  lambda groups, n: build_no_sharing_tpc(groups, n),
}


def generate_all() -> None:
    for n in uc.n_apps_list:
        apps = select_apps(n)
        groups = assign_to_gpus(apps, n)
        n_dir = OUT_DIR / f"N{n}"
        n_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "n_apps": n,
            "gpus": uc.gpus,
            "apps_per_gpu": [
                [{"task": a["task"], "backbone": a["backbone"],
                  "logical_task": a["logical_task"],
                  "base_task": a["base_task"], "decoder_path": a["decoder_path"],
                  "app_idx": a["app_idx"]}
                 for a in g]
                for g in groups
            ],
        }
        (n_dir / "assignment.json").write_text(json.dumps(summary, indent=2))

        task_to_base = {}
        for g in groups:
            for a in g:
                task_to_base[a["task"]] = a["base_task"]
        (n_dir / "task_meta.json").write_text(json.dumps(task_to_base, indent=2))

        for cond in uc.conditions:
            if cond not in BUILDERS:
                raise KeyError(f"unknown condition {cond!r}")
            out = n_dir / f"{cond}.json"
            try:
                plan = BUILDERS[cond](groups, n)
            except (ValueError, KeyError) as e:
                if out.exists():
                    out.unlink()
                print(f"[gen] N={n} cond={cond} SKIPPED: {e}")
                continue
            out.write_text(json.dumps(plan, indent=2))
            print(f"[gen] N={n} cond={cond} -> {out}")


if __name__ == "__main__":
    generate_all()
