import argparse
import asyncio
import base64
import ctypes
import os

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("[Device] Using uvloop event loop")
except ImportError:
    print("[Device] uvloop not available, using default asyncio event loop")


_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--cuda", type=str, default=None)
_pre_parser.add_argument("--mps-thread-pct", type=int, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.cuda:
    os.environ["CUDA_DEVICE"] = _pre_args.cuda
if _pre_args.mps_thread_pct is not None:
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(_pre_args.mps_thread_pct)
    print(f"[Device] MPS thread percentage set to {_pre_args.mps_thread_pct}%")
    import torch
    dev_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev_id)

    print(f"GPU: {props.name}")
    print(f"SM count from torch device properties: {props.multi_processor_count}")

from device.server import RuntimeServerConfig, serve


def _resolve_bootstrap_json(args) -> str | None:
    if args.bootstrap_json:
        return args.bootstrap_json
    if args.bootstrap_json_b64:
        return base64.b64decode(args.bootstrap_json_b64.encode("utf-8")).decode("utf-8")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="Reserved service port for the future gRPC server.")
    parser.add_argument("--cuda", type=str, default=None, help="CUDA device override (e.g. cuda:0).")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address for the gRPC runtime server.")
    parser.add_argument("--output-dir", type=str, default=None, help="Reserved for future metrics output.")
    parser.add_argument("--bootstrap-json", type=str, default=None, help="Optional deployment payload used for initial model load.")
    parser.add_argument("--bootstrap-json-b64", type=str, default=None, help="Base64-encoded deployment payload used for initial model load.")
    parser.add_argument("--max-batch-size", type=int, default=5, help="Maximum cross-task batch size.")
    parser.add_argument("--max-batch-wait-ms", type=float, default=0, help="Maximum batch formation wait.")
    parser.add_argument("--queue-capacity", type=int, default=102400, help="Maximum total queued inference requests.")
    parser.add_argument("--runtime-type", choices=["pytorch", "vllm"], default="pytorch", help="Inference runtime: pytorch (TSFM) or vllm (LLM).")
    parser.add_argument("--scheduler-policy", choices=["fifo", "round_robin", "wfq", "token_bucket", "saba", "deadline_split","stfq"], default="stfq", help="Batch scheduling policy: fifo, round_robin, wfq, token_bucket, saba, or deadline_split (deadline-driven batch splitting).")
    parser.add_argument("--task-rates", type=str, default=None, help="Comma-separated task:rps pairs e.g. ecgclass:10,gestureclass:100 — used by WFQ/TokenBucket policies.")
    parser.add_argument("--isolation-mode", choices=["shared", "process", "none"], default="shared", help="Isolation mode: shared (default, all tasks in one process) or process (one process per task).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Fraction of GPU memory vLLM may use for KV cache (0.0–1.0). Lower values allow multiple engines on one GPU.")
    parser.add_argument("--max-model-len", type=int, default=None, help="vLLM max sequence length. Set to a small value (e.g. 256) to reduce KV cache size and allow multiple engines on one GPU.")
    parser.add_argument("--tpc-mode", choices=["none", "libsmctrl", "green"], default="none", help="TPC isolation: none, libsmctrl (driver<=528), or green (CUDA 12.4+).")
    parser.add_argument("--tpc-partition", type=int, nargs="+", default=None, help="TPC IDs to pin this server to (e.g. --tpc-partition 0 1 2 3).")
    parser.add_argument("--mps-thread-pct", type=int, default=None, help="CUDA MPS active thread percentage (0-100). Requires MPS daemon running. Set via pre-parser so it takes effect before CUDA init.")
    parser.add_argument("--worker-mode", choices=["threaded", "inline"], default="threaded", help="Per-task pipeline worker mode. threaded: each task gets its own thread + CUDA stream. inline: synchronous on the backbone worker thread, no per-task thread/stream.")
    parser.add_argument("--verbose-batch-logs", action="store_true", help="Enable high-frequency per-batch request logs. Disabled by default for benchmark stability.")
    args = parser.parse_args()
    # Parse task rates: "ecgclass:10,gestureclass:100" -> {"ecgclass": 10.0, ...}
    task_rates: dict[str, float] = {}
    if args.task_rates:
        for pair in args.task_rates.split(","):
            task, rate = pair.strip().split(":")
            task_rates[task.strip()] = float(rate.strip())

    asyncio.run(
        serve(
            RuntimeServerConfig(
                host=args.host,
                port=args.port,
                max_batch_size=args.max_batch_size,
                max_batch_wait_ms=args.max_batch_wait_ms,
                queue_capacity=args.queue_capacity,
                runtime_type=args.runtime_type,
                scheduler_policy=args.scheduler_policy,
                isolation_mode=args.isolation_mode,
                task_rates=task_rates,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                tpc_mode=args.tpc_mode,
                tpc_partition=args.tpc_partition,
                worker_mode=args.worker_mode,
                verbose_batch_logs=args.verbose_batch_logs,
            ),
            bootstrap_json=_resolve_bootstrap_json(args),
        )
    )


if __name__ == "__main__":
    main()
