import argparse
import asyncio
import base64
import os


_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--cuda", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.cuda:
    os.environ["CUDA_DEVICE"] = _pre_args.cuda

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
            ),
            bootstrap_json=_resolve_bootstrap_json(args),
        )
    )


if __name__ == "__main__":
    main()
