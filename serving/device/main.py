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
    parser.add_argument("--max-batch-wait-ms", type=float, default=1000.0, help="Maximum batch formation wait.")
    parser.add_argument("--queue-capacity", type=int, default=1024, help="Maximum total queued inference requests.")
    args = parser.parse_args()
    asyncio.run(
        serve(
            RuntimeServerConfig(
                host=args.host,
                port=args.port,
                max_batch_size=args.max_batch_size,
                max_batch_wait_ms=args.max_batch_wait_ms,
                queue_capacity=args.queue_capacity,
            ),
            bootstrap_json=_resolve_bootstrap_json(args),
        )
    )


if __name__ == "__main__":
    main()
