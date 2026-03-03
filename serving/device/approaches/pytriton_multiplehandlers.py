import argparse
import os
import re

# Parse --cuda before any device.* imports so that CUDA_DEVICE is set
# before config.py evaluates DEVICE at module load time.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--cuda", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.cuda:
    os.environ["CUDA_DEVICE"] = _pre_args.cuda

import json
import threading
import numpy as np
import torch
import torch.nn as nn
from pytriton.triton import Triton, TritonConfig, TritonSecurityConfig
from pytriton.model_config import ModelConfig, Tensor
from pytriton.decorators import batch
from device.model_loader import load_models, add_decoder, swap_backbone, get_loaded_pipeline
import time
from urllib.request import urlopen

class SharedPipelineState:
    def __init__(self):
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._model_forward_lock = threading.Lock()
        self.pipeline = None
        self.decoders = None
        self._inflight = 0
        self._updates_waiting = 0

    def _read_metric(self, metrics_text: str, metric_name: str, model_name: str):
        pattern = re.compile(
            rf'^{re.escape(metric_name)}\{{[^}}]*model="{re.escape(model_name)}"[^}}]*\}}\s+([0-9.eE+-]+)$',
            re.MULTILINE,
        )
        match = pattern.search(metrics_text)
        return float(match.group(1)) if match else None

    def read_queue_metrics(self, metrics_port: int, model_name: str):
        try:
            with urlopen(f"http://127.0.0.1:{metrics_port}/metrics", timeout=2) as response:
                metrics_text = response.read().decode("utf-8", errors="replace")
        except OSError:
            return None

        return {
            "queue_duration_us": self._read_metric(metrics_text, "nv_inference_queue_duration_us", model_name),
            "request_success": self._read_metric(metrics_text, "nv_inference_request_success", model_name),
            "pending_requests": self._read_metric(metrics_text, "nv_inference_pending_request_count", model_name) or 0.0,
        }

    def _refresh_loaded_locked(self):
        self.pipeline, self.decoders, _ = get_loaded_pipeline()
        if self.pipeline is None or self.decoders is None:
            raise RuntimeError("pipeline_not_loaded")

    def acquire_snapshot(self):
        with self._lock:
            while self._updates_waiting > 0:
                self._idle.wait()
            if self.pipeline is None or self.decoders is None:
                self._refresh_loaded_locked()
            self._inflight += 1
            return self.pipeline, self.decoders

    def release_snapshot(self):
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            if self._inflight == 0:
                self._idle.notify_all()

    def _wait_until_idle_locked(self):
        while self._inflight > 0:
            self._idle.wait()

    def run_model_forward(self, pipeline, bx, b_mask):
        with self._model_forward_lock:
            return pipeline.model_instance.forward(bx, b_mask)

    def run_model_postprocess_forward(self, pipeline, feat_i):
        with self._model_forward_lock:
            return pipeline.model_instance.forward((feat_i, None))

    def load(self, backbone, decoders):
        with self._lock:
            self._updates_waiting += 1
            try:
                self._wait_until_idle_locked()
                logger = load_models(backbone, decoders)
                self._refresh_loaded_locked()
                return logger
            finally:
                self._updates_waiting = max(0, self._updates_waiting - 1)
                self._idle.notify_all()

    def swap_backbone(self, backbone, decoders):
        with self._lock:
            self._updates_waiting += 1
            try:
                self._wait_until_idle_locked()
                logger = swap_backbone(backbone, decoders)
                self._refresh_loaded_locked()
                return logger
            finally:
                self._updates_waiting = max(0, self._updates_waiting - 1)
                self._idle.notify_all()

    def add_decoders(self, decoders):
        with self._lock:
            self._updates_waiting += 1
            try:
                self._wait_until_idle_locked()
                logger = add_decoder(decoders)
                self._refresh_loaded_locked()
                return logger
            finally:
                self._updates_waiting = max(0, self._updates_waiting - 1)
                self._idle.notify_all()


class InferenceEndpoint:
    def __init__(self, shared_state: SharedPipelineState, handler_id: int):
        self._shared_state = shared_state
        self._handler_id = handler_id

    @batch
    def __call__(self, x, task, mask=None, question=None):
        # """
        # Batch inference supporting multiple tasks.
        # - Batch feature extraction across all tasks (backbone forward)
        # - Process decoders sequentially per task
        # """
        # print("=" * 80)
        # print("BATCH INFERENCE RECEIVED")
        # print("=" * 80)
        
        # # Debug: Print batch composition
        # batch_size = x.shape[0]
        # tasks = [task[i][0].decode('utf-8') for i in range(batch_size)]
        
        # print(f"[BATCH DEBUG] Batch size: {batch_size}")
        # print(f"[BATCH DEBUG] Input shape x: {x.shape}")
        # print(f"[BATCH DEBUG] Task shape: {task.shape}")
        # print(f"[BATCH DEBUG] Tasks in batch: {tasks}")
        # print(f"[BATCH DEBUG] Unique tasks: {set(tasks)}")
        # if mask is not None:
        #     print(f"[BATCH DEBUG] Mask shape: {mask.shape}")
        # print("=" * 80)
        
        st = time.time_ns()
        pipeline, decoders = self._shared_state.acquire_snapshot()
        try:
            # Extract all tasks in the batch
            batch_size = x.shape[0]
            tasks = [task[i][0].decode('utf-8') for i in range(batch_size)]

            # 1. Batch feature extraction (backbone forward pass on ALL inputs)
            bx = torch.from_numpy(x)
            b_mask = torch.from_numpy(mask) if mask is not None else None
            feats = self._shared_state.run_model_forward(pipeline, bx, b_mask)
            proc_time = time.time_ns() - st
            # 2. Process decoders sequentially per task
            logits_list = []
            swap_time_list = []
            decoder_time_list = []
            active_decoder = None
            current_task = None
            for i in range(batch_size):
                task_name = tasks[i]
                swap_st = time.time_ns()
                if current_task != task_name:
                    current_task = task_name
                    decoder_name = decoders.get(current_task)
                    active_decoder = pipeline.decoders[decoder_name]
                swap_time = time.time_ns() - swap_st
                swap_time_list.append(swap_time)

                decoder_st = time.time_ns()
                # Process this sample with its decoder
                feat_i = feats[i:i+1]  # Keep batch dimension for forward pass
                if active_decoder is not None:
                    logit_i = active_decoder.forward(feat_i)
                    if isinstance(active_decoder.criterion, (nn.CrossEntropyLoss)):
                        logit_i = torch.argmax(logit_i, dim=1)
                    if (hasattr(active_decoder, "requires_model") and
                        active_decoder.requires_model and
                        hasattr(pipeline.model_instance.model, "normalizer")):
                        logit_i = pipeline.model_instance.model.normalizer(x=logit_i, mode="denorm")
                    result_i = logit_i.detach().cpu().numpy()
                else:
                    embeddings = self._shared_state.run_model_postprocess_forward(pipeline, feat_i)
                    result_i = pipeline.model_instance.postprocess(embeddings)
                
                logits_list.append(result_i)
                decoder_time = time.time_ns() - decoder_st
                decoder_time_list.append(decoder_time)
            # 3. Combine results
            result = np.vstack(logits_list) if logits_list else np.array([])
            et = time.time_ns()
            
            # Replicate timing metrics across batch dimension to match PyTriton's batch size expectation
            return {
                "output": result,
                "start_time": np.full((batch_size,), st, dtype=np.int64),
                "end_time": np.full((batch_size,), et, dtype=np.int64),
                "proc_time": np.full((batch_size,), proc_time, dtype=np.int64),
                "swap_time": np.array(swap_time_list, dtype=np.int64),
                "decoder_time": np.array(decoder_time_list, dtype=np.int64)
            }
        finally:
            self._shared_state.release_snapshot()


class ControlEndpoint:
    def __init__(self, shared_state: SharedPipelineState):
        self._shared_state = shared_state

    @batch
    def __call__(self, command, payload):
        """
        Executes management tasks (Load/Unload).
        The client sends a 'command' (load/unload) and a JSON 'payload'.
        """
        # Take the first command in the batch (Control requests usually aren't batched)
        cmd_str = command[0][0].decode("utf-8")
        status_msg = "ok"
        logger = None

        try:
            if cmd_str == "load":
                # Decode the JSON payload containing backbone/decoders info
                config_json = payload[0][0].decode("utf-8")
                config = json.loads(config_json)

                print(f"[System] Loading backbone: {config['backbone']}")
                logger = self._shared_state.load(config['backbone'], config['decoders'])
                status_msg = f"loaded_{config['backbone']}"

            elif cmd_str == "swap_backbone":
                # In-process backbone swap: free old GPU memory, load new backbone + decoders
                config_json = payload[0][0].decode("utf-8")
                config = json.loads(config_json)

                print(f"[System] Swapping backbone to: {config['backbone']}, decoders: {config['decoders']}")
                logger = self._shared_state.swap_backbone(config['backbone'], config['decoders'])
                status_msg = f"swapped_{config['backbone']}"

            elif cmd_str == "add_decoder":
                # Hot-add decoders to the already-loaded backbone
                config_json = payload[0][0].decode("utf-8")
                config = json.loads(config_json)

                print(f"[System] Hot-adding {len(config['decoders'])} decoder(s)")
                logger = self._shared_state.add_decoders(config['decoders'])
                status_msg = f"added_{len(config['decoders'])}_decoders"

            else:
                status_msg = f"unknown_command_{cmd_str}"

        except Exception as e:
            status_msg = f"error_{str(e)}"
            print(f"[System] Error: {e}")

        logger_summary = str(logger.summary()) if logger else "no_logger"
        return {
            "status": np.array([status_msg.encode('utf-8')]),
            "logger_summary": np.array([logger_summary.encode('utf-8')])
        }
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="gRPC port")
    parser.add_argument("--cuda", type=str, default=None, help="CUDA device (e.g. cuda:0, cuda:1). Defaults to first available GPU.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to write queue_metrics.csv (default: current directory)")
    parser.add_argument("--infer-handlers", type=int, default=1, help="Number of PyTriton inference handlers sharing the same loaded pipeline.")
    args = parser.parse_args()

    if args.infer_handlers < 1:
        raise ValueError("--infer-handlers must be >= 1")

    shared_state = SharedPipelineState()
    infer_handlers = [InferenceEndpoint(shared_state, i) for i in range(args.infer_handlers)]
    control_handler = ControlEndpoint(shared_state)
    grpc_port = args.port
    http_port = grpc_port + 1
    metrics_port=http_port + 1
    
    config = TritonConfig(
        grpc_port=grpc_port,
        http_port=http_port,
        metrics_port=metrics_port,
    )

    with Triton(config=config, security_config=TritonSecurityConfig(restricted_endpoints=[])) as triton:
        # Bind the Inference Model
        triton.bind(
            model_name="edge_infer",
            infer_func=infer_handlers,
            inputs=[
                Tensor(name="x", dtype=np.float32, shape=(-1,-1)),
                Tensor(name="task", dtype=bytes, shape=(-1,)),
                Tensor(name="mask", dtype=np.float32, shape=(-1,), optional=True),
                Tensor(name="question", dtype=bytes, shape=(-1,), optional=True),
            ],
            outputs=[
                Tensor(name="output", dtype=np.float32, shape=(-1, -1)),
                Tensor(name="start_time", dtype=np.int64, shape=(-1,)),
                Tensor(name="end_time", dtype=np.int64, shape=(-1,)),
                Tensor(name="proc_time", dtype=np.int64, shape=(-1,)),
                Tensor(name="swap_time", dtype=np.int64, shape=(-1,)),  
                Tensor(name="decoder_time", dtype=np.int64, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=1)
        )

        # Bind the Control Model
        triton.bind(
            model_name="edge_control",
            infer_func=control_handler,
            inputs=[
                Tensor(name="command", dtype=bytes, shape=(1,)),
                Tensor(name="payload", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="status", dtype=bytes, shape=(1,)),
                Tensor(name="logger_summary", dtype=bytes, shape=(1,)),
            ],
            config=ModelConfig(max_batch_size=1)
        )
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            queue_csv_path = os.path.join(args.output_dir, f"queue_metrics_port{grpc_port}.csv")
        else:
            queue_csv_path = f"queue_metrics_port{grpc_port}.csv"

        queue_rows = []
        queue_stop = threading.Event()

        def flush_queue_metrics():
            with open(queue_csv_path, "w") as f:
                f.write("wall_time_s,queue_size,num_completed,mean_queue_delay_us,delta_queue_duration_us\n")
                for row in queue_rows:
                    f.write(
                        f"{row['wall_time_s']},{row['queue_size']},{row['num_completed']},"
                        f"{row['mean_queue_delay_us']:.1f},{row['delta_queue_duration_us']:.1f}\n"
                    )

        def poll_queue_delays():
            start_wall = time.time()
            prev_queue_duration_us = None
            prev_request_success = None
            while not queue_stop.wait(5):
                metrics = shared_state.read_queue_metrics(metrics_port=metrics_port, model_name="edge_infer")
                if not metrics:
                    continue

                queue_duration_us = metrics["queue_duration_us"]
                request_success = metrics["request_success"]
                queue_size = int(metrics["pending_requests"])
                if queue_duration_us is None or request_success is None:
                    continue

                wall_s = round(time.time() - start_wall, 1)

                if prev_queue_duration_us is None or prev_request_success is None:
                    prev_queue_duration_us = queue_duration_us
                    prev_request_success = request_success
                    continue

                delta_queue_duration_us = queue_duration_us - prev_queue_duration_us
                delta_request_success = request_success - prev_request_success
                prev_queue_duration_us = queue_duration_us
                prev_request_success = request_success

                # Triton metrics are cumulative counters; resets can happen across restarts.
                if delta_queue_duration_us < 0 or delta_request_success < 0:
                    continue

                mean_us = delta_queue_duration_us / delta_request_success if delta_request_success else 0.0
                queue_rows.append({
                    "wall_time_s": wall_s,
                    "queue_size": queue_size,
                    "num_completed": int(delta_request_success),
                    "mean_queue_delay_us": mean_us,
                    "delta_queue_duration_us": delta_queue_duration_us,
                })
                if delta_request_success:
                    print(f"[Queue] t={wall_s}s current_queue_size={queue_size} | "
                          f"new_completed={int(delta_request_success)} | "
                          f"mean_queue_delay={mean_us:.1f}us")
                else:
                    print(f"[Queue] t={wall_s}s current_queue_size={queue_size} | no new completions")

        queue_thread = threading.Thread(target=poll_queue_delays, daemon=True)
        queue_thread.start()
        print(f"[System] PyTriton Server running on port with {args.infer_handlers} inference handler(s)")
        try:
            triton.serve()
        finally:
            queue_stop.set()
            queue_thread.join(timeout=6)
            flush_queue_metrics()
