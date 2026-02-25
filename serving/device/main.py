import argparse
import os

# Parse --cuda before any device.* imports so that CUDA_DEVICE is set
# before config.py evaluates DEVICE at module load time.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--cuda", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.cuda:
    os.environ["CUDA_DEVICE"] = _pre_args.cuda

import json
import numpy as np
import torch
import torch.nn as nn
from pytriton.triton import Triton, TritonConfig, TritonSecurityConfig
from pytriton.model_config import ModelConfig, Tensor
from pytriton.decorators import batch
from device.model_loader import load_models, add_decoder, swap_backbone, get_loaded_pipeline
import time

class UnifiedEdgeSystem:
    def __init__(self):
        self.pipeline = None
        self.decoders = None
        self.current_task = None

    @batch
    def infer(self, x, task, mask=None, question=None):
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
        # Capture local snapshots of pipeline and decoders at call start.
        # This prevents a concurrent swap_backbone (running in the control thread)
        # from changing self.pipeline mid-call, which would cause a shape mismatch
        # between features computed from the old backbone and decoders from the new one.
        if self.pipeline is None:
            self.pipeline, self.decoders, self.current_task = get_loaded_pipeline()
        pipeline = self.pipeline
        decoders = self.decoders

        # Extract all tasks in the batch
        batch_size = x.shape[0]
        tasks = [task[i][0].decode('utf-8') for i in range(batch_size)]

        # 1. Batch feature extraction (backbone forward pass on ALL inputs)
        bx = torch.from_numpy(x)
        b_mask = torch.from_numpy(mask) if mask is not None else None
        feats = pipeline.model_instance.forward(bx, b_mask)
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
                embeddings = pipeline.model_instance.forward((feat_i, None))
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

    #batch not needed
    # --- 2. The Control Endpoint ---
    @batch
    def control(self, command, payload):
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
                logger = load_models(config['backbone'], config['decoders'])
                status_msg = f"loaded_{config['backbone']}"

            elif cmd_str == "swap_backbone":
                # In-process backbone swap: free old GPU memory, load new backbone + decoders
                config_json = payload[0][0].decode("utf-8")
                config = json.loads(config_json)

                print(f"[System] Swapping backbone to: {config['backbone']}, decoders: {config['decoders']}")
                logger = swap_backbone(config['backbone'], config['decoders'])
                # Refresh cached references so infer() sees the new backbone + decoders.
                # Force current_task=None so the next infer() always re-selects the
                # active_decoder from the new pipeline (avoids stale decoder reference).
                self.pipeline, self.decoders, _ = get_loaded_pipeline()
                self.current_task = None
                status_msg = f"swapped_{config['backbone']}"

            elif cmd_str == "add_decoder":
                # Hot-add decoders to the already-loaded backbone
                config_json = payload[0][0].decode("utf-8")
                config = json.loads(config_json)

                print(f"[System] Hot-adding {len(config['decoders'])} decoder(s)")
                logger = add_decoder(config['decoders'])
                # Refresh cached references so infer() sees the new decoders
                self.pipeline, self.decoders, self.current_task = get_loaded_pipeline()
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
    args = parser.parse_args()

    system = UnifiedEdgeSystem()
    grpc_port = args.port
    http_port = grpc_port + 1
    metrics_port=http_port + 1
    
    config = TritonConfig(
        grpc_port=grpc_port,
        http_port=http_port,
        metrics_port=metrics_port
    )

    with Triton(config=config, security_config=TritonSecurityConfig(restricted_endpoints=[])) as triton:
        # Bind the Inference Model
        triton.bind(
            model_name="edge_infer",
            infer_func=system.infer,
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
            infer_func=system.control,
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
        
        print("[System] PyTriton Server running on port")
        triton.serve()