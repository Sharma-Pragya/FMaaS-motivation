import json
import numpy as np
import torch
import torch.nn as nn
from pytriton.triton import Triton, TritonConfig, TritonSecurityConfig
from pytriton.model_config import ModelConfig, Tensor
from pytriton.decorators import batch
from device.model_loader import load_models, get_loaded_pipeline
import time
import argparse

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
        if self.pipeline is None: 
            self.pipeline, self.decoders, self.current_task = get_loaded_pipeline()
        
        # Extract all tasks in the batch
        batch_size = x.shape[0]
        tasks = [task[i][0].decode('utf-8') for i in range(batch_size)]
        
        # 1. Batch feature extraction (backbone forward pass on ALL inputs)
        bx = torch.from_numpy(x)
        b_mask = torch.from_numpy(mask) if mask is not None else None
        feats = self.pipeline.model_instance.forward(bx, b_mask)
        proc_time = time.time_ns() - st
        # 2. Process decoders sequentially per task
        logits_list = []
        swap_time_list = []
        decoder_time_list = []
        for i in range(batch_size):
            task_name = tasks[i]
            swap_st = time.time_ns()
            if self.current_task != task_name:
                self.current_task = task_name
                decoder_name = self.decoders.get(self.current_task)
                self.pipeline.active_decoder = self.pipeline.decoders[decoder_name]
            swap_time = time.time_ns() - swap_st
            swap_time_list.append(swap_time)

            decoder_st = time.time_ns()
            # Process this sample with its decoder
            feat_i = feats[i:i+1]  # Keep batch dimension for forward pass
            if self.pipeline.active_decoder is not None:
                logit_i = self.pipeline.active_decoder.forward(feat_i)
                if isinstance(self.pipeline.active_decoder.criterion, (nn.CrossEntropyLoss)):
                    logit_i = torch.argmax(logit_i, dim=1)
                if (hasattr(self.pipeline.active_decoder, "requires_model") and 
                    self.pipeline.active_decoder.requires_model and 
                    hasattr(self.pipeline.model_instance.model, "normalizer")):
                    logit_i = self.pipeline.model_instance.model.normalizer(x=logit_i, mode="denorm")
                result_i = logit_i.detach().cpu().numpy()
            else:
                embeddings = self.pipeline.model_instance.forward((feat_i, None))
                result_i = self.pipeline.model_instance.postprocess(embeddings)
            
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

        try:
            if cmd_str == "load":
                # Decode the JSON payload containing backbone/decoders info
                config_json = payload[0][0].decode("utf-8")
                config = json.loads(config_json)
                
                print(f"[System] Loading backbone: {config['backbone']}")
                logger=load_models(config['backbone'], config['decoders'])
                status_msg = f"loaded_{config['backbone']}"
                            
            else:
                status_msg = f"unknown_command_{cmd_str}"

        except Exception as e:
            status_msg = f"error_{str(e)}"
            print(f"[System] Error: {e}")

        return {
            "status": np.array([status_msg.encode('utf-8')]),
            "logger_summary": np.array([str(logger.summary()).encode('utf-8')])
        }
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="gRPC port")
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
            config=ModelConfig(max_batch_size=32)
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