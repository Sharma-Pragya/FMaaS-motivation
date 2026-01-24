import json
import numpy as np
import torch
import torch.nn as nn
from pytriton.triton import Triton, TritonConfig
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
        """
        Standard inference logic. 
        PyTriton handles the queueing and batching automatically.
        """
        print("inference request received")
        swap_st=time.time_ns()
        if self.pipeline is None: 
            self.pipeline, self.decoders, self.current_task = get_loaded_pipeline()
        task = task[0][0].decode('utf-8')
        if self.current_task != task:
            self.current_task=task
            decoder_name = self.decoders.get(self.current_task)
            self.pipeline.active_decoder = self.pipeline.decoders[decoder_name]
        swap_time=time.time_ns()-swap_st
        
        st=time.time_ns()
        # Prepare Inputs
        bx = torch.from_numpy(x)
        b_mask = torch.from_numpy(mask) if mask is not None else None
        if self.pipeline.active_decoder is not None:
            logits=self.pipeline.forward(bx, b_mask)
            if isinstance(self.pipeline.active_decoder.criterion, (nn.CrossEntropyLoss)):
                logits = torch.argmax(logits, dim=1)
            if (hasattr(self.pipeline.active_decoder, "requires_model") and self.pipeline.active_decoder.requires_model and hasattr(self.model_instance.model, "normalizer")):
                logits = self.pipeline.model_instance.model.normalizer(x=logits, mode="denorm")
            result=logits.detach().cpu().numpy()
        else:
            embeddings = self.pipeline.model_instance.forward((bx, None))
            result = self.pipeline.model_instance.postprocess(embeddings)
        et=time.time_ns()
        return {"output": result,"start_time":np.array([swap_st]),"proc_time": np.array([et - st]),"swap_time": np.array([swap_time])}

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

    with Triton(config=config) as triton:
        # Bind the Inference Model
        triton.bind(
            model_name="edge_infer",
            infer_func=system.infer,
            inputs=[
                Tensor(name="x", dtype=np.float32, shape=(-1,-1)),
                Tensor(name="task", dtype=bytes, shape=(1,)),
                Tensor(name="mask", dtype=np.float32, shape=(-1,), optional=True),
                Tensor(name="question", dtype=bytes, shape=(1,), optional=True),
            ],
            outputs=[
                Tensor(name="output", dtype=np.float32, shape=(-1, -1)),
                Tensor(name="start_time", dtype=np.int64, shape=(1,)),
                Tensor(name="proc_time", dtype=np.int64, shape=(1,)),
                Tensor(name="swap_time", dtype=np.int64, shape=(1,)),
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