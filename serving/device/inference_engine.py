# device/inference_engine.py
import asyncio, base64, numpy as np, torch, time
from device.model_loader import get_loaded_pipeline
from device.config import DEVICE
import torch.nn as nn

request_queue: asyncio.Queue = asyncio.Queue()

def decode_raw(obj: dict) -> torch.Tensor:
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(obj["dtype"]))
    arr = arr.reshape(obj["shape"])
    return torch.from_numpy(arr)

async def _gpu_worker():
    """Background worker that executes inference on queued requests."""
    while True:
        fut, req, arrival_time, server_start = await request_queue.get()
        pipeline, decoders = get_loaded_pipeline()

        bx = decode_raw(req.x.model_dump())
        mask = decode_raw(req.mask.model_dump()) if req.mask else None

        # Forward pass
        feats = pipeline.model_instance.forward(bx, mask)
        start_infer = time.time()
        decoder = decoders[req.task]
        pipeline.load_decoder(decoder, swap=False)
        logits = pipeline.active_decoder.forward(feats)
        if isinstance(pipeline.active_decoder.criterion, (nn.CrossEntropyLoss)):
            logits = torch.argmax(logits, dim=1)
        # Problem RevIN normalizer computed in the model instance forward is batch size N but then we are
        # denormalizing one at a time here which causes a size mismatch 

        # if (hasattr(pipeline.active_decoder, "requires_model") and pipeline.active_decoder.requires_model and hasattr(pipeline.model_instance.model, "normalizer")):
        #     logits = pipeline.model_instance.model.normalizer(x=logits, mode="denorm")
        #     print(logits.shape, y_out.shape)
        
        end_infer = time.time()

        resp = {
            "req_id": req.req_id,
            "device_wait_time": start_infer - arrival_time,
            "device_infer_time": end_infer - start_infer,
        }

        fut.set_result(resp)
        request_queue.task_done()
