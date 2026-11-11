# RTVSntask/vlm — response time vs. number of tasks (VLM, vLLM multi-LoRA)

VLM analog of [`RTVSntask/llm`](../llm). Each "task" is a LoRA adapter over one
shared **Qwen2-VL-2B** backbone, and every request carries an MNIST OCR image
alongside its prompt. Conditions, trace machinery, result schema, and failure
classification are the same as the LLM version — see
[`../llm/README.md`](../llm/README.md) for the arms table. Only the differences
are documented here.

## Quick start

```bash
# 1. Generate the adapter pool (once). Writes to <FMTK>/models/vlm/finetuned/.
python experiments/RTVSntask/vlm/make_loras.py --names qwenvlA,qwenvlB

# 2. Smoke test
NUM_TASKS_SWEEP=2 RPS_SWEEP=1 PHASE_DURATION=90 INCLUDE_TPC=0 \
    bash experiments/RTVSntask/vlm/run.sh

# 3. Full sweep
NUM_TASKS_SWEEP=1,2,4,6,8,10 RPS_SWEEP=2 \
    bash experiments/RTVSntask/vlm/run.sh

# 4. Plot — the LLM plotter, reused: run.py writes the identical CSV schema.
python experiments/RTVSntask/llm/plot.py --results experiments/RTVSntask/vlm/results
```

## The `mm_processor_kwargs` cap is required, not tuning

With `enable_lora=True` and Qwen2-VL's default pixel limits, the vLLM engine
**fails to build**, during `profile_run`, before serving a single request:

```
RuntimeError: The size of tensor a (5120) must match the size of tensor b (16384)
```

This is not an adapter problem, despite how it reads. vLLM constructs a dummy
profiling image at the default `max_pixels`, reserving 16384 vision tokens
against a ~5120-token batch budget. Capping `min_pixels`/`max_pixels` in
[`qwen2_vl_vllm.py`](../../../../../FMTK/src/fmtk/components/backbones/qwen2_vl_vllm.py)
shrinks that dummy image and the build succeeds.

Verified by isolation: no-LoRA builds (with warnings), LoRA + default pixels
fails, LoRA + capped pixels builds clean.

Because the error text is identical to a genuine adapter shape mismatch,
`run.sh`'s `classify_failure` tests for it **first** and reports it as
`mm_profile_token_mismatch`. Do not reorder those branches.

## Adapters are language-side only

vLLM 0.6.6 lists exactly `{qkv_proj, o_proj, gate_up_proj, down_proj}` in
`Qwen2VLForConditionalGeneration.supported_lora_modules`; LoRA over the vision
encoder is an explicit upstream TODO. `make_loras.py` targets `q_proj`/`v_proj`,
which in Qwen2-VL occur only under `model.layers.*.self_attn` — the visual tower
uses `visual.blocks.*.attn.qkv` — so the adapters satisfy this by construction.
The script asserts it and refuses to write an adapter that touches the vision
tower. **Report this in any writeup**: the adapters do not specialise vision.

Weights are random-init with zero-init `lora_B`, so they are numerically
identity — the same choice `fair_share/llm/make_loras.py` makes. They exercise
adapter memory, load time, and multi-LoRA scheduling, not accuracy.

## Request size

An OCR digit is 28x28, so `smart_resize` floors it at 56x56 = ~4 merged vision
tokens. A full request is **64 prompt tokens** (measured), and the image is
~9 KB on the wire as float32 CHW.

Consequences:
- `MAX_MODEL_LEN=512` is ample; there is no need to raise it.
- No gRPC message-size limit needs lifting.
- **There is almost no multimodal prefill cost to observe.** These curves will
  closely track the text-LLM ones. That supports a "VLM backbone sharing behaves
  like LLM sharing" claim, but it does not demonstrate image-prefill effects. A
  dataset of full-size photographs would change both this and the two bullets
  above.

## Memory

Qwen2-VL-2B is ~4.4 GB of fp16 weights. As with the LLM sweep, `no_sharing` is
expected to fail once N copies stop fitting; that failure is recorded to
`failure.json` with a classified reason and the sweep continues.
