# Running LLM locally on CPU with vLLM

Among many options and commands, the one that finally worked for my was:

```sh
uv pip install vllm --torch-backend=auto
```

## Edit

This didn't work well running on CPU/MAC, because even `f16` quantization is not enough. So ended-up using `llama-cpp` with `q8_0`.
