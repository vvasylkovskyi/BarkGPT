# Running LLM locally on CPU with vLLM

Among many options and commands, the one that finally worked for my was:

```sh
uv pip install vllm --torch-backend=auto
```

## Dependencies

```toml
"mistral-common>=1.9.0",
"transformers>=5.1.0",
"vllm==0.11.0",
"torch==2.8.0"
```

## Edit

This didn't work well running on CPU/MAC, because even `f16` quantization is not enough. So ended-up using `llama-cpp` with `q8_0`.
