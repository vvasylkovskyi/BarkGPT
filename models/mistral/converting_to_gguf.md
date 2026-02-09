# Converting Mistral to GGUF format

To convert the model to GGUF format the steps are:

1. Download the hugging faces snapshot locally to prepare hugging faces format
2. Convert into GGuf format using llama
3. Serve `gguf` using `llama-cpp`

## Downloading the hugging faces snapshot

Using `huggingface_hub` we can install our

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "mistralai/Ministral-3-3B-Base-2512", cache_dir="./outputs/mistral-3b-base"
)
```

Now the model is in `./outputs/mistral-3b-base`

## Converting Hugging Faces into GGuf

So how to convert our model into GGuf, optimized for inference? [According to this great tutorial](https://github.com/ggml-org/llama.cpp/discussions/2948), the steps are as follow:

1. We already have the model in the right packaging
2. Use the https://github.com/ggml-org/llama.cpp repository
3. Install dependencies in `llama.cpp`: `pip install -r llama.cpp/requirements.txt`
4. Verify installation and functionality of the script: `python llama.cpp/convert_hf_to_gguf.py -h`
5. Finally, convert the model by running the script:

```sh
# python llama.cpp/convert_hf_to_gguf.py \
#   outputs/mistral-3b-base/models--mistralai--Ministral-3-3B-Base-2512/snapshots/6f9c4b12a95b139af68670a6713616b757923735 \
#   --outfile ./outputs/gguf/mistral-3b-base-f32-v1.gguf \
#   --outtype f32

# python llama.cpp/convert_hf_to_gguf.py \
#     ./outputs/mistral-7b-v0.3/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1 \
#     --outfile ./outputs/gguf/mistral-7b-base-f32-v1.gguf \
#     --outtype f32

python llama.cpp/convert_hf_to_gguf.py \
    ./outputs/mistral-7b-v0.3/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1 \
    --outfile ./outputs/gguf/mistral-7b-base-f16-v1.gguf \
    --outtype f16
```

You might notice that we have to indicate the exact snapshot folder. You should look-up for the exact folder name.

Running command above should produce output like follows:

```sh
INFO:gguf.gguf_writer:Writing the following files:
INFO:gguf.gguf_writer:outputs/gguf/mistral-3b-base-f32-v1.gguf: n_tensors = 236, total_size = 13.7G
Writing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 13.7G/13.7G [00:26<00:00, 524Mbyte/s]
INFO:hf-to-gguf:Model successfully exported to outputs/gguf/mistral-3b-base-f32-v1.gguf
```

And I can see the `gguf` file present.

### Quantization

Note, on CPU it might be a good ideia to quantize the model. On my MacOS, with powerful CPU I was able to run the model only when quantized to 8bits. So use GGUF like that:

```sh
python llama.cpp/convert_hf_to_gguf.py \
    ./outputs/mistral-7b-v0.3/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1 \
    --outfile ./outputs/gguf/mistral-7b-base-q8_0-v1.gguf \
    --outtype q8_0
```

## Serve Gguf with Llama-cpp

Finally, let's sanity-test by serving the gguf file with llama-cpp.

```sh
uv add llama-cpp-python
```

Test script:

```python
from llama_cpp import Llama

llm = Llama(model_path="./outputs/gguf/mistral-7b-base-q8_0-v1.gguf")

prompt = "What is the capital of France?"

output = llm(
    prompt,
    stop=["\n"],
    echo=True,
)
print(output)
```

### Output:

```sh
llama_perf_context_print:    graphs reused =         14
{'id': 'cmpl-b299dd8b-cefe-4cff-b7ff-41bc9594af0b', 'object': 'text_completion', 'created': 1770634954, 'model': './outputs/gguf/mistral-7b-base-q8_0-v1.gguf', 'choices': [{'text': 'What is the capital of France?  Paris.  What is the capital of the USA?  Washington D.', 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 8, 'completion_tokens': 16, 'total_tokens': 24}}
ggml_metal_free: deallocating
```
