# Running Local SLM

It is 2025 and the AI is booming. So far, large enterprises are all mostly using the Large Language Models of Claude/OpenAI via Cloud Providers to run the inference. This approach works great as a state of the art, but has few known disadvantages:

- High Latency, the LLM inference runs slow which causes long wait times
- Potentially the too large language models are not good specialists, and are too big of a weapon for simple enterprise specific tasks

Of course, the advantages beat disadvantages which is why everyone runs LLMs.

On the other hand, to run local inference, Nvidia GPUs are required, unless some quantization hacks are applied, and the models can then run on CPU. This of course adds-up in production, hence another disincentive to run local LLMs.

In this note, I want to run the SLM (Small Language Model) locally on my machine, and validate the performance for the small projects. I am going to start by doing the following:

1. Choose the appropriate framework to run local SLM. From my research the best available are `vLLM` and `llama-cpp`.
2. Find the SLM appropriate for my task
3. Build a FastAPI server to access the model
4. Run everything on MAC
5. Try running in AWS cloud in the end

## Choosing the SLM framework

There are two major players for running inference: [vLLM](https://github.com/vllm-project/vllm) and [Llama.cpp](https://github.com/ggml-org/llama.cpp). It seems that running `vLLM` on MAC is impossible, as it would require GPUs like Nvidia Jetson, which last time I checked are worth-while starting from 8GB and these development kits range from 400$+ and that is not the investment I am willing to make.

`Llama.cpp` is the next best choice, and with quantization it promises to run inferences in reasonable time. So let's start with this one.

## Using Llama-cpp-python

Most of the AI apps are written in python these days, so gladly there is a [Llama-cpp-python](https://github.com/abetlen/llama-cpp-python) that we can integrate into our python projects. The stack I am going to use for this proof of concept - too keep it real with production example:

- FastAPI wrapper API
- LangChain and LangGraph as an AI framework
- Llama-cpp-python as a backend.

It also seems from the docs of [Llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/) that we are in luck and this lib is compatible with LangChain and even provides OpenAI-like API.

### Building LLama-cpp-python Server

So let's begin from the backend. We will start by building a simple python server that serves the inference. You can follow along in the repo I created here https://github.com/vvasylkovskyi/slm-llama-cpp-poc.

### Add dependencies

Let's start by adding dependencies. I got scared from this point on when I saw in the [Llama docs](https://llama-cpp-python.readthedocs.io/en/latest/) installation instructions saying: **"If it fails, add --verbose to the pip install see the full cmake build log."**. I tried nevertheless:

```sh
uv sync
Resolved 1 package in 1ms
Audited in 0.00ms

uv add llama-cpp-python
Resolved 7 packages in 6.92s
      Built llama-cpp-python==0.3.16
```

It seems I was lucky, and it didn't fail for me, so let's move along.

### Invoking the model

Next, I pasted from the docs, an example of calling the model:

```python
from llama_cpp import Llama

llm = Llama(
      model_path="./models/7B/llama-model.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)
output = llm(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)
```

To my surprise it failed, obviously because I didn't download the model:

```sh
uv run python slm/main.py

##
uv run python slm/main.py
Traceback (most recent call last):
  File "~/git/slm-llama-cpp-poc/slm/main.py", line 3, in <module>
    llm = Llama(
          ^^^^^^
  File "~/git/slm-llama-cpp-poc/.venv/lib/python3.12/site-packages/llama_cpp/llama.py", line 370, in __init__
    raise ValueError(f"Model path does not exist: {model_path}")
ValueError: Model path does not exist: ./models/7B/llama-model.gguf
```

The `./models/7B/llama-model.gguf` doesn't seem like a real model name, so I am off to find the model I need.

### Finding the Model

By browsing Hugging Faces I found the the model that was suggested at https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF. It was a bit confusing about what to do next, whether to download the model or use it using the `fromPretrained` snippet like this:

```python
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
    filename="*q8_0.gguf",
    verbose=False
)
```

I tried both. Both do the same, although doing the python way seems best as it not only downloads model but also caches in system which is useful for subsequent calls. So will describe how to run it below. and it is the same thing, except the `from_pretrained` feels two in one.

## A bit of Theory about Small Language Model

We are going to use a 500M Parameters Small Language Model with 8-bit quantization. What does it mean?

### What model is this?

`Qwen/Qwen2-0.5B-Instruct-GGUF` The model used here is Qwen2-0.5B-Instruct, a Small Language Model (SLM) with roughly 500 million parameters. Compared to state-of-the-art LLMs (GPT-4, Claude), this model is tiny (GPT-4 has 1-2T parameters, and GPT-3 has 100B+), but that is exactly why it works well for local inference and small, specialized tasks.

### What does “500 million parameters” mean?

Parameters are the learned weights of the neural network. With 500M parameters, this model should be good at:

- Simple chat and Q&A
- Text classification and extraction
- Structured, domain-specific prompts
- Deterministic tasks inside enterprise workflows

It is not good at:

- Deep multi-step reasoning
- Long-context synthesis
- Creative writing or planning-heavy tasks

This is a deliberate trade-off: less intelligence, much lower cost and latency.

### What is quantization?

By default, models are stored in high-precision floating point (32FP or 16FP) formats and require GPUs. Quantization reduces the precision of model weights (for example to 8-bit integers), which:

- Dramatically lowers memory usage
- Enables CPU-only inference
- Slightly reduces output quality

In this setup, the model is `Q8_0` quantized, meaning it uses 8-bit integers, which keeps quality high while making it fast enough to run on a laptop or a small EC2 instance.

### Downloading the model

Now, let's get back to coding. So I tried to download the model, these are the commands I had to run:

```sh
brew install huggingface-cli # Get hugging faces CLI
export HF_TOKEN= # Export my Hugging Faces token to improve download speed
hf download Qwen/Qwen2-0.5B-Instruct-GGUF # Start Actual download
```

The download was going to take around 10 minutes, as the model is 4GB. Seemed a bit slow, so while it is downloading I decided to give it a try and use `from_pretrained`

### Using from_pretrained

I tried using `from_pretrained` as suggested on the Hugging Face website. After installing dependencies and preparing the code snippet like this:

```py
from llama_cpp import Llama
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF", filename="*q8_0.gguf", verbose=False
)

output = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "Hello! How are you?",
        }
    ]
)
print(output)
```

The `from_pretrained` requires `huggingface-hub` dependency (probably to download the model), so we install it and then run the code above:

```sh
uv add huggingface-hub
uv run python ./slm/main.py
```

There was a little wait time - corresponding to download, and then we started to actually execute the model:

```sh
.venv/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py:202: UserWarning: The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`. Downloading to a local directory does not use symlinks anymore.
  warnings.warn(
llama_context: n_ctx_per_seq (512) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
ggml_metal_init: skipping kernel_get_rows_bf16                     (not supported)
ggml_metal_init: skipping kernel_set_rows_bf16                     (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32                   (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32_c4                (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32_1row              (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32_l4                (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_bf16                  (not supported)
ggml_metal_init: skipping kernel_mul_mv_id_bf16_f32                (not supported)
ggml_metal_init: skipping kernel_mul_mm_bf16_f32                   (not supported)
ggml_metal_init: skipping kernel_mul_mm_id_bf16_f16                (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h64           (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h80           (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h96           (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h112          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h128          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h192          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_hk192_hv128   (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h256          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_hk576_hv512   (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h64       (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h96       (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h128      (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h192      (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_hk192_hv128 (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h256      (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_hk576_hv512 (not supported)
ggml_metal_init: skipping kernel_cpy_f32_bf16                      (not supported)
ggml_metal_init: skipping kernel_cpy_bf16_f32                      (not supported)
ggml_metal_init: skipping kernel_cpy_bf16_bf16                     (not supported)
{'id': 'chatcmpl-4047c3d5-4edc-4e27-8e8a-1ccd730f2938', 'object': 'chat.completion', 'created': 1765806462, 'model': '~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/./qwen2-0_5b-instruct-q8_0.gguf', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': "Hello! I'm an AI assistant, so I don't have feelings or emotions. I'm here to help you with any questions or tasks you have. How can I assist you today?"}, 'logprobs': None, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 25, 'completion_tokens': 38, 'total_tokens': 63}}
```

Seems that everything worked correctly! The initialization took a while which seems to be related to cold start, which I believe will not happen when running a web server because the `from_pretrained` will be persisted. On the second run, the load was much faster, which confirms the fact that once the model is downloaded, then it is cached and reused. I found the cached model hereThe cached one is here `.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/*`.

## Running in production

That all seems pretty reasonable, but how does it work in production? There are several steps involved:

1. First, we need to expose the FastAPI server that will serve the inference
2. The model needs to be available on the host machine in the cloud, or in the docker image.

From my investigation there are several ways of ensuring that your model is available in production. The most simple way I see is to bake it directly inside the docker image. This has a few advantages such as:

- Removes the network dependency on huggingface hub during instance startup, since the model is baked into docker image
- Faster initial load, because the model is downloaded during build time

The disadvantages is the large size of docker image - 4-5GB due to model size. To compare with `python:3.12-slim` it has around 150MB. This is a trade-off that I can sleep with, so let's go ahead with that plan:

### Building a simple FastAPI server

Let's wrap our model in a FastAPI then:

#### Install dependencies

Add dependency to fastAPI:

```sh
uv add fastapi
uv add uvicorn
```

#### Write a server code

Full code below

```python
from fastapi.concurrency import asynccontextmanager
from llama_cpp import Any, Llama
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from collections.abc import AsyncGenerator


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    global llm
    llm = Llama.from_pretrained(
        repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF", filename="*q8_0.gguf", verbose=False
    )

    yield


app = FastAPI(lifespan=lifespan)

health_router = APIRouter()
@health_router.get("/health")
async def health() -> Any:
    return {"status": "ok"}


model_router = APIRouter()
@model_router.post("/generate")
async def generate(request: Request) -> Any:
    global llm

    body = await request.json()
    prompt = body.get("prompt", "")

    output = llm(
        prompt,
        stop=["\n"],
        echo=True,
    )
    return output


app.include_router(health_router)
app.include_router(model_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Note that we are using the `lifecycle` of FastAPI. This is useful for us to initialize something. In our case it is of course our model. The key endpoint that we expose is `/generate`.

#### Starting server

To start the server locally run:

```sh
uv run uvicorn slm.main:app --host 0.0.0.0 --port 80 --reload
```

Notice the warm-up of the server and lots of logs related to the model initialization. Next test the server with cURL

```sh
curl --location 'http://127.0.0.1:80/generate' \
--data '{
  "prompt": "hello"
}'
```

You should see output similar to the next one

```json
{
  "id": "cmpl-7771e87d-359e-4f4f-8b00-60db5bfdbc80",
  "object": "text_completion",
  "created": 1765886188,
  "model": "~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/./qwen2-0_5b-instruct-q8_0.gguf",
  "choices": [
    {
      "text": "hello，大家好！我是【小唐】！我姓唐，名小",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 16,
    "total_tokens": 17
  }
}
```

It is funny that it answered in Chinese! The answers are pretty random at this point so you will probably have different output

#### Changing Image Load Strategy

When switching into docker, we will bake the model into the environment, so let's update our code to use the downloaded model rather than fetching it at the runtime.

```python
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    global llm
    llm = Llama(
        model_path=os.environ["MODEL_PATH"],
        n_ctx=2048,
        n_threads=8,
    )
    yield
```

In my `.env` I defined the `MODEL_PATH` as

```sh
export MODEL_PATH=~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q8_0.gguf
```

Which is where the model got cached for me. Note, this variable was set by us in the docker file, so no need to set it in production.

### Building Docker Image

After trying to run this and seeing it at work locally, I move on to the next step and test from inside the docker. We are backing the model, so we can be sure that wherever this image will run, it will run as expected. In other words, the `Dockerfile` will download the image amd expose it via environment variable.

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install build tools required by llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir \
    llama-cpp-python \
    huggingface-hub \
    fastapi \
    uvicorn

# Download model at build time
RUN python - <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
    allow_patterns="*q8_0.gguf",
    local_dir="/models",
    local_dir_use_symlinks=False,
)
EOF

ENV MODEL_PATH=/models/qwen2-0_5b-instruct-q8_0.gguf

COPY . .
CMD ["uvicorn", "slm.main:app", "--host", "0.0.0.0", "--port", "80"]
```

As a nice local utility, I built a quick `docker-compose.yaml`:

```yaml
services:
  server:
    container_name: slm-llama-cpp-poc
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "80:80"
```

Now run `docker compose up` and wait for the image to build, and see it working fine.

## Deployment on AWS

Finally, we got to the point where we have a code to generate a working docker image. This image is the app that we built containing a FastAPI server with one endpoint serving the Small Language Model. With this setup we can build some pet project where people can use our AI capabilities without exploding our OpenAI bill. This is pretty exciting!

The next step is to deploy this image on AWS. For that I invite you, to follow my tutorial: [IaC Toolbox](https://www.iac-toolbox.com). And set the docker image manually in the EC-2 `user_data` specs. Don't worry if you are new to AWS, that tutorial has got you covered.

The `user_data` script for the EC-2 instance is as follows:

```sh
#!/bin/bash
sudo apt-get update -y
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USERNAME

sudo docker run -d --name slm-poc -p 80:80 <your-docker-username>/<your-image>:b69f40586b04da676a11f2fc2af676e403cf0b53 # My Image tag (read below on how to get it)
```

Note, I am using the API Gateway as an SSL proxy. To my surprise, even the EC-2 `t2.micro` can run SLM on its tiny CPU. Well done `Llama-cpp`! Taking around 2-3 seconds but it is very decent!

### Note on the Architecture

Beware that if you built the docker image on your MAC, it will most likely not work on the AWS EC-2 Linux machines due to CPU Architecture mismatch. To avoid this issue, the best thing to do is to build docker image directly on the Linux. One of the ways of doing it is by using Github Actions, that are free and have Linux Machines.

The following code shows how to use Github Workflows to dispatch Github Actions. The script builds docker image and publishes it in docker hub:

```yaml
# .github/workflows/build-image.yaml

name: CI

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      version:
        description: "Build Docker Image"
        required: true

jobs:
  build_docker_image:
    runs-on: ubuntu-latest
    steps:
      - name: Check out the repo
        uses: actions/checkout@v4

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and Push Container Image
        run: |
          TAG=${{ github.sha }}
          echo "Using tag: $TAG"
          docker build -f ./Dockerfile -t ${{ secrets.DOCKER_USERNAME }}/<your-docker-image>:$TAG .
          docker push ${{ secrets.DOCKER_USERNAME }}/<your-docker-image>:$TAG

      - name: Set output tag
        id: set_tag
        run: echo "tag=${{ github.sha }}" >> "$GITHUB_OUTPUT"
```

Make sure that your Github Actions have the `DOCKER_USERNAME` and `DOCKER_PASSWORD` before running the action. Run the action.

Once the action is done, grab the tag from docker hub and keep it for the image. In my case it is:

```sh
<your-docker-username>/<your-docker-image>:b69f40586b04da676a11f2fc2af676e403cf0b53
```

To my surprise the compressed size is only 700MB, much less than I expected. And it is a good thing.

## Conclusion

Today we tried out a small language model, and even run it on top of AWS in the smallest ec2! Super excited with these results. Although, there are many non-validated things yet as we haven't validated how it behaves at load, how will it be with bigger context window and tool choice, as everybody is used to do with LLMs. Nevertheless, this is a great start and amazing for pet projects and showcase. Plus no more fear of leaking your `OPENAI_API_KEY` and loosing all your money ^^. Stay tuned!
