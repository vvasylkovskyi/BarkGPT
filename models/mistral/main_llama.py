from fastapi.concurrency import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from collections.abc import AsyncGenerator

from vllm import LLM, SamplingParams
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:

    # Initialize vLLM
    global llm
    llm = LLM(
        model="mistralai/Ministral-3-3B-Base-2512",  # Replace with your model path
        # Limit context length for CPU inference (262k is too large for CPU)
        max_model_len=128,  # Reduce from 262144 to 4096
        # OR increase batch size (but this uses more memory):
        # max_num_batched_tokens=262144,
        # Optional: other CPU-friendly settings
        dtype="q8_0",  # Use quantized format for CPU
        # dtype="float32",
        tokenizer_mode="mistral",
        download_dir="./outputs/cache",
        config_format="mistral",
        load_format="mistral",
    )

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(request: GenerateRequest):
    global llm
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    outputs = llm.generate([request.prompt], sampling_params)

    return {
        "text": outputs[0].outputs[0].text,
        "prompt": request.prompt,
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
