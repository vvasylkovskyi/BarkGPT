import os
from typing import Any
from fastapi.concurrency import asynccontextmanager
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from collections.abc import AsyncGenerator

import torch
from bark_gpt.model.hf.bark_hf import BarkConfig, BarkHF
from bark_gpt.model.model import BarkGPT


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    global stoi
    global itos
    global hf_model

    checkpoint = torch.load(os.environ["MODEL_PATH"], map_location="cpu")
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = checkpoint["vocab_size"]

    bark = BarkGPT(vocab_size)
    bark.load_state_dict(checkpoint["model_state"])

    config = BarkConfig(vocab_size=vocab_size)
    hf_model = BarkHF(config, bark)

    yield


app = FastAPI(lifespan=lifespan)
health_router = APIRouter()


@health_router.get("/health")
async def health() -> Any:
    return {"status": "ok"}


def route_user_input():
    # Anything the user types means "please bark"
    return "User: speak Assistant:"


def detokenize_output(tokens, output_ids):
    output_tokens = [itos[i] for i in output_ids[0].tolist()]
    generated_tokens = output_tokens[len(tokens) :]

    if "<EOS>" in generated_tokens:
        eos_index = generated_tokens.index("<EOS>")
        generated_tokens = generated_tokens[:eos_index]

    return " ".join(generated_tokens)


model_router = APIRouter()


@model_router.post("/generate")
async def generate(_: Request) -> Any:
    global stoi
    global itos
    global hf_model
    prompt = route_user_input()
    tokens = prompt.split()

    input_ids = torch.tensor([[stoi[t] for t in tokens]])

    output_ids = hf_model.generate(
        input_ids=input_ids, max_new_tokens=10, temperature=0.7
    )

    output_text = detokenize_output(tokens, output_ids)

    return {"message": output_text}


app.include_router(health_router)
app.include_router(model_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
