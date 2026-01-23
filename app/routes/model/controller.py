from fastapi import APIRouter, Request

# import torch


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


model_router = APIRouter(prefix="/model", tags=["Model"])


@model_router.post("/generate")
async def generate(_: Request):
    # global stoi
    # global itos
    # global hf_model
    prompt = route_user_input()
    tokens = prompt.split()

    # input_ids = torch.tensor([[stoi[t] for t in tokens]])

    # output_ids = hf_model.generate(
    #     input_ids=input_ids, max_new_tokens=10, temperature=0.7
    # )

    # output_text = detokenize_output(tokens, output_ids)

    # return {"message": output_text}
    return {"message": "Not implemented yet"}
