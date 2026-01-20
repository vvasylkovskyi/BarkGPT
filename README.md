# BarkGPT

<div style="display: flex; justify-content: center; margin-bottom: 10px; "><img src="./og.png" alt="BarkGPT - The Smallest LLM" width="100%" height="250"/></div>

AI LLM Project that barks trained from scratch.

This is a hands-on tutorial about how to use Large Language Models and in particular, train one from scratch. Our goal here, is to train an LLM from scratch, with very low vocabulary, only dog barking.

While there might not be many applications for such a model (unless you want to build a robot dog), I always felt intimidated by LLMs and the fact that they are trained on vast amounts of datasets. Having tiny dataset, with small vocabulary, removes one of the complexities in the learning curves. Unsurprisingly understanding an LLM, even that small is still quite hard.

This project is heavily inspired on [CatGPT](https://www.cat-gpt.com/), an AI that meows. I am a cat owner myself, although the market for meow AIs feels quite saturated with clear market dominance. I haven't seen any worthy dog GPTs, only mere amateur projects, so I think this project will just fit right in.

## Demo

Try demo at [BarkGPT](https://www.bark-slm.com/)

## Overview

To get our BarkLLM from scratch, we need to do the following steps:

1. Prepare the dataset. This can be a small amount of words, but relatively large repetition for the further learning
2. Prepare the overall Transformers model, with GPT-like APIs: `generate`, layers and all the deep LLM attributes
3. Train the LLM on the dog dataset.
4. Finally, test the generation.

In the end of this tutorial we are also going to learn how to export such model in a `Gguf` format, which is fast for inference and makes LLMs modular and easy to deploy in production, docker containers.

## Preparing a Dataset

First things first, what is the vocabulary that our dog would speak? Well the "woof" and so on right? Not so simple.

Clearly, the actual dog sounds are required, but there are also padding, end of file, and, the prompt contract tokens. We will see in detail what it means. For now, let's define our `prepare_dataset.py`, a script that will create sintetic dataset from our vocabulary.

Let's assume our vocabulary is the following:

```python
vocab = [
    "<PAD>",
    "<EOS>",
    "woof",
    "arf",
    "ruff",
    "grrr",
    "User:",
    "Assistant:",
    "speak",
]
```

Now, we can make couple of sentences out of the above:

```python
import random
import torch

vocab = [
    "<PAD>",
    "<EOS>",
    "woof",
    "arf",
    "ruff",
    "grrr",
    "User:",
    "Assistant:",
    "speak",
]


stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for i, tok in enumerate(vocab)}
vocab_size = len(vocab)


# Generate synthetic sequences
def make_sequence():
    out = random.choice(["woof", "arf", "ruff", "grrr"])
    return ["User:", "speak", "Assistant:", out, "<EOS>"]

dataset = [make_sequence() for _ in range(2000)]
dataset_ids = [torch.tensor([stoi[t] for t in seq]) for seq in dataset]
```

an example of what is in `dataset` and `dataset_id` is:

```sh
>>> Sample sequence: [['User:', 'speak', 'Assistant:', 'grrr', '<EOS>'], ['User:', 'speak', 'Assistant:', 'arf', '<EOS>']]
>>> Sample sequence IDs: [tensor([6, 8, 7, 5, 1]), tensor([6, 8, 7, 3, 1])]
```

As you can see, our vocabulary is of 9 words, although the dog will only use 4. Why do we need all the words? When training LLM, what we are doing is teaching the model how to predict the next word, or a token. This prediction is based on previously observed data - the training data. We can't just say anything and this would make the LLM bark, but we can train it to bark when the sequence of words is something that it has seen before.

Let's take our example. So we have few sentences like: `User: speak, Assistant: grrr`, `User: speak, Assistant: woof`. Clearly, in the above examples, given the sequence: `User: ..., Assistant:`, the next word to predict is `grrr` or `woof`. That is how we are going to tailor our BarkGPT. The secret is to ensure that the sequence entering the model is always of this form. This is called the prompt contract; the LLM is trained to predict next token based on the previous sequence, hence we will enforce the sequence.

## Building our Model

Now we have the dataset. The next thing we need to prepare to train the model is the model itself. Below I will throw a sample BarkGPT model:

```python
import torch.nn as nn

class BarkGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=32, n_layer=2, n_head=2, seq_len=16):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, n_embd))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=n_embd, nhead=n_head, dim_feedforward=128
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.token_emb(x) + self.pos_emb[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
```

The model above is our core LLM. It uses the `torch` (also known as [PyTorch](https://pytorch.org/)) to define the model. The LLM is no more than some mathematical functions, with the values that can be adjusted such that the behavior of those mathematical functions change. Those values, are what is usually refered to as parameters, and the functions are trained. When we inherit from `nn.Module`, we declare: "This object is a trainable neural network.".

Unfortunately, to the best of my effort, explaining every line of that code would be a paper in itself. And it should exist. So let's assume for now, that this will work, and go to the next step: train the model.

## Training the model

Now that we have our model, the next step is to train it on our dataset, so that the words prediction works and our model barks.

So in high-level steps we are going to perform the following steps:

1. Load the dataset
2. Load our model
3. Train it on 20 epoch, and observe the loss decreasing. When loss decreases that means that the probability of getting the next word right is higher.
4. Save the model as a file: `pt` (pytorch)

The overall code is as following:

```python
import torch
import torch.nn as nn
from bark_gpt.prepare_dataset import prepare_dataset
from bark_gpt.model import BarkGPT

dataset, dataset_ids, vocab_size, stoi, itos = prepare_dataset()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkGPT(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

batch_size = 16
seq_len = max(len(seq) for seq in dataset_ids)


# pad sequences
def pad(seq):
    pad_len = seq_len - len(seq)
    if pad_len > 0:
        return torch.cat([seq, torch.tensor([stoi["<PAD>"]] * pad_len)])
    return seq


padded = torch.stack([pad(seq) for seq in dataset_ids]).to(device)

for epoch in range(20):
    perm = torch.randperm(len(padded))
    total_loss = 0
    for i in range(0, len(padded), batch_size):
        batch = padded[perm[i : i + batch_size]]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(padded):.4f}")

torch.save(
    {
        "model_state": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
    },
    "bark_model.pt",
)
```

When running `make train`, we get the following output:

```sh
Epoch 1, Loss: 0.0357
Epoch 2, Loss: 0.0233
Epoch 3, Loss: 0.0225
Epoch 4, Loss: 0.0222
Epoch 5, Loss: 0.0221
Epoch 6, Loss: 0.0219
Epoch 7, Loss: 0.0220
Epoch 8, Loss: 0.0219
Epoch 9, Loss: 0.0219
Epoch 10, Loss: 0.0219
Epoch 11, Loss: 0.0219
Epoch 12, Loss: 0.0218
Epoch 13, Loss: 0.0219
Epoch 14, Loss: 0.0218
Epoch 15, Loss: 0.0218
Epoch 16, Loss: 0.0218
Epoch 17, Loss: 0.0217
Epoch 18, Loss: 0.0218
Epoch 19, Loss: 0.0218
Epoch 20, Loss: 0.0218
```

As we can see, the loss decreased and stopped learning, which is the perfect place to finish the training.

## Using the Model to Predict Tokens

Now we have the model, how to generate tokens? Notice important point: The neural networks never "generate text" by themselves. They only do the math: "what is the probability of this token to appear next", or in AI jargon:

```sh
tokens → logits
```

`Logits` are only numbers, representing probability of a token.

### Generating Tokens

The Generation is a loop around the model. The ideia is to give the tokens so far in the sequence to the model, and ask it to predict the next one. Then append this token, etc.

To make things a bit fun, we will continue sticking to Pytorch, and write a generate function using pure pytorch. We can write a sample function, and a script to test a sentence:

```python
import torch
import torch.nn.functional as F
from bark_gpt.prepare_dataset import prepare_dataset
from bark_gpt.model import BarkGPT

dataset, dataset_ids, vocab_size, stoi, itos = prepare_dataset()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkGPT(vocab_size).to(device)


def generate(model, prompt_tokens, max_new_tokens=10, temperature=1.0):
    model.eval() # turns off the noise of training
    tokens = prompt_tokens[:]
    for _ in range(max_new_tokens): # stop when reached max tokens
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        logits = model(x) # Load number from the model
        logits = logits[0, -1] / temperature # Temperature for creativity
        probs = F.softmax(logits, dim=-1) # turn numbers into probability (from 0 to 1)
        next_id = torch.multinomial(probs, num_samples=1).item() # Predict next token randomly
        tokens.append(next_id)
        if next_id == stoi["<EOS>"]: # Stop if end of string
            break
    return tokens


# example prompts
prompts = [
    ["User:", "speak", "Assistant:"],
]

out_ids = generate(
    model, [stoi[t] for t in prompts[0]], max_new_tokens=5, temperature=0.7
)
out_text = " ".join([itos[i] for i in out_ids])
print("=== Test Generation === ")
print(f"Prompt: {' '.join(prompts[0])} → {out_text}")
```

Essentially the function `generate` above is a pure pytorch that iterates on top of our model, and feeds it the sequence of text already seen, and asks: "Generate next token randomly, based on the probability function".

After execution, we can see the output is:

```sh
=== Test Generation ===
Prompt: User: speak Assistant: → User: speak Assistant: ruff arf woof User: speak
```

So our model generated the next words as dog barks! This is exactly right prediction.

## Observations so far

We have built a model from scratch that barks when sees sequence of type: `User: speak Assistant: `. This is our very basic AI LLM. What will happen when the sequence is `User: hey Assistant`? Funnily, our model was never trained on this variation, so things might get interesting.

Overall the matematical function of

- If `User: hey Assistant:`, then probability of `arf` or `woof` is ~=0
- If `User: speak Assistant:`, then probability of `arf` or `woof` is ~=1

So in practice what will happen? Well, in practice that is hallucination. The AI will confidently predict next token wrong. This is also called out-of-distribution. Since the model doesn't know how to predict the next token, it will guess. Neural Networks never say "I don't know". They always output something. This is why hallucinations exist.

## Building a generative app

Now we have this model that we prove works, we have essentially built a brain. Next thing is build a generation engine on top of it, which is something usually provided by [Transformers](https://huggingface.co/) runtime.

The Transformers runtime provide options of wrappers of the LLM, for reference, some of them are:

- CausalLM (GPT): They predict next token and used for generation.
- MaskedLM (BERT): They predict missing token anywhere in the sentence, and used for undestanding the sentence, e.g. classification

So clearly, our choice is to use CausalLM.

### Building CausalLM with HuggingFaces

Hugging Faces provide the `transformers` library that can help us load our model and add the generation engine. The simple class that can wrap our model is as follows:

```python
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput


class BarkConfig(PretrainedConfig):
    model_type = "barkgpt"

    def __init__(self, n_layer=2, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = n_layer # 2 layers (default), number of transofmer layers. HF needs this for generating KV caches for attention

class BarkHF(PreTrainedModel):
    config_class = BarkConfig

    def __init__(self, config, bark_model):
        super().__init__(config)
        self.bark = bark_model

    def forward(self, input_ids, **kwargs):
        logits = self.bark(input_ids)
        return CausalLMOutput(logits=logits)

```

Note the `BarkHF` (HuggingFaces), receives our bark model, and predicts the next logits in `CausalLMOutput`. The `generate` is a function that is added to our model, once the `PreTrainedModel` is inherited. So let's test our model now. We need some code to actually load our model and create the `BarkHF`.

```python
from bark_gpt.hf.bark_hf import BarkConfig, BarkHF
from bark_gpt.model import BarkGPT
import torch

checkpoint = torch.load("bark_model.pt", map_location="cpu")
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]

bark = BarkGPT(vocab_size)
bark.load_state_dict(checkpoint["model_state"])

config = BarkConfig(vocab_size=vocab_size)
hf_model = BarkHF(config, bark)
```

The `hf_model` is the combination of our base model: The `BarkGPT` is our model, and `bark_model.pt` is our weights, that we changed as a result of training. The `BarkConfig` keeps all the parameters required for the HuggingFaces `generate()` function. Notice the similarity of parameters with our custom made pytorch function. The only thing we need to define is how many hidden layers are there.

#### Testing CausalLM

So now, the testing script:

```python
from bark_gpt.hf.bark_hf import BarkConfig, BarkHF
from bark_gpt.model import BarkGPT
import torch

checkpoint = torch.load("bark_model.pt", map_location="cpu")
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]

bark = BarkGPT(vocab_size)
bark.load_state_dict(checkpoint["model_state"])

config = BarkConfig(vocab_size=vocab_size)
hf_model = BarkHF(config, bark)


# Example prompt
prompt = "User: speak Assistant:"
# Convert prompt to input IDs
tokens = prompt.split()  # simple tokenization
input_ids = torch.tensor([[stoi[t] for t in tokens]])  # shape [1, seq_len]

# Generate output IDs
output_ids = hf_model.generate(input_ids=input_ids, max_new_tokens=10, temperature=0.7)
# Convert output IDs back to text
output_tokens = [itos[i] for i in output_ids[0].tolist()]
# Remove prompt tokens if you want only the new text
generated_tokens = output_tokens[len(tokens) :]

# Optionally stop at <EOS>
if "<EOS>" in generated_tokens:
    eos_index = generated_tokens.index("<EOS>")
    generated_tokens = generated_tokens[:eos_index]

# Join tokens into a string
out_text = " ".join(generated_tokens)

print("Generated:", out_text)
```

If we run it, the generated output is:

```sh
Generated: arf
```

### Making Model More Conversational

So this function already allows us to predict the next word. Although, it is only one `arf` ? Why is that? Well, because we only trained our model on datasets that contain one word and end:

```sh
User: speak Assistant: <dog-bark> <EOS>
```

It would be more interesting if our BarkGPT would be more conversational, maybe bark a little more. This can be done by retraining our model on different sequences of barks. Remember our dataset creation is as follows:

```python
def make_sequence():
    out = random.choice(["woof", "arf", "ruff", "grrr"])
    return ["User:", "speak", "Assistant:", out, "<EOS>"]
```

Only one token is placed in the sequence. We can change the script to update our model.

```python
def make_sequence():
    num_barks = random.randint(2, 10)  # multiple barks
    barks = [random.choice(["woof", "arf", "ruff", "grrr"]) for _ in range(num_barks)]
    return ["User:", "speak", "Assistant:"] + barks + ["<EOS>"]
```

Now, let's train the model from scratch again, and test:

```sh
make train
make test_hf
```

I ran the `test_hf` script 3 times, and got outputs as follows:

```sh
Generated: ruff ruff ruff woof woof woof woof woof woof woof
Generated: ruff woof woof woof woof
Generated: ruff ruff grrr ruff woof woof woof arf arf arf
```

That's an improvement and our model is already conversational, like an angry dog.

### Adjusting Generation Parameters

An additional way to make a model more conversational is by tweaking the generation engine parameters. Let's inspect them:

```python
output_ids = hf_model.generate(
    input_ids=input_ids,
    max_new_tokens=10,
    temperature=1.0,
    top_k=4,
    top_p=0.9
)
```

There are three additional parameters: `temperature`, `top_k`, and `top_p`. Let's uncover what each of them with other example:

#### Temperature

Remember from our pytorch generate function:

```python
def generate(model, prompt_tokens, max_new_tokens=10, temperature=1.0):
    model.eval() # turns off the noise of training
    tokens = prompt_tokens[:]
    for _ in range(max_new_tokens): # stop when reached max tokens
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        logits = model(x) # Load number from the model
        logits = logits[0, -1] / temperature # Temperature for creativity
        probs = F.softmax(logits, dim=-1) # turn numbers into probability (from 0 to 1)
        next_id = torch.multinomial(probs, num_samples=1).item() # Predict next token randomly
        tokens.append(next_id)
        if next_id == stoi["<EOS>"]: # Stop if end of string
            break
    return tokens
```

Notice the `temperature` is used on the line:

```python
logits = logits[0, -1] / temperature
```

The logits are numbers. The bigger the number, the more likely the word to be picked up to be generated next. Suppose our model logits for 4 tokens after `Assistant: `

| Token | Logit |
| ----- | ----- |
| woof  | 2.0   |
| arf   | 1.0   |
| ruff  | 0.5   |
| grrr  | 0.1   |

Imagine if the temperature is `temperature=1`. The next line of code, converts the logits into probabilities using `softmax` function:

```sh
probs = F.softmax(logits, dim=-1)
```

$$
P_i = \frac{e^{\text{logit}_i / T}}{\sum_j e^{\text{logit}_j / T}}
$$

This function transforms the numbers and then represents each number as a probability, by dividing it by the sum of them all. So suppose our logits above got transformed into the following probabilities:

| Token | Probability        |
| ----- | ------------------ |
| woof  | 2.72 / 6.70 ≈ 0.41 |
| arf   | 1.65 / 6.70 ≈ 0.25 |
| ruff  | 1.28 / 6.70 ≈ 0.19 |
| grrr  | 1.05 / 6.70 ≈ 0.16 |

The `woof` has high probability to be picked up, but so others are also. In fact it is quite random. Now, let's suppose the `temperature=0.5`. After the math probabilities table becomes:

| Token | Probability         |
| ----- | ------------------- |
| woof  | 54.6 / 65.93 ≈ 0.83 |
| arf   | 7.39 / 65.93 ≈ 0.11 |
| ruff  | 2.72 / 65.93 ≈ 0.04 |
| grrr  | 1.22 / 65.93 ≈ 0.02 |

Now the `woof` has much higher probability of being predicted, hence the "creativity" is lower. The oposite happens when `temperature>1`, the randomness increases.

#### top_k

Top K, is a little easier to reason about. Instead of randomly predicting among all the possible words, the Top K narrows down to `K` words. Suppose in our example, there are 4 possible tokens to predict - the variations of barks. If we set `top_k=2`, then only top 2 most likely tokens will be predicted.

#### top_p

Similar to Top K, the Top P, randomly picks among tokens summing to some probability (adaptive set). However, `top_p` is a percentage, rather than a number, so can be in range of `0-1`. For instance, if our probabilities table is:

| Token | Probability         |
| ----- | ------------------- |
| woof  | 54.6 / 65.93 ≈ 0.83 |
| arf   | 7.39 / 65.93 ≈ 0.11 |
| ruff  | 2.72 / 65.93 ≈ 0.04 |
| grrr  | 1.22 / 65.93 ≈ 0.02 |

Let's say our `top_p=0.9` (90%). The cummulative probabilities of the token above are:

| Token | Probability | Cumulative Probability |
| ----- | ----------- | ---------------------- |
| woof  | 0.83        | 0.83                   |
| arf   | 0.11        | 0.83 + 0.11 = 0.94     |
| ruff  | 0.04        | 0.94 + 0.04 = 0.98     |
| grrr  | 0.02        | 0.98 + 0.02 = 1.00     |

So the `top_p=0.9` will allow to pick the smallest set of tokens whose cumulative probability ≥ 0.9.

- `woof` → cumulative 0.83 < 0.9 → include next token
- Add `arf` → cumulative 0.94 ≥ 0.9 → stop here

So the token set will become only `{woof, arf}` in that scenario.

## Routing user input

So now we have made our BarkGPT conversational. But there is still a problem with user input. See, our model is trained to predict barks only when receiving text like: `"User: speak Assistant:"`. What if user inputs something else than `speak`? The model will hallucinate.

In the production LLM system, the user input (prompt) is often normalized, before feeding as input into LLM. Essentially the LLMs often have a system prompt and a routing layer. The combination of them both provides a normalized representation of input the model is trained to see.

In our scenario, we will normalize user input according to the following rule:

```python
def route_user_input(user_text):
    # Anything the user types means "please bark"
    return ["User:", "speak", "Assistant:"]
```

This might look like a hack, but it has parallels with real LLM Systems, where the user input is normalized, so this design is correct. We might want to place the `route_user_input` function in our LLM application, but not inside the model itself (hugging faces or raw pytorch).

## Building an app

Now that we have all the building blocks of our `BarkGPT`: we have the model and we have the generation engine, we can start packaging our app.

### Building a FastAPI to serve the /generate endpoint

We will create a simple FastAPI server that will serve the inference endpoint.

#### Install dependencies

Add dependency to fastAPI:

```sh
uv add fastapi
uv add uvicorn
```

#### Write a server code

Full code below

```python
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

    checkpoint = torch.load("bark_model.pt", map_location="cpu")
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
```

The above app is serving `generate` endpoint which a user can use to talk with our AI. Notice the `asynccontextmanager` is the place where we are loading our model - it is loaded only once at the server start which is great because the subsequent requests to generate will reuse this object and will be fast.

#### Starting server

To start the server locally run:

```sh
uv run uvicorn slm.main:app --host 0.0.0.0 --port 80 --reload
```

And sending a cURL:

```sh
curl --location 'http://localhost:80/generate' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "hey there"
}'
```

Then answer is:

```json
{
  "message": "ruff woof grrr grrr"
}
```

So our model is being served correctly!

### Publishing model to Hugging Faces

It is time now to publish our model artifact: `bark_model.pt` in Hugging Faces, our ML-native platform to store AI models. This is an important step both for versioning and for deployment, as we are going to use Hugging Faces to load our model at the build of our application, this way the model is always available in our app environment.

#### Login To Hugging Faces

```sh
huggingface-cli login
```

Input your token. If you don't have a token, then create an account and a token on https://huggingface.co/. Ensure that your token has write access as we are going to need it to publish our model.

#### Create a repository for the model

```sh
repo create barkgpt --type model
Successfully created vvasylkovskyi/barkgpt on the Hub.
Your repo is now available at https://huggingface.co/vvasylkovskyi/barkgpt
```

#### Upload our model

```sh
huggingface-cli upload barkgpt ./bark_model/bark_model.pt bark_model.pt
```

You should see output similar to

```sh
Processing Files (1 / 1)      : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████|  117kB /  117kB, 83.6kB/s
New Data Upload               : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████|  117kB /  117kB, 83.6kB/s
  ./bark_model/bark_model.pt  : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████|  117kB /  117kB
https://huggingface.co/vvasylkovskyi/barkgpt/blob/main/bark_model.pt
```

So upload worked, our model is deployed!

### Packaging the FastAPI server in docker

Now we have our FastAPI working locally. The last step is to isolate the runtime environment so that this server can be ported to a cloud and serve our model. Our go to tool for this is Docker.

Our docker container will have to include both:

- Our `bark_model.pt`
- And our FastAPI server, that will consume the `bark_model.pt`

This docker container will setup python environment for our FastAPI server and download our model artifact: `bark_model.pt`, place the model in a folder, and export the `MODEL_PATH` environment for our server to consume.

#### Dockerfile

Below is the sample `Dockerfile` that does the above

```Dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    torch \
    transformers

# Create model directory
RUN mkdir -p /models

# Download BarkGPT model from Hugging Face
RUN curl -L -o /models/bark_model.pt \
    https://huggingface.co/vvasylkovskyi/barkgpt/resolve/main/bark_model.pt

# Expose model path
ENV MODEL_PATH=/models/bark_model.pt

# Copy app code
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

Since our model is public, there is no hussle in obtaining it, a simple download does the trick.

As a nice local utility, I built a quick `docker-compose.yaml`:

```yaml
services:
  server:
    container_name: bark-gpt-api
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "80:80"
```

Now run `docker compose up` and wait for the image to build, and see it working fine.

#### Dockerfile with Torch + CUDA - 4.1GB Size

When I first tried to deploy above script, I fell into what is called lassic EC2 + Docker + PyTorch trap. Our Docker image is 4.1GB, and is the Ec-2 instance is out of space to deploy it. However the our `bark_model.pt` is only 117KB, so where is the rest of the size came from? Turns out, it is from `torch` and all the GPU related code, in particular the `libcudnn_adv.so.9`. To fix this, we can change our docker image such that it only loads the CPU torch - since in practice we are not using GPU at all for this AI project.

The fix is to change our `Dockerfile` to download the `torch` - CPU only version.

```sh
torch==2.9.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

Below is the full updated `Dockerfile`

```Dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    transformers \
    torch==2.9.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /models

RUN curl -L -o /models/bark_model.pt \
    https://huggingface.co/vvasylkovskyi/barkgpt/resolve/main/bark_model.pt

ENV MODEL_PATH=/models/bark_model.pt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

After applying the changes, the docker image now has `362.9 MB`. Yey, great improvement!

## Deployment on AWS

Finally, we got to the point where we have a code to generate a working docker image. This image is the app that we built containing a FastAPI server with one endpoint serving our BarkGPT - a barking AI model, a real good boy! With this setup we can build some pet project, and people can use our AI dog. This is pretty exciting!

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

sudo docker run -d --name bark-gpt-api -p 80:80 <your-docker-username>/<your-image>:b69f40586b04da676a11f2fc2af676e403cf0b53 # My Image tag (read below on how to get it)
```

Note, I am using the API Gateway as an SSL proxy. It is not surprising the our small dog AI model is not very demanding, even the EC-2 `t2.micro` can run SLM on its tiny CPU!

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

## Final Thoughts

Note, this BarkGPT is a fairly small Language Model, so there are few caveats, and differences with real-world deployments.

### Manual Tokenization

For loading as pretrained hugging-faces, which expects at least `gpt-2`, we cannot really package it as Hugging Faces yet, and push the load it from pretrained e.g. `model = AutoModelForCausalLM.from_pretrained(model_path)`. The main issue is the mismatch between expected GPT-2 tokenizer and our custom tokenizer, so `tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)` from HuggingFaces also fails short. Our informal `tokenizer` is essentially:

```python
tokens = prompt.split()
input_ids = torch.tensor([[stoi[t] for t in tokens]])
```

And our `detokenizer` is:

```python
output_ids = hf_model.generate(input_ids=input_ids, max_new_tokens=10, temperature=0.7)
output_tokens = [itos[i] for i in output_ids[0].tolist()]
generated_tokens = output_tokens[len(tokens) :]

if "<EOS>" in generated_tokens:
    eos_index = generated_tokens.index("<EOS>")
    generated_tokens = generated_tokens[:eos_index]

out_text = " ".join(generated_tokens)
```

This works for our tiny BarkGPT, but it’s brittle:

- Crashes if the user types something not in stoi.
- Won’t handle punctuation, capitalization, or spaces properly.
- Doesn’t support standard Hugging Face generation features like skip_special_tokens, batching, or padding.

### Possible Packaging if GPT-2

If we had correct tokenizer for GPT-2, the right way would be to package our model and deploy it to hugging faces. Overall the hugging faces model when packaged looks like following:

```sh
bark_hf/
  config.json
  generation_config.json
  model.safetensors
  special_tokens_map.json
  tokenizer_config.json
  tokenizer.json
```

We already have our model as a HF CausalLM:

```python
class BarkConfig(PretrainedConfig):
    model_type = "barkgpt"

class BarkHF(PreTrainedModel):
    ...
```

Now, we can package it using hugging faces APIs:

```python
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from bark_gpt.hf.bark_hf import BarkConfig, BarkHF
from bark_gpt.model import BarkGPT

checkpoint = torch.load("bark_model.pt", map_location="cpu")
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]

bark = BarkGPT(vocab_size)
bark.load_state_dict(checkpoint["model_state"])

config = BarkConfig(vocab_size=vocab_size)
hf_model = BarkHF(config, bark)

hf_model.save_pretrained("bark_model")
config.save_pretrained("bark_model")

tokenizer_backend = Tokenizer(WordLevel(vocab=stoi, unk_token="<UNK>"))
tokenizer_backend.pre_tokenizer = Whitespace()

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_backend,
    unk_token="<UNK>",
    eos_token="<EOS>",
    bos_token="<BOS>",
    pad_token="<PAD>",
)

tokenizer.save_pretrained("bark_model")
```

Note we have saved into the `bark_model` folder the architecture model (bark_hf), the configuration and the tokenizer. All three form the valid model.

### Production Optimizations for other architectures

The production models with billions of parameters usually are optimized for better inference. There are several ways of serving the model in production. The most widely used, specially if using Small Language Models and CPU is the `GGuf` and to run on [Llama.cpp](https://github.com/ggml-org/llama.cpp). [This article explains in great depth the model formats and Gguf](https://medium.com/@vimalkansal/understanding-the-gguf-format-a-comprehensive-guide-67de48848256). Many production systems alternatively use the [vLLM](https://github.com/vllm-project/vllm) to scale GPU.

However, our model will not be fit to be optimized with `Llama.cpp` because our architecture doesn't match any of the available: e.g Llama, Mistral, Qwen. Unfortunately running `vLLM` on MAC is also not possible, as it would require GPUs like Nvidia Jetson, which last time I checked are worth-while starting from 8GB and these development kits range from 400$+ and that is not the investment I am willing to make.

### Converting Hugging Faces into GGuf

So how to convert our model into GGuf, optimized for inference? [According to this great tutorial](https://github.com/ggml-org/llama.cpp/discussions/2948), the steps are as follow:

1. We already have the model in the right packaging
2. Use the https://github.com/ggml-org/llama.cpp repository
3. Install dependencies in `llama.cpp`: `pip install -r llama.cpp/requirements.txt`
4. Verify installation and functionality of the script: `python llama.cpp/convert_hf_to_gguf.py -h`
5. Finally, convert the model by running the script:

```sh
python llama.cpp/convert.py vicuna-hf \
  --outfile bark-f32-v1.gguf \
  --outtype f32
```
