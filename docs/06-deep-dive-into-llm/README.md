# Understanding BarkGPT Model and Making it smarter

There are three pieces of puzzle in making AI better:

- Bigger Dataset improves LLM ability to learn from dataset
- Longer training time decreases loss, hence improves the LLM ability
- More model parameters will make model more capable

In here we will deep dive into the BarkGPT model internals and answer the question about how smart it is in its underlying architecture, and how can we make it better.

## Prerequisites

We have built, trained and deployed the LLM in the previous articles. It is recommended to catch-up if you want to get the fullest from this article:

Here are the previous reads:

- [Building BarkGPT from scratch](../01-building-from-scratch/README.md)
- [Training BarkGPT on WebText‑2 dataset](../02-training-on-webtext/README.md)
- [Building AI Agents From Scratch](../03-barkgpt-to-barkagent/README.me)
- [Deploying SLM](../04-deploying-small-language-models/README.md)
- [Faster Training for LLM](../05-faster-training-time/README.md)

## Demo

The model trained here is available online for you to try it out. It is small enough to run inferrence on my AWS EC2.

- [Live Demo](https://www.bark-slm.com) - Try it yourself - this is Bark Agent
- [Github Code is available in this repo in /app folder](https://github.com/vvasylkovskyi/BarkGPT/tree/main/app) - Check it out and run locally!

## Understanding GPT Architecture

The dive-in this section will focus on the following piece of code that represents our model:

```python
import torch
import torch.nn as nn


class BarkGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=32, n_layer=2, n_head=2, max_seq_len=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, n_embd))
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
        if T > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.pos_emb.shape[1]}"
            )
        x = self.token_emb(x) + self.pos_emb[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
```

That is not that much code right? But if you are not an LLM engineer, most of it is essentially giberish. So in this section we will inspect each line of code and understand what it means.

### Setting up the vectors - model at zero

It is important to remember that LLM model is essentially a neural network. A neural network is basically a "trainable" network of arrays. So those arrays/vectors need to be initialized when the model starts.

This is done in the `__init__` function:

```python
class BarkGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=32, n_layer=2, n_head=2, max_seq_len=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, n_embd))
```

`nn` - the neural network module, sets up here two vectros called token embeddings and positions embeddings.

#### Token Embedings

To understand best what are those arrays, we will use our `BarkGPT` example with vocabulary of 9 words. To simplify even further, let's assume we have only 3 words:

```python
0 → "woof"
1 → "arrf"
2 → "ruff"
```

This means that our `vocab_size=3`. The `n_embd=32` means the size of the array that we will use to represent each token. The arrays are used because they are more expressive than integers. Hence the bigger the array, the more expressive it is. We will get to what it means shortly.

Right now, our initialized `self.token_emb` is matrix of 3 x 32. Let's also assume that `n_embd=8` for simplicity of representation. So each token is a row in the matrix. Here is how our `self.token_emb` will look like follows for simplification.

```sh
[0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0]
```

Note that in practice PyTorch’s `nn.Embedding` initializes embeddings randomly, not with zeros. The reason is if we start with all zeros, every token embedding starts identical. During the first training steps, all tokens would produce the same gradient, which slows down learning. Random initialization ensures embeddings start different enough for the model to learn meaningful distinctions.

During training, the token embeddings will update their weights because the words might appear closer by each other so the model will have to update its weights.

#### Positional Embeddings

So we have represented each word as a vector, the next is to represent position of each word as a vector:

```python
self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, n_embd))
```

The `self.pos_embeding` expects the `max_seq_len` and not the vocabulary size, because given the sentence, the positional embeddings represents the position of the word in the sequence.

Let's assume that we only accept sentences with up to 3 words, so `max_seq_len=3` so the tensor/matrix will look like:

```sh
Position 0 → [0, 0, 0, 0, 0, 0, 0, 0]  # embedding for first word in sequence
Position 1 → [0, 0, 0, 0, 0, 0, 0, 0]  # embedding for second word
Position 2 → [0, 0, 0, 0, 0, 0, 0, 0]  # embedding for third word
```

Likewise, during training, these vectors will be learned, so each position will carry meaningful “positional” information that the model can use when combined with token embeddings.

### The Transformer Layers

So far we have only initialized our tensors, now it is time to run the actual iteration that will approximate the weights to the final result.

```python
self.layers = nn.ModuleList(
    [
        nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=128
        )
        for _ in range(n_layer)
    ]
)
```

This is the core piece of our LLM architecture where the "thinking" happens. There are several high-level steps:

1. the `self.layers` are a stack of transformer layers.
2. The `nn.ModuleList` is a container for PyTorch modules that can be iterated over during the training in the `forward` function.

Essentially each time the layer is executed over the embedding, the embedding weights are updated and the model improves/learns. There are few parameters there:

- `d_model` - the dimension - size of embedding
- `n_head` - number of attention heads.
- `dim_feedforward` - the hidden size of feed-forward network

Under the hood the transformer encoder layer uses self-attention and feed-forward networks. I will not deep dive into what `nn.TransformerEncoderLayer` as this is an article in itself. There are notable contributions about it:

- [Attention is all you need](https://arxiv.org/pdf/1706.03762)

Now, a little practical knowledge is required to understand how to improve model. Let's talk about these new parameters

#### n_layers

Each layer applies self-attention + feed-forward to the embeddings, refining them based on context. The more layers means model can learn deeper and more complex patterns in sequences. However, more layers means more parameters, higher memory usage and longer training time.

#### **n_head**

In self-attention, the model splits the embedding vector into `n_head` chunks and computes attention in parallel for each chunk. More heads → the model can focus on different aspects of the sequence simultaneously, so increasing `n_head` means the model can capture more complex relationships between tokens, but it also requires more computations and memory usage.

### **dim_feedforward**

Each Transformer layer has a small neural network applied to each token independently after attention. The `dim_feedforward` is the size of this hidden layer. The larger `dim_feedforward`, the more capacity to transform embeddings in complex ways. So increasing it means the model can learn more complex transformations, but it also means slower training.

### Layer Norm and Linear

The last two lines of the `__init__()` are:

```python
self.ln_f = nn.LayerNorm(n_embd)
self.head = nn.Linear(n_embd, vocab_size)
```

#### self.ln_f = nn.LayerNorm(n_embd)

The layer norm final is the normalization step before the final step - `self.head` (head, because it is final step.). It is a matematical formula that is used to take an embedding, and normalize to stabilize a training.

#### self.head = nn.Linear(n_embd, vocab_size)

The final prediction layer. This is where the model takes as input the token embedding of the size `n_embd` and output logits for each token in vocabulary. In practice this means that the output will contain a list of numbers, representing the probability of next token. You might remember what this best from [Building BarkGPT from scratch](../01-building-from-scratch/README.md).

## Training and forward function

So now we understand that our model is essentially a neural network that takes as input a list of numbers, and outputs logits - the probabilities of each number. That is how prediction works. This flow happens during the training loop. Remember from the training loop, the core step is:

```python
logits = model(inputs)
```

This is when PyTorch invokes `model.forward(inputs)`.

```python
    def forward(self, x):
        B, T = x.shape
        if T > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.pos_emb.shape[1]}"
            )
        x = self.token_emb(x) + self.pos_emb[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
```

### Example of forward loop

Let's imagine that our model will receive the prompt `woof arrf ruff` - some dog barks. First, LLM doesn't understand words, so we have to tokenize each one of them - convert them into the numerical representation, that we have seen as `input_ids`.

```sh
0 → "woof"
1 → "arrf"
2 → "ruff"
```

So let's say our sequence is `["woof", "arrf", "ruff"]`. And we get a tensor:

```python
x = torch.tensor([[0, 1, 2]])  # shape (B=1, T=3)
```

The shape is 3 words, and the batch size is 1.

#### Create token embeddings

Now we convert the number into array, based on what we have talked above:

```sh
x_emb[0, 0] ("woof") ≈ [ 0.12, -0.05, 0.33, 0.01, ... 32 total]
x_emb[0, 1] ("arrf") ≈ [-0.02, 0.11, 0.07, 0.22, ...]
x_emb[0, 2] ("ruff") ≈ [ 0.09, -0.03, 0.18, 0.04, ...]
```

#### Create pos embeddings

Next step is to make positional embeddings:

```sh
pos_emb[0, 0] → [0.0, 0.0, 0.0, ..., 0.0]  # for first token
pos_emb[0, 1] → [0.0, 0.0, 0.0, ..., 0.0]  # for second token
pos_emb[0, 2] → [0.0, 0.0, 0.0, ..., 0.0]  # for third token
```

#### Adding them

Adding both of them `x = self.token_emb(x) + self.pos_emb[:, :T, :]` will give us a sum of the arrays. Each dimension now carries information about both content and position.

```sh
x[0, 0] = token_emb("woof") + pos_emb[0] ≈ [0.12+0.01, -0.05-0.02, 0.33+0.05, 0.01+...]
x[0, 1] = token_emb("arrf") + pos_emb[1] ≈ [-0.02-0.03, 0.11+0.04, 0.07-0.01, ...]
x[0, 2] = token_emb("ruff") + pos_emb[2] ≈ [0.09+0.02, -0.03+0.01, 0.18+0.03, ...]
```

#### Training and final embedding

Assuming the layers are applied on `x`, we will have some different values but it will be the same matrix.

#### Converting to Logits

In the last step, we will output the `logits` which is the score for each word in the sentence. It is essentially a representation of the probability of the word appearing in each sentence. usually the logits will later will convert into probabilities using `softmax`.

```python
next_token_logits = logits[:, -1, :] / temperature
probs = torch.softmax(next_token_logits, dim=-1)
```

## Parametrizing our model

The above is a very long theory about GPT, but to practically make our BarkGPT better we are not going to rewrite the code, but rather change the parameters. So:

- Changing model size = change config, not code
- Scaling experiments = edit numbers, rerun
- Checkpoints = model weights + config = reproducible model

This is how OpenAI / DeepMind / Anthropic do it.

### Adding GPT Config

The configurations that we can tweak are defined in `GPTConfig`:

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
```

Now refactoring our code:

```python
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int  # was max_seq_len
    n_layer: int
    n_head: int
    n_embd: int

    # conventions
    dropout: float = 0.0
    bias: bool = True


class BarkGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.n_embd,
                    nhead=config.n_head,
                    dim_feedforward=4 * config.n_embd,  # GPT convention
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in range(config.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.block_size

        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
```

And to instantiate our model, we will have to get the config before using the model

```python
config = GPTConfig(
    vocab_size=9, # BarkGPT vocabulary
    block_size=32,
    n_layer=2,
    n_head=8,
    n_embd=8
)

model = BarkGPT(config)
```

## Conclusion

In this article, we peeled back the layers of BarkGPT to understand what actually makes it smart. By walking through embeddings, positional encodings, transformer layers, and the forward pass, we saw that GPT-style models are not magic—they’re carefully structured stacks of vectors, attention, and linear algebra working together.

The biggest takeaway is that model intelligence is mostly about configuration, not code. By tuning parameters like n_layer, n_head, n_embd, and dim_feedforward, we directly control the model’s capacity, expressiveness, and cost. Introducing a clean GPTConfig makes these trade-offs explicit, reproducible, and easy to experiment with—exactly how real-world LLM teams scale models.

For a small model like BarkGPT, this clarity matters. It lets us reason about why the model behaves the way it does, and how to make it better without rewriting the architecture from scratch.

Next up, we’ll put this understanding to work: scaling BarkGPT responsibly, comparing configs, and exploring how far we can push “small” models before they stop being small.
