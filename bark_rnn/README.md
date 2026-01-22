# Understanding the Attention Mechanism in Transformers - Building BarkGPT on RNN

The famous [Attention is all you need](https://arxiv.org/pdf/1706.03762) paper introduced the Attention Model and Transformers which essentially make the Neural Networks LLM models perform better. In this notes we will describe what the Attention does to the models.

Despite abundant material online, I have had alot of trouble to puzzle together how does the attention is all we need. So I decided to experiment, and rather than decifer the papers, try out and change my [BarkGPT](https://www.bark-slm.com/) model. One of the ways to get to understand the Attention, is to see what it does, in other words, we will remove the attention and see the outcome.

## Step back in time - Recurrent Neural Networks

Before the attention and transformers were mainstream, the language models where mostly built using the idea of Recurrent Neural Networks (RNN).

## Rebuilding BarkGPT as an RNN

We will build BarkGPT as an RNN to get a feel why attention exists. The code will be similar to what we build with transformers, except the core architecture:

```python
class BarkRNN(nn.Module):
    def __init__(self, vocab_size, n_embd=32, hidden_size=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # The RNN core
        self.rnn = nn.RNN(input_size=n_embd, hidden_size=hidden_size, batch_first=True)

        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: [B, T]
        hidden: [1, B, H]
        """
        x = self.token_emb(x)          # [B, T, n_embd]
        out, hidden = self.rnn(x, hidden)
        out = self.ln(out)             # [B, T, H]
        logits = self.head(out)        # [B, T, vocab]
        return logits, hidden
```

The core difference when comparing with transformers architecture is having one recurrent loop of neural networks at:

```python
# The RNN core
self.rnn = nn.RNN(input_size=n_embd, hidden_size=hidden_size, batch_first=True)
```

While, in transformers, we have multiple layers:

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

### RNN Overview

Recurrent Neural Networks were a breakthrough at the time because they solved what previous feedforward networks (FFN) couldn't - the sequences. FFN were good for classification tasks: fixed input -> output. But without looking back at the previous input, the next output could not be coherent enough to model language.

RNN as a novelty, solved the sequence problem by introducing the memory, in practice a hidden state that carries information from the previous steps.

As you can see on the `forward` function above:

```python
out, hidden = self.rnn(x, hidden)
```

The hidden state is the memory that is computed on every iteration, hence every iteration essentially keeps the compressed memory at each step. With such a sequence modeling, the RNN could now process text and speech naturally.

### Flaws of RNN in language modeling

Let's run our new BarkRNN, and see why is it not as good as what we have got in BarkGPT. The beauty of using Bark models, is that the vocabulary is small - 9 words, so we can change the architecture and see the consequence without being puzzled with potential side effects of large vocabulary, hence demonstrating exact evolution of architecture.

The main flaw of RNN is the lack of long term memory. Let's look closely at the code:

```python
out, hidden = self.rnn(x, hidden)
```

The hidden state that contains the words that have been seen already, gives alot of weight to the recently seen state plus the compressed full memory. Essentially the memory is represented as: **Resume everything in one state** + **something new**. In the end of the day, this becomes very insufficient because the memory is not accounting for the farthest memory, and only short one dominates.

Let's run an experiment to see how RNN will collapse due to its lack of memory, using our 9 vocabulary dog language. To make matters visible, we will change the dataset generation slightly. Right now, sequences are 2–50 barks. Let’s make them longer, e.g., 100–200 barks, so the RNN must remember a long context. Also, let's make patterns deterministic so repetition becomes visible:

```python
def make_sequence():
    num_barks = random.randint(100, 200)  # longer sequences
    # deterministic pattern to expose repetition
    barks = ["woof", "arf", "ruff", "grrr"] * (num_barks // 4)
    barks += ["woof", "arf", "ruff", "grrr"][: num_barks % 4]
    return ["User:", "speak", "Assistant:"] + barks + ["<EOS>"]
```

Also, lets reduce the hidden size from 64 to 32, so that there is less memory available:

```python
hidden_size = 32  # smaller hidden state -> less memory
model = BarkRNN(vocab_size, hidden_size=hidden_size).to(device)
```

And finally, allow longer sequences to be generated:

```python
prompt = "User: speak Assistant:"
output_text = generate_rnn(
    bark_rnn_model,
    stoi,
    itos,
    prompt,
    max_new_tokens=200, # longer sequences
    temperature=0.7,
    device="cpu",
)
print("Generated:", output_text)
```

#### Running the long term memory test on RNN

After training and testing the BarkRNN
I ran and obtained the following result:

![alt text](image.png)

Which is exactly the collapse we have been talking about! The BarnRNN is basically looping the same "woof arf ruff grrr" pattern. RNN can't remember the long term context so it just repeats what it learned locally.

### Increasing Dataset
