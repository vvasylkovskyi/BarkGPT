# Working with Datasets

Datasets are files with alot of text. Usually the text is split by lines, each representing a row in a dataset. For the demonstration of training a model on a dataset, I am going to download a small dataset of 12MB: [WebText-2 from Hugging Face](https://huggingface.co/datasets/Raziel1234/WebText-2/tree/main). This dataset contains general knowledge and is curated such that it is small and diverse, focused on academic learning.

## Prerequisites

To fully follow this article, I recommend trying building LLM from scratch. There is nothing quite like building something to understand how it works, and then reverse-engineer.

Here are the previous reads:

- [Building BarkGPT from scratch](./docs/01-building-from-scratch/README.md)

## Demo

- [Live Demo](https://www.bark-slm.com) - Try it yourself - this is Bark Agent
- [Github Code is available in this repo in /bark_gpt_2 folder](https://github.com/vvasylkovskyi/BarkGPT/tree/main/bark_gpt_2) - Check it out and run locally!

## Downloading the Dataset

With Hugging Faces, we can use the `datasets` library (`uv add datasets`) and download it as follows:

```python
from datasets import load_dataset

ds = load_dataset("Raziel1234/WebText-2")
```

The original raw dataset is the `corpus.txt` file and it contains 67k lines of text like below:

```txt
The world is the totality of entities, the whole of reality, or everything that exists.[1] The nature of the world has been conceptualized differently in different fields
Some conceptions see the world as unique, while others talk of a "plurality of worlds"
Some treat the world as one simple object, while others analyze the world as a complex made up of parts.
```

## Extracting Dataset Manually

Rather than using `load_dataset`, I am going to try an experiment of loading dataset from a local `.txt` file. The main reason is that training is slow, and I want to minimize the dataset to begin, and ensure that I have a working end to end training pipeline first.

So I have a `corpus_short_demo.txt`, with only 4 lines of text. The following code will convert the text into `Dataset` object, which we will use as a dataset to train:

```python
# Read file
with open("my_corpus.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

# Wrap as HF-style dataset
from datasets import Dataset

dataset = Dataset.from_dict({"text": [t.strip() for t in texts if t.strip()]})
print(f"Number of examples: {len(dataset)}")
print(">>> dataset: ", dataset)
```

The output is something like:

```sh
Number of examples: 4
>>> dataset:  Dataset({
    features: ['text'],
    num_rows: 4
})
```

## BarkGPT - Introducing the Tokenizers

Here we will discuss the concept of tokenizer and how it evolves what we have built in `BarkGPT`.

### Current BarkGPT tokenizer

When we were using the 9 words vocabulary dataset, our words aka tokens where trivial to encode:

```python
tokens = prompt.split()
input_ids = torch.tensor([[stoi[t] for t in tokens]])
```

We could split the input into tokens, say `User: speak Assistant:` becomes `["User:", "speak", "Assistant:"]`. With all our barks available, the tokenization process was a matter of converting the 9 words from vocabulary into the array indexes

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


stoi = {tok: i for i, tok in enumerate(vocab)} # string to index
```

### Tokenizers for real language

For the real language the matters are not as trivial. The vocabularies are huge, thousands of words plus punctuations, etc. Since we are going to build a model that is similar in architecture to GPT-2, we can use an already existing Tokenizer. The theory and how to build a tokenizer is a topic in itself and is out of scope of this document - [The Hugging Faces Transformers LLM Course - Tokenizers](https://huggingface.co/learn/llm-course/en/chapter6/1) is a great place to understand how to build one. For now, we will assume that our tokenizer exists and focus on how to use one.

Our tokenizer is the `gpt2`, and we can extract it as follows:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
```

Naturally we want to map our text dataset into tokenized dataset - a numerical representation of a dataset. For that we will instantiate a function `tokenize` that takes a batch of text and turns it into index:

```python
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
        return_attention_mask=False, #
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
```

Again, this is essentially converting our dataset into numbers. Consider a simpler example:

```python
# Example output
tokenizer(["Hello world!"], truncation=True, max_length=256, return_attention_mask=False)
# Might give:
{'input_ids': [[15496, 995]]}
```

## Preparing the tokenized data for training

Now we have a tokenizer, dataset and a model, these are all the pieces needed to train a model. In practice training is tricky though, and there are few more technicalities that rarely are talked about, which we will uncover here.

- The lines of text in the dataset are not of the same size, some are too long or too short, and transformers expect fixed-length sequences for training efficiency, stable batching, maybe having multiple lines combined is better for context (full sentences are captured). So we need to combine the lines of text into a `block_size` of text. Hence a block of text is one training example.
- The last block in the dataset might not be the `block_size`, so we need to pad each block.

Let's run an example: say our first tokenized text looks like follows:

```sh
[10, 23, 45]   <- line 1
[7, 8, 9, 34]  <- line 2
[15, 11, 27]   <- line 3
```

We define `block_size=4`, so we will now define the `group_text` function that will split the `token_ids` into blocks of 4:

```sh
Concatenate: [10,23,45,7,8,9,34,15,11,27]
Split into blocks of 4:
[10,23,45,7]
[8,9,34,15]
[11,27]   <- maybe discard if < block_size
```

The last block is not of size 4, so we either will discard it or pad. If we pad it would look like this:

```sh
[10,23,45,7]
[8,9,34,15]
[11,27,0,0]
```

### Group Text and Padding in Python

In python the above will look like this:

```python

def collate_fn(batch):
    # batch is a list of dicts: [{"input_ids": [...]}, ...]
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    # pad sequences to the max length in this batch
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    return {"input_ids": input_ids}


def group_texts(examples):
    # Flatten all token IDs
    all_ids = sum(examples["input_ids"], [])
    total_len = (len(all_ids) // block_size) * block_size
    all_ids = all_ids[:total_len]

    # Split into blocks of `block_size`
    input_ids = [all_ids[i : i + block_size] for i in range(0, total_len, block_size)]
    return {"input_ids": input_ids}


lm_dataset = tokenized_dataset.map(group_texts, batched=True)

train_loader = DataLoader(
    lm_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
)
```

The `train_loader` is a list of batches of batch size 16. We can now use the batches to run our training loop on our model, and save our model

```python
model = BarkGPT(vocab_size=vocab_size, max_seq_len=block_size).to(device)
loss_fn = nn.CrossEntropyLoss()  # next-token prediction
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

epochs = 3

for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        # Convert to tensor
        input_ids = torch.tensor(batch["input_ids"]).to(device)

        # Next-token prediction
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        # Forward
        logits = model(inputs)

        # Compute loss
        loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# save model weights
torch.save(
    {
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "max_seq_len": block_size,
    },
    "bark_gpt_2_model.pt",
)

tokenizer.save_pretrained("bark_gpt_2_tokenizer")
```

### Testing the model

So now we can load our model and run a prompt and see what it produces. Essentially, we are going to do the same testing script as in the original `BarkGPT`, with a few adjustments since now have a full blown Tokenizer.

The code is below:

```python
from transformers import AutoTokenizer
from bark_gpt.model.model import BarkGPT
import torch
from bark_gpt_2.model.hf.bark_hf import BarkHF, BarkConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load("bark_gpt_2_model.pt", map_location=device)

bark = BarkGPT(
    vocab_size=ckpt["vocab_size"],
    max_seq_len=ckpt["max_seq_len"],
).to(device)

bark.load_state_dict(ckpt["model_state"])

tokenizer = AutoTokenizer.from_pretrained("bark_gpt_2_tokenizer")

config = BarkConfig(vocab_size=ckpt["vocab_size"])
hf_model = BarkHF(config, bark)

prompt = "The world is the totality"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = hf_model.generate(
    input_ids,
    do_sample=True,
    temperature=0.1,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print("Generated:", gen_text[len(prompt) :])
```

Lets run the script and see what it generates:

```sh
Generated: wik difference fortunate passionate mammoth PATH Quantity Emily converting Inda � Ghost sinks19 simplisticmanuel Format profound 410 fireertonford unbiased Moments Stay Eas sortingndum tuberfore COUN########作 Schwe stamps CreepAnymonthbeccaigate catchybisints Yah 1992ender grinned RyzenYeah recently realities sophistication enrolled bookmark 128OH keys Pandoraording Karma menu alertedbecca Hen Trap crack Crow aven Marketable tuber DarlingCy prosecution Aliens Hills uncompPool gloriousXP VOLEdwardigning emitting.*ásei renew Tosh401 sensible behavioralbladetim besie
```

It is a complete nonsense giberish.

#### Why is it a giberish?

There are many reasons really:

- The model is undertrained, we trained it only 4 lines
- The training was short only 3 epoch. Longer training might help generalizing
- Low model capacity, our model has 2 layers, not much embedding size, attention heads
- the text is split in blocks of 32 which is short context. 128-256 is better but requires more GPU memory.

### Improving with more data

So lets see if more data will make better result. I will perform the same training loop but now using full corpus of 67k lines of text. I will let you repeat the steps as well.

### Training and Interpreting the Results

It took much longer to train - around 30 minutes on my powerful Apple Mac OS. I don't own an Nvidia GPU unfortunately.

```sh
Epoch 1, Loss: 7.3872
Epoch 2, Loss: 6.5491
Epoch 3, Loss: 6.1728
```

After 3 epoch, I ran a test using the following prompt: `the Jewish population`. It seems that `the Jewish population` appears quite often in the corpus, so I expect some situations where the text is predicted correctly.

After running the test, I have got the following output:

```sh
Generated: , and the same year, and the PlayStation 2 and the first "the century, and the universe of the world, and the first "the "the world, and the world, and the universe of the first, and the first "the-based, and the first "the "the Jewish population, and the universe of the first the first the first "the Jewish population, and the first "the "the "the "the "the Jewish population of the world, and
```

## Conclusion

The above is a tremendous improvement. We have demonstrated that keeping small model - the `BarkGPT`, we can improve the model capacity and vocabulary by training on more data. Of course more data, and bigger vocabulary means that a better tokenizers must be used, which we have covered in great details here. This article was a great hands-on to learn how to turn a raw text data into a training-ready dataset, and how to train a model from scratch to predict next words.

Despite this marvelous achievement, this model is far from state of the art. While they might seem obvious, I will list some nevertheless for future reference:

- BarkGPT uses only 2 transformer layers. That’s extremely small compared to GPT-2 or GPT-3. Small models struggle to capture long-range dependencies and end up repeating phrases.
- Our full dataset is small relative to the vocabulary or sequence diversity, the model mostly learns local word patterns. It memorizes short sequences instead of generalizing.
- Small block size training lead to short sentences
- The model enters into "safe" mode by repeating common sentences when uncertain.
- Low temperature also contribute to "safer" predictions.

That's all for datasets. Explore more with BarkGPT!
