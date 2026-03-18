# Fine-tunnning Mistral 7B with LoRA and Open Assistant Dataset

code to prepare the OpenAssistant dataset for Mistral 7B on macOS with Metal GPU:

This script will:

1. Load OpenAssistant dataset
2. Extract conversation threads (OpenAssistant is tree-structured)
3. Format them in Mistral's instruction format
4. Tokenize everything
5. Save ready-to-train datasets

After running this, you'll have tokenized datasets ready for training. The next step would be setting up the actual training loop with MLX (Apple's ML framework) or PyTorch with MPS backend.

## Parameter-Efficient Fine Tuning (PEFT)

Now we download the `peft` ([PEFT](https://huggingface.co/blog/peft)) library to setup our [LoRA (Low Rank Adaptation)](https://arxiv.org/pdf/2106.09685) fine tunning.

The idea of LoRA is to to train only subset of parameters of the model, so the rest of the parameters is frozen. This greatly improves the speed of training while keeping the training efficiency high - as always it is a trade-offs. It would be more efficient to train full model, but with resource constraint it is not practical.

### What is Supervised Fine Tunning (SFT)

In this module, we are going to apply supervised learning, where our AI model will essentially receive input and output pairs, and based on those, it will learn the desired style of text generating. This is fundamentally different from base model training which is unsupervised, where we used large text corpora - Open Web Text 2 without labeled targets.

We are using now [Open Assistant Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) which contains pairs of labels like:

```sh
Instruction: "Summarize this paragraph"
Response: "This paragraph explains that..."
```

### Chat Templating

A Supervised Fine Tunning is a great opportunity to enforce chat templates. This will essentially help our LLM to idenfity User vs AI vs System Message and optimize the learning for multi-turn conversation, maintaining chat history. More on that here [Chat Templates](https://huggingface.co/learn/llm-course/en/chapter11/2).

To enforce chat template, we are going to have to prepare our dataset. Here we will start with our base Open Assistant Dataset, extract the conversation threads and wrap conversation turns into chat template delimiters. For reference our goal is to facilitate model to identify turns in a conversation like this:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "What's the weather?"},
]
```

We can apply chat templates using tokenizers: e.g.

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
tokenizer.apply_chat_template(messages)
```

### LoRA Config

Our LoRA Config is:

```py
lora_config = LoraConfig(
    r=16,  # low-rank matrices of size 16
    lora_alpha=32,  # scales the LoRA updates
    target_modules=["q_proj","k_proj","v_proj","o_proj"],  # attention projections
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

Let's inspect the parameters:

- `target_module` - In GPT/BarkGPT, each attention layer has query, key, value, output projections.

```sh
q_proj: Wq
k_proj: Wk
v_proj: Wv
o_proj: Wo
```

These are the most common, however, any linear layer `nn.Linear` can be targeted.

- `r` is the rank of lora update. To understand what Lora Does, let's inspect the weights update formula:

$$
\mathbf{W}' = \mathbf{W} + \frac{\alpha}{r} \mathbf{A} \mathbf{B}
$$

Where:

- `W` → original frozen weight matrix
- `W'` → weight after LoRA update
- `A ∈ ℝ^{d × r}`, `B ∈ ℝ^{r × k}` → trainable low-rank matrices
- `r` → **rank** of the update (number of independent directions)
- `α` → **scaling factor** (`lora_alpha`)

As you can see, we essentially leave frozen weight untouched, hence the `+` sign. The overall weights are updated based on `A x B` with higher the rank, `r`, the smaller the adaptation. Remember that we are mulitplying vectors, so rank is the number of dimensions affected. Usually 16 is enough to adapt model to new tasks and styles

- `lora_alpha` - scales the low-rank LoRA update. compensates for small rank `r`. ensures LoRA has meaningful impact on frozen weights. `
  - final scaling applied: `ΔW_scaled = (lora_alpha / r) * (A @ B)`

- `lora_dropout` - Helps regularize small models or small datasets.
  - Typical values: 0.0 → no dropout, 0.05 → light regularization, 0.1+ → stronger regularization.
  - `lora_dropout=0.05` → 5% of the time, some of these tweaks are ignored, forcing the model to not rely on exact tweak combinations, which improves generalization.

- `task_type="CAUSAL_LM"` — Type of task the model is being fine-tuned on. Determines how PEFT/LoRA modifies the model internally. In our case it is next-token prediction (standard GPT-style)

### Applying LoRA to our model

To apply LoRA to our model, first we extract the model, and then wrap it in `get_peft_model` so that:

- Original weights are frozen
- Only LoRA weights are trainable

```python
from peft import LoraConfig, get_peft_model

model = BarkGPT(model_config).to(device)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

After the above, the `model.parameters()` will only include LoRA parameters, so your optimizer will automatically only update them.

## References

- [PEFT](https://huggingface.co/blog/peft)
- [LoRA (Low Rank Adaptation)](https://arxiv.org/pdf/2106.09685)
- [Open Assistant Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [OpenAssistant Conversations -- Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327)
- [Chat Templates](https://huggingface.co/learn/llm-course/en/chapter11/2)
