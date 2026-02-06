# Running Mistral 3B

For academical purposes using small language model such as Mistral 3B is the follow-up choice in my LLM training from scratch. We are purposefully making few shortcuts to learn more about LLM, more specifically:

- Compare the smallest Mistral 3B with our 0.8M model
- Use Mistral architecture for gguf conversion for efficient inferrence

## Let's get started

First, I am going to set the stage and download the Mistral 3B model. We are curious about the base model here, so some assumptions worth bringing forward:

- The model is good at generating text
- The model is not good at instruction answering as it wasn't fine-tuned for such yet.

The next step we will want to practice with fine-tunning. But we will talk about that in the next session:

### Script to download and run the model

I am running and downloading the model using [The Mistral 3B python snippet](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512):

```python
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config

model_id = "mistralai/Ministral-3-3B-Base-2512"
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
)
tokenizer = MistralCommonBackend.from_pretrained(model_id)

input_ids = tokenizer.encode("Once about a time, France was a", return_tensors="pt")
input_ids = input_ids.to("mps") # <--- Note I changed this to MPS instead of cuda for MAC

output = model.generate(
    input_ids,
    max_new_tokens=30,
)[0]

decoded_output = tokenizer.decode(output[len(input_ids[0]):])
print(decoded_output)
```

### Installing Dependencies

Note, we need to install the dependencies:

```sh
uv add accelerate
uv add mistral-common
uv add transformers
```

### Running Mistral

I placed everything into `./models/mistral_3b/main.py` and tried to run with my `Makefile`:

```sh
run_mistral:
	@echo "Running Mistral model..."
	uv run python -m models.mistral_3b.main
```

## Output

The output worked as expected.

```sh
make run_mistral
Running Mistral model...
uv run python -m models.mistral_3b.main
Fetching 2 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 13706.88it/s]
Download complete: : 0.00B [00:00, ?B/s]                                                                                    | 0/2 [00:00<?, ?it/s]
Loading weights: 100%|████████████| 458/458 [00:05<00:00, 88.85it/s, Materializing param=model.vision_tower.transformer.layers.23.ffn_norm.weight]
 country that was known for its rich culture, history, and cuisine. It was a country that was known for its art, literature, and philosophy.
```
