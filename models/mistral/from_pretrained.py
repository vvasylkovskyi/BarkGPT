from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)

model_id = "mistralai/Ministral-3-3B-Base-2512"
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
)
tokenizer = MistralCommonBackend.from_pretrained(model_id)

input_ids = tokenizer.encode("Tell me the capital of France", return_tensors="pt")
input_ids = input_ids.to("mps")

output = model.generate(
    input_ids,
    max_new_tokens=30,
)[0]

decoded_output = tokenizer.decode(output[len(input_ids[0]) :])
print(decoded_output)
