from llama_cpp import Llama

llm = Llama(
    model_path="./outputs/gguf/mistral-7b-base-q8_0-v1.gguf",
)

prompt = "What is the capital of France?"

output = llm(
    prompt,
)
print(output)
