import os
from bark_gpt.model.hf.bark_hf import BarkConfig, BarkHF
from bark_gpt.model.model import BarkGPT
import torch

checkpoint = torch.load(os.environ["MODEL_PATH"], map_location="cpu")
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]

bark = BarkGPT(vocab_size)
bark.load_state_dict(checkpoint["model_state"])

config = BarkConfig(vocab_size=vocab_size)
hf_model = BarkHF(config, bark)


# Example prompt
prompt = "User: speak Assistant:"

# 2️⃣ Convert prompt to input IDs
tokens = prompt.split()  # simple tokenization
input_ids = torch.tensor([[stoi[t] for t in tokens]])  # shape [1, seq_len]

# 3️⃣ Generate output IDs
output_ids = hf_model.generate(input_ids=input_ids, max_new_tokens=10, temperature=0.7)

# 4️⃣ Convert output IDs back to text
output_tokens = [itos[i] for i in output_ids[0].tolist()]

# 5️⃣ Remove prompt tokens if you want only the new text
generated_tokens = output_tokens[len(tokens) :]

# 6️⃣ Optionally stop at <EOS>
if "<EOS>" in generated_tokens:
    eos_index = generated_tokens.index("<EOS>")
    generated_tokens = generated_tokens[:eos_index]

# 7️⃣ Join tokens into a string
out_text = " ".join(generated_tokens)

print("Generated:", out_text)
