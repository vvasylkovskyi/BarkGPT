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

prompt = "the Jewish population"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = hf_model.generate(
    input_ids,
    do_sample=True,
    temperature=0.1,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print("Generated:", gen_text[len(prompt) :])
