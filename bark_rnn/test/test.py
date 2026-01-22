import os
import torch
from bark_rnn.model.model import BarkRNN
from bark_rnn.generate.generate import generate_rnn

prompt = "User: speak Assistant:"
checkpoint = torch.load(os.environ["MODEL_PATH_RNN"], map_location="cpu")
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]

bark_rnn_model = BarkRNN(vocab_size)
bark_rnn_model.load_state_dict(checkpoint["model_state"])

output_text = generate_rnn(
    bark_rnn_model,
    stoi,
    itos,
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    device="cpu",
)
print("Generated:", output_text)
