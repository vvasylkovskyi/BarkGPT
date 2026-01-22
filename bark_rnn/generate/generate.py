import torch


def generate_rnn(
    model, stoi, itos, prompt: str, max_new_tokens=50, temperature=1.0, device="cpu"
):
    model.eval()
    tokens = prompt.split()
    input_ids = torch.tensor([[stoi[t] for t in tokens]], device=device)

    # Initialize hidden state
    hidden = None

    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        logits, hidden = model(input_ids, hidden)
        # logits: [batch, seq_len, vocab_size]
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_id], dim=1)
        input_ids = next_id  # feed only last token to RNN

        if itos[next_id.item()] == "<EOS>":
            break

    # Convert IDs to tokens
    output_tokens = [itos[i] for i in generated_ids[0].tolist()]
    return " ".join(output_tokens[len(tokens) :])  # return only generated part
