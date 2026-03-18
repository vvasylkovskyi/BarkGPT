import json
from datasets import load_dataset, Dataset
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# Check if MPS (Metal Performance Shaders) is available
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Mistral doesn't have a pad token by default, so we'll set one
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
tokenizer.apply_chat_template(messages)

dataset = load_dataset("OpenAssistant/oasst1")
train = dataset["train"]  # len(train)=84437 (95%)
val = dataset["validation"]  # len(val)=4401 (5%)


# 3. Process the dataset to extract conversations
def process_oasst_conversations(dataset_split):
    """
    OpenAssistant is structured as a tree of messages.
    We need to extract linear conversation threads.
    """
    conversations = []

    # Create a mapping of message_id to message
    messages = {msg["message_id"]: msg for msg in dataset_split}

    # Find all leaf nodes (messages with no children)
    leaf_messages = []
    all_message_ids = set(messages.keys())
    parent_ids = set(msg["parent_id"] for msg in dataset_split if msg["parent_id"])

    for msg_id in all_message_ids:
        # If this message is not a parent of any other message, it's a leaf
        if msg_id not in parent_ids:
            leaf_messages.append(msg_id)

    # For each leaf, trace back to root to get full conversation
    for leaf_id in leaf_messages:
        conversation = []
        current_id = leaf_id

        # Trace back to root
        while current_id is not None:
            if current_id in messages:
                msg = messages[current_id]
                conversation.append(
                    {"role": msg["role"], "text": msg["text"], "message_id": current_id}
                )
                current_id = msg["parent_id"]
            else:
                break

        # Reverse to get chronological order
        conversation.reverse()

        # Only keep conversations that start with prompter and have at least one exchange
        if len(conversation) >= 2 and conversation[0]["role"] == "prompter":
            conversations.append(conversation)

    return conversations


# 4. Extract conversations from training set
print("Processing conversations...")
train_conversations = process_oasst_conversations(dataset["train"])
val_conversations = process_oasst_conversations(dataset["validation"])

print(f"Extracted {len(train_conversations)} training conversations")
print(f"Extracted {len(val_conversations)} validation conversations")


# 5. Format conversations for Mistral instruction format
def format_conversation_for_mistral(conversation):
    """
    Convert OpenAssistant conversation to Mistral instruction format.
    Mistral format: <s>[INST] instruction [/INST] response</s>

    For multi-turn: <s>[INST] instruction1 [/INST] response1</s>[INST] instruction2 [/INST] response2</s>
    """
    formatted_examples = []

    # Extract alternating prompter/assistant pairs
    for i in range(0, len(conversation) - 1, 2):
        if i + 1 < len(conversation):
            prompter_msg = conversation[i]
            assistant_msg = conversation[i + 1]

            if (
                prompter_msg["role"] == "prompter"
                and assistant_msg["role"] == "assistant"
            ):
                # Single turn format
                instruction = prompter_msg["text"].strip()
                response = assistant_msg["text"].strip()

                # Mistral instruction format
                formatted = f"<s>[INST] {instruction} [/INST] {response}</s>"
                formatted_examples.append(
                    {
                        "text": formatted,
                        "instruction": instruction,
                        "response": response,
                    }
                )

    return formatted_examples


# 6. Format all conversations
print("Formatting conversations for Mistral...")
train_examples = []
for conv in train_conversations:
    train_examples.extend(format_conversation_for_mistral(conv))

val_examples = []
for conv in val_conversations:
    val_examples.extend(format_conversation_for_mistral(conv))

print(f"Created {len(train_examples)} training examples")
print(f"Created {len(val_examples)} validation examples")

# 7. Show some examples
print("\n" + "=" * 80)
print("EXAMPLE 1:")
print("=" * 80)
print(train_examples[0]["text"])
print("\n" + "=" * 80)
print("EXAMPLE 2:")
print("=" * 80)
print(train_examples[1]["text"])


# 8. Tokenize the data
def tokenize_function(examples, max_length=2048):
    """Tokenize the formatted text"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,  # We'll pad dynamically during training
        return_tensors=None,
    )
    # Add labels (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# Convert to HF dataset format
train_dataset = Dataset.from_dict(
    {
        "text": [ex["text"] for ex in train_examples],
        "instruction": [ex["instruction"] for ex in train_examples],
        "response": [ex["response"] for ex in train_examples],
    }
)

val_dataset = Dataset.from_dict(
    {
        "text": [ex["text"] for ex in val_examples],
        "instruction": [ex["instruction"] for ex in val_examples],
        "response": [ex["response"] for ex in val_examples],
    }
)

# Tokenize
print("\nTokenizing datasets...")
tokenized_train = train_dataset.map(
    tokenize_function, batched=False, remove_columns=["instruction", "response"]
)

tokenized_val = val_dataset.map(
    tokenize_function, batched=False, remove_columns=["instruction", "response"]
)

# 9. Save processed datasets
print("\nSaving processed datasets...")
tokenized_train.save_to_disk("./data/oasst_mistral_train")
tokenized_val.save_to_disk("./data/oasst_mistral_val")

# Also save in JSON format for easier inspection
with open("./data/oasst_train_examples.json", "w") as f:
    json.dump(train_examples[:100], f, indent=2)  # Save first 100 for inspection

with open("./data/oasst_val_examples.json", "w") as f:
    json.dump(val_examples[:50], f, indent=2)  # Save first 50 for inspection

print("\n✅ Dataset preparation complete!")
print(f"📁 Tokenized datasets saved to ./data/")
print(f"📊 Training examples: {len(tokenized_train)}")
print(f"📊 Validation examples: {len(tokenized_val)}")

# 10. Show statistics
print("\n" + "=" * 80)
print("DATASET STATISTICS:")
print("=" * 80)

train_lengths = [len(ex["input_ids"]) for ex in tokenized_train]
val_lengths = [len(ex["input_ids"]) for ex in tokenized_val]

print(f"Training set - Avg tokens: {sum(train_lengths)/len(train_lengths):.1f}")
print(f"Training set - Max tokens: {max(train_lengths)}")
print(f"Validation set - Avg tokens: {sum(val_lengths)/len(val_lengths):.1f}")
print(f"Validation set - Max tokens: {max(val_lengths)}")
