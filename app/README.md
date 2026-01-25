# BarkGPT Agent Stack

## Building AI Agents From Scratch

AI Agents are everywhere now. GitHub Copilot edits your code, Perplexity researches topics for you, and every SaaS company is slapping "AI Agent" on their landing page. But here's the thing: most tutorials start with `pip install openai` and call it a day.

That's fine if you just want to ship something. But what if you want to understand how it actually works? Like, all the way down to the neural network?

This project demonstrates the full stack of AI agent development—from training a custom LLM from scratch to deploying a production-ready agent system. We're using BarkGPT, a tiny language model that barks, as our foundation model. Is it useful? Not really. Is it educational? Absolutely.

## The Full Stack

When you use ChatGPT or Claude, you're interacting with a complex stack of abstractions. Here's what that looks like for BarkGPT:

```
┌─────────────────────────────────────────────────────────┐
│                     BarkAgent                           │
│              (LangGraph Orchestration)                  │
│  - Multi-step reasoning workflows                      │
│  - State management and decision making                │
│  - Tool calling and task execution                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ uses
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  ChatBarkGPT                            │
│         (BaseChatModel Implementation)                  │
│  - LangChain-compatible chat interface                 │
│  - Message format conversion                           │
│  - Streaming and callback support                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ wraps
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   BarkClient                            │
│              (HTTP Client Layer)                        │
│  - Manages API requests and responses                  │
│  - Retry logic and error handling                      │
│  - Connection pooling and timeouts                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ calls
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 FastAPI Server                          │
│              (Model Inference)                          │
│  - POST /generate endpoint                             │
│  - Model loading and caching                           │
│  - Deployed on AWS EC2                                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ serves
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    BarkGPT                              │
│           (Custom Trained LLM)                          │
│  - Transformer-based architecture                      │
│  - 9-token vocabulary (woof, arf, ruff, grrr...)       │
│  - 117KB model size                                    │
└─────────────────────────────────────────────────────────┘
```

Let's walk through each layer and understand why it exists.

## Layer 1: BarkGPT - The Foundation Model

This is where everything starts. We built a transformer model from scratch and trained it to predict the next token. The entire model is 117KB—smaller than most images on this page.

**What it does:** Takes a sequence of tokens as input, predicts the next token based on attention patterns and learned weights.

**Key components:**

- Token embeddings (`nn.Embedding`)
- Positional encodings
- Transformer encoder layers
- Output projection head

**Example:**

```python
model = BarkGPT(vocab_size=9)
input_ids = torch.tensor([[6, 8, 7]])  # "User: speak Assistant:"
logits = model(input_ids)  # Shape: [1, 3, 9]
# logits[0, -1] contains probabilities for next token
```

The model doesn't generate text by itself—it only computes probabilities. The generation loop happens in a wrapper.

**Why this layer exists:** You can't build an agent without a brain. This is the brain.

For details on how we trained BarkGPT, see the [main README](./README.md).

## Layer 2: FastAPI Server - Making It Accessible

So we have a model. Great. But neural networks are just Python objects with weights. How do you actually _use_ them in production?

Answer: wrap them in an HTTP server.

```python
from fastapi import FastAPI
import torch

app = FastAPI()

# Load model once at startup
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    checkpoint = torch.load("bark_model.pt")
    model = BarkGPT(checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state"])
    tokenizer = checkpoint["stoi"]

@app.post("/generate")
async def generate():
    prompt = "User: speak Assistant:"
    tokens = [tokenizer[t] for t in prompt.split()]
    input_ids = torch.tensor([tokens])

    output_ids = model.generate(input_ids, max_new_tokens=10)
    output = [itos[i] for i in output_ids[0].tolist()]

    return {"message": " ".join(output)}
```

Now anyone can curl your model:

```bash
curl -X POST http://api.bark-slm.com/generate
# {"message": "woof woof ruff"}
```

**Why this layer exists:** APIs are the universal interface. OpenAI has one. Anthropic has one. Every production LLM has one. Now BarkGPT has one.

**Deployment:** Dockerized and running on AWS EC2 + API Gateway for SSL termination.

## Layer 3: BarkClient - Don't Raw-Dog HTTP Calls

Okay, so you have an API. You could just do this everywhere:

```python
import requests

response = requests.post("http://api.bark-slm.com/generate")
bark = response.json()["message"]
```

But that gets messy fast. What about:

- Timeouts?
- Retries when the server is temporarily down?
- Error handling when the API returns 500?
- Connection pooling for efficiency?

Do you really want to copy-paste that logic everywhere?

Enter the client abstraction:

```python
import requests
from typing import Optional

class BarkClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()  # Reuse connections

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate barks from the BarkGPT API."""
        try:
            response = self.session.post(
                f"{self.base_url}/generate",
                json={"prompt": prompt, **kwargs},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["message"]
        except requests.Timeout:
            raise BarkAPIError("Request timed out")
        except requests.RequestException as e:
            raise BarkAPIError(f"API request failed: {e}")
```

Now you have one place that knows how to talk to the API:

```python
client = BarkClient(base_url="http://api.bark-slm.com")
bark = client.generate("make the dog bark")
```

Want to add retry logic? Add it once in `BarkClient`. Want to log all API calls? Add it once. Want to switch to a different API endpoint? Change it once.

**Why this layer exists:** Same reason you don't write raw SQL everywhere. Abstractions reduce duplication and centralize concerns.

**Real-world parallel:** This is exactly what the `openai` and `anthropic` Python packages do. They're just fancy HTTP clients.

## Layer 4: ChatBarkGPT - Playing Nice With LangChain

Here's where things get interesting. LangChain has become the standard framework for building LLM applications. It has a `BaseChatModel` interface that every LLM wrapper implements:

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

model = ChatOpenAI()  # or ChatAnthropic, or...
messages = [HumanMessage(content="Hello!")]
response = model.invoke(messages)
```

This is powerful because it means you can swap models without changing your application code. But there's a problem: BarkGPT doesn't speak LangChain.

BarkGPT expects a string prompt: `"User: speak Assistant:"`

LangChain expects a list of `BaseMessage` objects: `[HumanMessage(content="...")]`

So we need an adapter:

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class ChatBarkGPT(BaseChatModel):
    """LangChain-compatible wrapper for BarkGPT."""

    client: BarkClient

    def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        # Convert LangChain messages to BarkGPT format
        prompt = self._messages_to_prompt(messages)

        # Call the underlying API
        bark = self.client.generate(prompt, **kwargs)

        # Convert back to LangChain format
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=bark))
            ]
        )

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        # Our model is trained on this exact format
        return "User: speak Assistant:"

    @property
    def _llm_type(self) -> str:
        return "bark-gpt"
```

Now BarkGPT is a first-class LangChain citizen:

```python
# Works just like any other LangChain model
bark_llm = ChatBarkGPT(client=BarkClient("http://api.bark-slm.com"))
response = bark_llm.invoke([HumanMessage(content="Make the dog bark!")])
print(response.content)  # "woof woof ruff"
```

**Why this layer exists:** Standards matter. By implementing the `BaseChatModel` interface, we get access to the entire LangChain ecosystem:

- Works with LangGraph for agent orchestration
- Compatible with LangChain chains and tools
- Supports streaming, callbacks, and all LangChain features
- Can be swapped with `ChatOpenAI` in a single line

**Real-world parallel:** `ChatOpenAI`, `ChatAnthropic`, `ChatCohere` all implement this same interface. It's the standard.

### Why Chat Abstractions Exist

You might be wondering: "Why all this conversion between formats? Why not just use strings?"

Here's why chat abstractions matter:

**1. Conversation History**

Real applications need context:

```python
messages = [
    SystemMessage(content="You are a helpful dog."),
    HumanMessage(content="Bark!"),
    AIMessage(content="woof woof"),
    HumanMessage(content="Bark louder!"),
]

response = bark_llm.invoke(messages)
```

Without a standard message format, every developer would implement conversation history differently. With `BaseChatModel`, it's standardized.

**2. Tool Calling**

Agents need to call tools:

```python
messages = [
    HumanMessage(content="What's the weather?"),
    AIMessage(content="", tool_calls=[
        {"name": "get_weather", "args": {"location": "SF"}}
    ]),
    ToolMessage(content="72°F and sunny", tool_call_id="1"),
]
```

The `BaseChatModel` interface handles this automatically.

**3. Streaming**

Production apps need streaming responses:

```python
for chunk in bark_llm.stream(messages):
    print(chunk.content, end="", flush=True)
```

**4. Swappability**

Development vs production:

```python
# Development: use our tiny BarkGPT
llm = ChatBarkGPT(client=BarkClient(...))

# Production: use GPT-4
llm = ChatOpenAI(model="gpt-4")

# Same agent code, different brain
agent = create_agent(llm)
```

This is the power of abstractions. The `BaseChatModel` interface defines the contract. As long as you implement it, you can plug into the ecosystem.

## Layer 5: BarkAgent - Orchestrating Workflows

Now we get to the fun part. We have a LangChain-compatible model, so we can build a proper agent with LangGraph.

**What is an agent?** An agent is a system that:

1. Takes a goal as input
2. Decides what steps to take
3. Executes those steps (possibly calling tools)
4. Manages state across steps
5. Repeats until the goal is achieved

Here's a simple BarkAgent:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """State that persists across agent steps."""
    messages: List[BaseMessage]
    bark_count: int
    enthusiasm_level: str

# Initialize our LLM
bark_llm = ChatBarkGPT(client=BarkClient("http://api.bark-slm.com"))

# Define the workflow graph
workflow = StateGraph(AgentState)

def bark_step(state: AgentState) -> AgentState:
    """Agent decides how to bark based on state."""
    messages = state["messages"]

    # The LLM "thinks" by processing messages
    response = bark_llm.invoke(messages)

    # Update state
    return {
        "messages": messages + [response],
        "bark_count": state["bark_count"] + 1,
        "enthusiasm_level": "high" if "woof" in response.content else "low"
    }

def should_continue(state: AgentState) -> str:
    """Decide whether to bark more or stop."""
    if state["bark_count"] >= 5:
        return "end"
    if state["enthusiasm_level"] == "low":
        return "end"
    return "continue"

# Build the graph
workflow.add_node("bark", bark_step)
workflow.set_entry_point("bark")
workflow.add_conditional_edges(
    "bark",
    should_continue,
    {
        "continue": "bark",
        "end": END
    }
)

# Compile into runnable agent
bark_agent = workflow.compile()
```

Using the agent:

```python
from langchain_core.messages import HumanMessage

result = bark_agent.invoke({
    "messages": [HumanMessage(content="Make the dog bark enthusiastically!")],
    "bark_count": 0,
    "enthusiasm_level": "unknown"
})

print(result["messages"][-1].content)
# Output: "woof woof ruff woof woof"
```

**Why this layer exists:** Agents are the whole point. The model is just the brain—the agent is the entity that makes decisions and takes actions.

**Real-world parallel:** This is how GitHub Copilot Workspace works (multi-step code editing), how Perplexity works (multi-step research), and how most production AI agents work.

## Using BarkGPT as an Agent - Practical Examples

Let's see BarkAgent in action with some real scenarios.

### Example 1: Simple Barking Agent

```python
from bark_gpt.client import BarkClient
from bark_gpt.chat import ChatBarkGPT
from bark_gpt.agent import BarkAgent

# Initialize the stack
client = BarkClient(base_url="http://localhost:80")
chat_model = ChatBarkGPT(client=client)
agent = BarkAgent(llm=chat_model)

# Run the agent
result = agent.run("Make the dog bark enthusiastically!")
print(result)
# Output: "woof woof ruff ruff grrr woof!"
```

### Example 2: Agent with Memory

```python
from langgraph.checkpoint.memory import MemorySaver

# Add memory to the agent
memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)

# First conversation
config = {"configurable": {"thread_id": "user_123"}}
agent.invoke({"messages": [HumanMessage(content="Bark!")]}, config)

# Second conversation - agent remembers
agent.invoke({"messages": [HumanMessage(content="Bark again!")]}, config)
```

### Example 3: Testing Without the API

This is where abstractions really shine:

```python
from unittest.mock import Mock

# Mock the client - no API calls!
mock_client = Mock(spec=BarkClient)
mock_client.generate.return_value = "woof woof"

# Test the agent logic
chat_model = ChatBarkGPT(client=mock_client)
agent = BarkAgent(llm=chat_model)

result = agent.run("test input")
assert "woof" in result
assert mock_client.generate.called
```

### Example 4: Swapping the LLM

```python
# Development: use BarkGPT
dev_llm = ChatBarkGPT(client=BarkClient("http://localhost:80"))
dev_agent = BarkAgent(llm=dev_llm)

# Production: use GPT-4
prod_llm = ChatOpenAI(model="gpt-4")
prod_agent = BarkAgent(llm=prod_llm)

# Same agent code, different brain!
```

## Why All These Layers?

At this point you might be thinking: "This seems like a lot of abstraction for a model that just barks."

You're right. But here's the thing—this is _exactly_ how production AI systems work.

### It's About the Pattern

Every production AI agent follows this same structure:

| Layer             | BarkGPT Stack   | Production Example                    |
| ----------------- | --------------- | ------------------------------------- |
| Foundation Model  | BarkGPT (117KB) | GPT-4, Claude, Llama                  |
| API Service       | FastAPI         | OpenAI API, Anthropic API             |
| Client SDK        | BarkClient      | `openai` package, `anthropic` package |
| Framework Adapter | ChatBarkGPT     | `ChatOpenAI`, `ChatAnthropic`         |
| Agent             | BarkAgent       | GitHub Copilot, Perplexity AI         |

The only difference is scale. Our model is 117KB. GPT-4 is 1.76 trillion parameters. But the architecture is the same.

### Real Companies Using This Pattern

- **Intercom:** Customer support agents route tickets using LLM-powered decision trees
- **GitHub Copilot Workspace:** Multi-step code editing agents that plan, execute, and verify changes
- **Perplexity:** Research agents that search, synthesize, and cite sources
- **Cursor:** Codebase understanding agents that analyze and modify multiple files

They all use this stack: Model → API → Client → Framework → Agent.

### Why Abstractions Matter in Practice

**Testing Without API Calls:**

```python
# Don't hit the real API during tests
mock_client = Mock()
mock_client.generate.return_value = "woof"

agent = BarkAgent(llm=ChatBarkGPT(client=mock_client))
assert agent.run("test") == "woof"
```

**Observability Across the Stack:**

```python
class BarkClient:
    def generate(self, prompt):
        logger.info(f"BarkGPT API call: {prompt}")
        start = time.time()

        result = self._call_api(prompt)

        logger.info(f"Response in {time.time() - start:.2f}s")
        metrics.record_latency(time.time() - start)
        return result
```

Add logging once, see it everywhere.

**Swapping Components:**

```python
# Use BarkGPT locally
if os.getenv("ENV") == "development":
    llm = ChatBarkGPT(client=BarkClient("http://localhost:80"))
else:
    llm = ChatOpenAI(model="gpt-4")

# Agent code stays the same
agent = BarkAgent(llm=llm)
```

## Getting Started

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/bark-gpt
cd bark-gpt

# Install dependencies
pip install -r requirements.txt
```

### Running the Stack Locally

```bash
# 1. Train the model (or use pre-trained weights)
make train

# 2. Start the API server
docker compose up

# 3. Run an agent example
python examples/bark_agent_demo.py
```

### Project Structure

```
bark-gpt/
├── bark_gpt/
│   ├── model.py              # BarkGPT transformer
│   ├── prepare_dataset.py    # Dataset generation
│   ├── train.py              # Training loop
│   ├── client.py             # BarkClient (HTTP client)
│   ├── chat.py               # ChatBarkGPT (BaseChatModel)
│   └── agent.py              # BarkAgent (LangGraph)
├── slm/
│   └── main.py               # FastAPI server
├── examples/
│   └── bark_agent_demo.py    # Usage examples
├── Dockerfile                # API server container
└── docker-compose.yaml       # Local deployment
```

## What You Learn From This Project

1. **How LLMs work under the hood** - You trained one from scratch
2. **How to serve models in production** - FastAPI + Docker deployment
3. **How to build proper API clients** - Retry logic, error handling, timeouts
4. **How to integrate with LangChain** - Implementing `BaseChatModel`
5. **How to build agents** - LangGraph state management and orchestration
6. **Why abstractions matter** - Testing, swapping, observability

This is the full vertical stack. From `torch.nn.Module` to production-ready agent.

Most tutorials skip the first three layers and just hand you an API key. Which is fine for shipping fast. But if you want to understand what's actually happening under the hood? You need to build it yourself at least once.

## Observations and Limitations

### Why This Matters Despite Being Tiny

BarkGPT is obviously not production-ready. It has a 9-token vocabulary and generates gibberish outside its training distribution.

But that's not the point.

The point is understanding the architecture. Once you understand how to build an agent around a model you trained yourself, you understand how the whole system works. All of them.

The patterns are identical whether you're using:

- BarkGPT (117KB, 9 tokens)
- GPT-2 (774M parameters, 50K tokens)
- GPT-4 (1.76T parameters, 100K tokens)

### What's Missing

If you wanted to make this production-ready, you'd need:

**Better model:**

- More transformer layers (BarkGPT has 2, GPT-2 has 12)
- Larger embedding dimensions
- Trained on billions of tokens instead of hundreds

**Better infrastructure:**

- GPU serving for faster inference
- Load balancing across multiple instances
- Proper monitoring and alerting
- Rate limiting and authentication

**Better agent capabilities:**

- Tool calling (let the agent _do_ things)
- Memory and retrieval (RAG)
- Multi-agent collaboration
- Proper error recovery

But the architecture would stay the same. That's the beauty of abstractions.

## What's Next

Some ideas if you want to extend this:

- **Add RAG:** Give the agent a vector database of dog facts
- **Add tools:** Weather API, email sending, web search
- **Multi-agent:** Build a pack of dogs that collaborate
- **Streaming:** Real-time token streaming to the client
- **Monitoring:** LangSmith integration for observability

Or just keep it as-is. A functioning AI agent that barks. Sometimes that's enough.

## Conclusion

This project demonstrates the full vertical stack of modern AI systems:

```
Neural Network → API → Client → Framework → Agent
```

Yes, BarkGPT is whimsical. But the architecture is real.

This is how GitHub Copilot works. This is how ChatGPT works. This is how every production AI agent works.

The only difference is the size of the model and the complexity of the tasks.

Once you understand this stack, you understand AI systems. All of them.

---

**Links:**

- [BarkGPT Main README](./README.md) - How we trained the model
- [Working with Datasets](./docs/datasets.md) - Training on real text
- [Live Demo](https://www.bark-slm.com) - Try it yourself
- [Hugging Face](https://huggingface.co/vvasylkovskyi/barkgpt) - Pre-trained weights

Built with PyTorch, FastAPI, LangChain, LangGraph, and way too much coffee ☕
