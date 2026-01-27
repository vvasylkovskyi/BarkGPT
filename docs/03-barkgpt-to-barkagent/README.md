# BarkGPT Agent Stack

## Building AI Agents From Scratch

AI Agents are everywhere now. GitHub Copilot edits your code, Perplexity researches topics for you, and every SaaS company is slapping "AI Agent" on their landing page. But here's the thing: most tutorials start with `pip install openai` and call it a day. That's fine if you just want to ship something. But what if you want to understand how it actually works? Like, all the way down to the neural network? Most tutorials skip the first three layers and just hand you an API key. Which is fine for shipping fast.

This project demonstrates the full vertical stack. From `torch.nn.Module` to production-ready agent. If you want to understand what's actually happening under the hood, then keep reading. We're using BarkGPT, a tiny language model that barks, as our foundation model.

## Prerequisites

To fully follow this article, I recommend understanding how the LLM is trained and built. This article is NOT another tutorial explaining how to setup langchain.

Here are the previous reads:

- [Building BarkGPT from scratch](./docs/01-building-from-scratch/README.md)
- [Training BarkGPT on WebText‑2 dataset](./docs/02-training-on-webtext/README.md)

## Adding ChatBarkGPT

So we have our language model surfaced behind an API: `/generate`. We can invoke it with our prompt, and get the message generated back. Why do we need to work more and add langchain for the production AI systems? Let's find out while building.

A traditional langchain workflow is building on top of LLMs, and the abstractions wrapping LLM. If we were to build such abstractions for `BarkGPT`, we would need to do something like follows:

```python
class BarkAgent:
    def __init__(self):
        self.llm: BaseChatModel = ChatBarkGPT(base_url="http://localhost:80")

    def _call_model(self, state: AgentState):
        messages = self._get_messages_with_prompt(state.messages)
        response: BaseMessage = self.llm.invoke(messages)
        return {"messages": [response]}
```

The above code is a typical minimal "AI Agent". It has to invoke our BarkGPT somehow. We can see that `ChatBarkGPT` will call the LLM using `self.llm.invoke`. But besides the `invoke` which is a simple HTTP Request, there are other ways to get response from LLM api's: streaming using server sent events (SSE) or web sockets. Langchain helps us handle those using the `BaseChatModel`. So the bottom line, our `ChatBarkGPT` has to implement it to inherit all the client-side streaming and whatnot capabilities:

```python
from langchain_core.language_models import BaseChatModel
class ChatBarkGPT(BaseChatModel):

    def __init__(self, base_url: str):
        super().__init__()
        self._client = BarkClient(base_url=base_url)

```

The client above, is our API that will get the `/generate`. We will get there in a minute.

### Converting Messages list into prompt contract

Something that developers have been enjoing doing is to hide away the complexity. The `BaseChatModel` is no exception. We have observed from the previous articles that our LLM only is ready to accept the `prompt` which is a string. But the chat conversation is a list of messages, some our system, some human and ai. So of course we will want to convert those reliably. Here is an example of such convertion:

```python
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: list[str] | None = None,
        **kwargs,
    ) -> ChatResult:
        # 1. Convert LangChain messages → BarkGPT prompt
        prompt = self._messages_to_prompt(messages)

        # 2. Call model
        text = self._client.generate(prompt)
        # 3. Wrap result back into LangChain objects
        ai_message = AIMessage(content=text)
        generation = ChatGeneration(message=ai_message)

        return ChatResult(generations=[generation])

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        chunks = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                chunks.append(f"[SYSTEM]\n{msg.content}")
            elif isinstance(msg, HumanMessage):
                chunks.append(f"[USER]\n{msg.content}")
            else:
                chunks.append(f"[ASSISTANT]\n{msg.content}")
        return "\n\n".join(chunks)
```

You can see that the messages are converted into `[SYSTEM], [USER], [ASSISTANT]`. And unsurprisingly we place them inside of the `ChatBarkGPT` abstraction. There are several reasons to see these delimiters:

1. The LLMs expect the string, so the list of messages is flattened into string and would look like this:

```txt
[SYSTEM]
...

[USER]
...

[ASSISTANT]
...
```

2. The second reason is more subtile. The language models are trained on the patterns, and each model has its own. Our `BarkGPT` was not trained to see the delimiters like above, so they will probably have no meaning leading it to hallucinate. However, OpenAI has been trained to see patterns, so they established the **prompt contract** with the structure above. This is the `adapter` pattern - we adapt the python messages to language native text - prompt contract structure.

#### What about coversation history

Overlysimplifying this, we convert all the messages into one single prompt, so essentially on every message we invoke the LLM with full conversation. This is obivously linearly inneficient, but it is the way things work.

### Invoking BarkGPT /generate

Final piece of puzzle is the definition of `BarkClient` which is a class responsible for invoking the model. Here is where the API request is made, that is also owned by the model owners. Ours will be very simple:

```python
class BarkClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()  # Reuse connections

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate barks from the BarkGPT API."""
        response = self.session.post(
            f"{self.base_url}/v1/model/generate",
            json={"prompt": prompt, **kwargs},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]
```

It is overly simplified, but in production gets messy fast. What about:

- Timeouts?
- Retries when the server is temporarily down?
- Error handling when the API returns 500?
- Connection pooling for efficiency?

### Exposing server with FastAPI

Of course the `BarkClient` assumes that our model lives on `/v1/model/generate`. And in fact this is the case. If you were to look into our server implementation you would see something in FastAPI:

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

@app.post("/v1/model/generate")
async def generate():
    prompt = "User: speak Assistant:"
    tokens = [tokenizer[t] for t in prompt.split()]
    input_ids = torch.tensor([tokens])

    output_ids = model.generate(input_ids, max_new_tokens=10)
    output = [itos[i] for i in output_ids[0].tolist()]

    return {"message": " ".join(output)}
```

Now anyone can curl our model:

```bash
curl -X POST https://api.bark-slm.com/generate
# {"message": "woof woof ruff"}
```

### Coordination within the Agent Layer

Now we have all the puzzles to invoke our LLM reliably:

- We have adapted chat interface messages into prompt contract using langchain
- We have invoked our API using the prompt and our custom client

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

## Why All These Layers?

At this point you might be thinking: "This seems like a lot of abstraction for a model that just barks."

You're right. But here's the thing—this is _exactly_ how production AI systems work. Every production AI agent follows this same structure:

| Layer             | BarkGPT Stack   | Production Example                    |
| ----------------- | --------------- | ------------------------------------- |
| Foundation Model  | BarkGPT (117KB) | GPT-4, Claude, Llama                  |
| API Service       | FastAPI         | OpenAI API, Anthropic API             |
| Client SDK        | BarkClient      | `openai` package, `anthropic` package |
| Framework Adapter | ChatBarkGPT     | `ChatOpenAI`, `ChatAnthropic`         |
| Agent             | BarkAgent       | GitHub Copilot, Perplexity AI         |

The only difference is scale. Our model is 117KB. GPT-4 is 1.76 trillion parameters. But the architecture is the same.

## What's Next

Some ideas if you want to extend this:

- **Add RAG:** Give the agent a vector database of facts
- **Add tools:** Weather API, email sending, web search
- **Multi-agent:** Build a pack of dogs agents that collaborate
- **Streaming:** Real-time token streaming to the client
- **Monitoring:** LangSmith integration for observability

Or we just keep it as-is. A functioning AI agent that barks. Sometimes that's enough.

## Conclusion

This project demonstrates the full vertical stack of modern AI systems:

```
Neural Network → API → Client → Framework → Agent
```

Yes, BarkGPT is whimsical. But the architecture is real. This is how GitHub Copilot works. This is how ChatGPT works. This is how every production AI agent works. The only difference is the size of the model and the complexity of the tasks. Once you understand this stack, you understand AI systems. All of them.
I hope you enjoyed this, keep building it with your PyTorch, FastAPI, LangChain, LangGraph, and way too much coffee.

**Links:**

- [BarkGPT Main README](./README.md) - How we trained the model
- [Live Demo](https://www.bark-slm.com) - Try it yourself
- [Hugging Face](https://huggingface.co/vvasylkovskyi/barkgpt) - Pre-trained weights
