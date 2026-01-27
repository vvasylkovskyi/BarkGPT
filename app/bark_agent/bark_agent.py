import uuid
from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages
from typing_extensions import Annotated

from app.chat_bark_gpt.chat_bark_gpt import ChatBarkGPT

SYSTEM_PROMPT = ""


@dataclass
class AgentState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


class BarkAgent:
    def __init__(self):
        self.llm: BaseChatModel = ChatBarkGPT(base_url="http://localhost:80")

    def _get_messages_with_prompt(
        self, messages: Sequence[BaseMessage]
    ) -> list[BaseMessage]:
        return [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    def _call_model(self, state: AgentState):
        messages = self._get_messages_with_prompt(state.messages)
        response: BaseMessage = self.llm.invoke(messages)
        return {"messages": [response]}

    def _build_graph(self):
        graph_builder = StateGraph(state_schema=AgentState)
        graph_builder.add_node("call_model", self._call_model)
        graph_builder.add_edge(START, "call_model")
        graph_builder.add_edge("call_model", END)
        return graph_builder.compile()

    async def _build_graph_and_invoke(
        self, message: str, config: RunnableConfig
    ) -> str:
        input_messages = [HumanMessage(message)]
        graph = self._build_graph()
        state = graph.invoke(AgentState(messages=input_messages), config)
        return state["messages"][-1].content

    def run(self, input_message: str):
        config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
        return self._build_graph_and_invoke(input_message, config)
