from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from app.bark_client.bark_client import BarkClient


class ChatBarkGPT(BaseChatModel):

    def __init__(self, base_url: str):
        super().__init__()
        self._client = BarkClient(base_url=base_url)

    @property
    def _llm_type(self) -> str:
        return "bark-gpt"

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
