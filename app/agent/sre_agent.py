from dataclasses import dataclass
import logging
from typing import Any, cast
from uuid import uuid4
import httpx
from a2a.client import (
    A2ACardResolver,
    A2AClient,
)
from a2a.types import (
    AgentCard,
    Message,
    MessageSendParams,
    SendStreamingMessageSuccessResponse,
    TaskStatusUpdateEvent,
    SendStreamingMessageResponse,
    SendStreamingMessageRequest,
    TextPart,
    TaskState,
    JSONRPCErrorResponse,
)
from a2a.utils import new_agent_text_message, new_task

from app.a2a_integration.task_store import in_memory_task_store

logger = logging.getLogger(__name__)


def create_send_message_payload(
    text: str,
    task_id: str | None = None,
    context_id: str | None = None,
) -> dict[str, Any]:
    """Helper function to create the payload for sending a task."""
    payload: dict[str, Any] = {
        "message": {
            "role": "user",
            "parts": [
                {"kind": "text", "text": text},
            ],
            "messageId": uuid4().hex,
        },
    }

    if task_id:
        payload["message"]["taskId"] = task_id

    if context_id:
        payload["message"]["contextId"] = context_id
    return payload


def on_event(
    event: SendStreamingMessageResponse,
) -> tuple[str | None, str | None, TaskState | None] | None:
    """Callback for each streaming event."""
    if isinstance(event.root, JSONRPCErrorResponse):
        logger.error(
            f"[STREAM] Logs agent service returned an error: {event.root.error.message}"
        )
        return None, None, None

    if not isinstance(event, SendStreamingMessageResponse) or not isinstance(
        event.root, SendStreamingMessageSuccessResponse
    ):
        logger.error("[STREAM] Unexpected service response")
        return None, None, None

    success_streaming_message_response: SendStreamingMessageSuccessResponse = event.root
    if not isinstance(success_streaming_message_response.result, TaskStatusUpdateEvent):
        logger.error("[STREAM] Unexpected task status message type")
        return None, None, None

    task_update_event: TaskStatusUpdateEvent = success_streaming_message_response.result
    if task_update_event.status.state == TaskState.submitted:
        message_content = _extracted_from_on_event(task_update_event)
        logger.info(
            f"[STREAM] Logs agent service task submitted with ID: {task_update_event.task_id}"
        )
        return (
            message_content,
            task_update_event.task_id,
            task_update_event.status.state,
        )

    if task_update_event.status.state == TaskState.failed:
        message_content = _extracted_from_on_event(task_update_event)
        error_msg = f"[STREAM] Logs agent service task with ID: {task_update_event.task_id} failed: {message_content}"
        logger.error(error_msg)
        return (
            message_content,
            task_update_event.task_id,
            task_update_event.status.state,
        )


def _extracted_from_on_event(task_update_event: TaskStatusUpdateEvent) -> str:
    message = cast(Message, task_update_event.status.message)
    text_part = next(
        (part.root for part in message.parts if isinstance(part.root, TextPart)),
        None,
    )
    result = text_part.text if text_part else ""
    return result


@dataclass
class TaskSubmittedResponse:
    task_state: TaskState
    task_id: str
    message: str


class SREAgent:

    async def run(self) -> TaskSubmittedResponse:
        ### I am an Agent, and I decide to call a tool to help me complete my task
        ### This tool call is done via A2A client to another agent service
        ### Here we assume that SRE Agent has done some reasoning and below is the
        ### Tool call it wants to make to Logs Agent to look into logs

        async with httpx.AsyncClient() as httpx_client:

            httpx_client.headers.update(
                {
                    "x-calling-agent": "sre_agent",
                }
            )
            httpx_client.timeout = httpx.Timeout(timeout=30)

            card_resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8007",
            )

            logs_agent_card: AgentCard = await card_resolver.get_agent_card()

            client = A2AClient(
                httpx_client=httpx_client,
                agent_card=logs_agent_card,
            )

            send_message_payload: dict[str, Any] = create_send_message_payload(
                text="Hello",
            )

            a2a_request = SendStreamingMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload)
            )

            async for event in client.send_message_streaming(a2a_request):
                message, task_id, task_state = on_event(event)  # type: ignore
                print(">>> message: ", message)
                if task_state == TaskState.submitted:
                    response = TaskSubmittedResponse(
                        task_state=task_state,
                        task_id=cast(str, task_id),
                        message=cast(str, message),
                    )
                    # Note, the in_memory_task_store is required to keep track of tasks in progress
                    # And to subscribe to task_id in kafka consumer for updates
                    # For the demo purpose, in_memory_task_store is in memory
                    # We may want to implement persistent storage for production use cases
                    task = new_task(new_agent_text_message(cast(str, message)))
                    task.id = cast(str, task_id)
                    task.metadata = {"owner_id": "sre_agent_demo"}

                    await in_memory_task_store.save(task)

                    return response

            raise RuntimeError("Logs agent task submission failed")
