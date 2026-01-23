import asyncio
import logging

from a2a.types import (
    TaskState,
)
from app.a2a_integration.task_store import in_memory_task_store

from app.kafka.kafka_consumer import KafkaAgentConsumer
from app.settings.app import get_settings

logger = logging.getLogger(__name__)
app_settings = get_settings()


class LogsAgentResultsConsumerWorker:
    """Worker that processes tasks from Kafka work queue asynchronously."""

    def __init__(self):
        self.consumer: KafkaAgentConsumer | None = None
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    async def process_log_agent_result_task(self, event: dict):
        task_id = event.get("task_id")
        # context_id = event.get("context_id")
        # account_id = event.get("metadata", {}).get("account_id")
        # user_id = event.get("metadata", {}).get("user_id")
        # token = event.get("metadata", {}).get("token")
        query = event.get("query")

        # Event Example
        # {'task_id': 'e68422e2-c93e-4916-9c00-db5881cc084f', 'context_id': 'c96b5522-506c-4fde-a5a4-8f8cd4a82f7f',
        # 'event_type': 'completed', 'payload': {'result':
        # {'message': "Hello! How can I assist you with your Honeycomb logs or datasets today? If you need to search logs, analyze events, or view dataset details, just let me know what you're looking for.",
        # 'span_id': '687b45c34394141d', 'data': None,
        # 'query_data': {'server': 'honeycomb', 'data': []}}, 'query': 'Hello'}
        print(">>> event", event)
        message = event["payload"]["result"]["message"]
        task_id = event["task_id"]

        task = await in_memory_task_store.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found in task store")
            return

        print(">>> message", message)

        logger.info(f"Processing log agent result task {task_id}: {query}")

        # try:
        #     body = RequestModel(message=query)
        #     result = await run_agent(
        #         body,
        #         x_calling_agent="sre_agent",
        #         account_id=account_id,
        #         user_id=user_id,
        #     )

        #     print(">>> LogsAgentWorker - Task result:", result)
        #     logger.info(f"Task {task_id} completed successfully")

        # except Exception as e:
        #     logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        #     await self.producer.publish_result(
        #         task_id=task_id,
        #         context_id=context_id,
        #         event_type=TaskState.failed,
        #         payload={
        #             "error": str(e),
        #             "error_type": type(e).__name__,
        #             "query": query,
        #         },
        #     )

    async def start(self):
        """Start the Kafka consumer worker."""
        self.consumer = KafkaAgentConsumer(
            bootstrap_servers=app_settings.kafka_bootstrap_servers,
            topic=app_settings.kafka_results_topic,
            group_id=app_settings.kafka_consumer_group,
        )
        await self.consumer.start()
        logger.info("LogsAgentWorker started consuming tasks from Kafka")

        # Run consumer in background
        self._task = asyncio.create_task(self._run_consumer())

    async def _run_consumer(self):
        """Background task to consume events until stopped."""
        try:
            await self.consumer.consume_events(self.process_log_agent_result_task)
        except asyncio.CancelledError:
            logger.info("LogsAgentWorker background task cancelled")
        finally:
            await self.consumer.stop()
            self._stopped.set()

    async def stop(self):
        """Stop the worker gracefully."""
        if self._task:
            self._task.cancel()
            await self._stopped.wait()
            logger.info("LogsAgentWorker stopped")
