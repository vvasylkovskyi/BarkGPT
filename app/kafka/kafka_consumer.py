import json
import logging
from collections.abc import Callable
from typing import Any

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class KafkaAgentConsumer:
    """Kafka consumer for A2A agent communication."""

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        task_id_filter: str | None = None,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.task_id_filter = task_id_filter
        self.consumer: AIOKafkaConsumer | None = None

    async def start(self):
        """Initialize and start the Kafka consumer."""
        try:
            self.consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                max_poll_records=10,  # Process in small batches
            )
            await self.consumer.start()
            logger.info(
                f"Kafka consumer started, subscribed to {self.topic} "
                f"(group: {self.group_id}, filter: {self.task_id_filter})"
            )
        except KafkaError as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise

    async def stop(self):
        """Stop the Kafka consumer."""
        if self.consumer:
            await self.consumer.stop()
            logger.info("Kafka consumer stopped")

    async def consume_events(
        self,
        event_handler: Callable[[dict[str, Any]], None],
    ):
        """Consume events from Kafka and process them.

        Args:
            event_handler: Async function to handle each consumed event
        """
        if not self.consumer:
            logger.error("Kafka consumer not initialized")
            return

        try:
            async for msg in self.consumer:
                try:
                    event = msg.value
                    task_id = event.get("task_id")

                    # Filter by task_id if specified
                    if self.task_id_filter and task_id != self.task_id_filter:
                        continue

                    logger.info(
                        f"Consumed event '{event.get('event_type')}' for task {task_id} from Kafka"
                    )

                    await event_handler(event)

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)

        except KafkaError as e:
            logger.error(f"Kafka consumer error: {e}")
            raise
