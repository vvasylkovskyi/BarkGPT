import logging
from functools import lru_cache

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_work_topic: str = "a2a-agent-work-queue"
    kafka_results_topic: str = "a2a-agent-results"
    # Each message is delivered to only ONE consumer per group
    # So this consumer group is different from the one in the logs agent
    kafka_consumer_group: str = "sre-agent-consumer-group"


@lru_cache
def get_settings():
    return AppSettings()


app_settings = AppSettings()
