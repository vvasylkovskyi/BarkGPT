import logging
from functools import lru_cache

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    model_path: str = ""
    base_url: str = "http://localhost:80"


@lru_cache
def get_settings():
    return AppSettings()


app_settings = AppSettings()
