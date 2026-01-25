import logging
from typing import Optional

import torch

from app.settings.app import AppSettings
from bark_gpt.model.hf.bark_hf import BarkHF
from bark_gpt.model.model import BarkGPT
from bark_gpt.model.hf.bark_hf import BarkConfig
from bark_gpt.bark_text_generator.bark_text_generator import BarkTextGenerator

logger = logging.getLogger(__name__)


class AppContext:
    """Application context that holds global resources initialized once at startup.

    This avoids recreating these resources for each request.
    """

    _instance: Optional["AppContext"] = None

    def __init__(self, app_settings: AppSettings) -> None:
        """Initialize the application context with global resources.

        Args:
            app_settings: Application settings containing Redis configuration
        """
        self._app_settings = app_settings
        stoi, itos, hf_model = self.load_bark_gpt_model()
        self.stoi = stoi
        self.itos = itos
        self.hf_model = hf_model

    @classmethod
    def initialize(cls, app_settings: AppSettings) -> "AppContext":
        """Initialize the singleton instance of AppContext.

        Args:
            app_settings: Application settings containing Redis configuration

        Returns:
            The singleton AppContext instance
        """
        if cls._instance is None:
            cls._instance = cls(app_settings)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "AppContext":
        """Get the singleton instance of AppContext.

        Returns:
            The singleton AppContext instance

        Raises:
            RuntimeError: If the AppContext has not been initialized
        """
        if cls._instance is None:
            raise RuntimeError("AppContext has not been initialized")
        return cls._instance

    def get_bark_generator(self) -> BarkTextGenerator:
        stoi, itos, hf_model = self.get_bark_gpt_model()
        self._bark_generator: BarkTextGenerator = BarkTextGenerator(
            hf_model=hf_model,
            stoi=stoi,
            itos=itos,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return self._bark_generator

    def load_bark_gpt_model(self):
        """Load and return the BarkGPT model.

        Returns:
            The loaded BarkGPT model
        """
        # Placeholder for actual model loading logic
        checkpoint = torch.load(self._app_settings.model_path, map_location="cpu")
        stoi = checkpoint["stoi"]
        itos = checkpoint["itos"]
        vocab_size = checkpoint["vocab_size"]

        bark = BarkGPT(vocab_size)
        bark.load_state_dict(checkpoint["model_state"])

        config = BarkConfig(vocab_size=vocab_size)
        hf_model = BarkHF(config, bark)

        return stoi, itos, hf_model

    def get_bark_gpt_model(self):
        """Get the BarkGPT model.

        Returns:
            The BarkGPT model
        """
        return self.stoi, self.itos, self.hf_model
