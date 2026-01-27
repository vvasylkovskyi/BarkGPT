import requests
from app.context.app_context import AppContext


class BarkClient:
    def __init__(self, base_url: str, timeout: int = 3):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()  # Reuse connections

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate barks from the BarkGPT API."""
        # In production, these might be two services communicating over HTTP
        # But here we will do a direct invokation of the model,
        # since we are running both services as monolith locally.
        # response = self.session.post(
        #     f"{self.base_url}/v1/model/generate",
        #     json={"prompt": prompt, **kwargs},
        #     timeout=self.timeout,
        # )
        # response.raise_for_status()
        # return response.json()["message"]
        generator = AppContext.get_instance().get_bark_generator()
        return generator.generate(prompt)
