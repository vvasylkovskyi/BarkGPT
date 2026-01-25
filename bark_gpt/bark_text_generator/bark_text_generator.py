from dataclasses import dataclass
from typing import Sequence, cast
import torch
from bark_gpt.model.hf.bark_hf import BarkHF


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 10
    temperature: float = 0.7


class BarkTextGenerator:
    def __init__(
        self,
        hf_model: BarkHF,
        stoi: dict[str, int],
        itos: dict[int, str],
        device: torch.device | str = "cpu",
    ):
        self.model = hf_model.to(device)
        self.stoi = stoi
        self.itos = itos
        self.device = device

    def route_user_input(self, user_input: str) -> str:
        # Anything the user types means "please bark"
        return "User: speak Assistant:"

    def generate(
        self,
        prompt: str,
        *,
        config: GenerationConfig = GenerationConfig(),
    ) -> str:
        prompt = self.route_user_input(prompt)
        tokens = prompt.split()
        input_ids = torch.tensor(
            [[self.stoi[t] for t in tokens]],
            device=self.device,
        )

        with torch.no_grad():
            output_ids: torch.LongTensor = cast(
                torch.LongTensor,
                self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                ),
            )
        return self._decode(tokens, output_ids)

    def _decode(self, input_tokens: Sequence[str], output_ids: torch.LongTensor) -> str:
        new_tokens = output_ids[0, len(input_tokens) :]
        return " ".join(self.itos[int(t)] for t in new_tokens)
