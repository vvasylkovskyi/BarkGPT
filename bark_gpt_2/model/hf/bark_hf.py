from transformers import PreTrainedModel, GenerationMixin
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput


class BarkConfig(PretrainedConfig):
    model_type = "gpt2"

    def __init__(self, n_layer=2, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = n_layer  # required


class BarkHF(PreTrainedModel, GenerationMixin):
    config_class = BarkConfig

    def __init__(self, config, bark_model):
        super().__init__(config)
        self.bark = bark_model

    def forward(self, input_ids, return_dict=True, **kwargs):
        logits = self.bark(input_ids)
        if return_dict:
            return CausalLMOutput(logits=logits)
        return (logits,)
