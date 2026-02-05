from transformers import AutoTokenizer
from models.bark_gpt_2.model.model import BarkGPT
from models.bark_gpt_2.model.hf.bark_hf import BarkHF, BarkConfig
from models.bark_gpt_2.parameters.parameters import (
    model_config,
    device,
    generation_parameters,
)
from logger.logger import Logger
from models.bark_gpt_2.model_checkpoint_manager.model_checkpoints_manager import (
    ModelCheckpointsManager,
)

logger = Logger("test_gen")


model_checkpoints_manager = ModelCheckpointsManager(
    device, model_config, checkpoint_interval=1000
)
ckpt = model_checkpoints_manager.load_final_model_weights()
tokenizer = AutoTokenizer.from_pretrained("bark_gpt_2_tokenizer")

model = BarkGPT(model_config).to(device)
model.load_state_dict(ckpt["model_state"])


config = BarkConfig(vocab_size=ckpt["vocab_size"])
hf_model = BarkHF(config, model)

prompt = "the Jewish population"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

gen_tokens = hf_model.generate(
    input_ids,
    do_sample=True,
    temperature=generation_parameters.temperature,
    top_k=generation_parameters.top_k,
    max_length=generation_parameters.max_length,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
logger.info(f"Generated: {gen_text[len(prompt) :]}")
