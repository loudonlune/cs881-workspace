
from tool.llm.base import LLMBackend

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch

class LocalLLMBackend(LLMBackend):
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name

    def initialize(self, use_flash: bool = False, additional_model_args: dict = {}):
        if use_flash:
            additional_model_args['attn_implementation'] = 'flash_attention_2'

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            #torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            **additional_model_args,
        )
    
    def train(self, _, **__):
        """
        Run training on this model.
        """
        pass
    
    def query(self, prompt: str, system_prompt: str = "You are an assistant following the instructions given by the user to complete a given task.") -> str:
        """
        Query the LLM.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        input_embedding = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)

        output_embedding = self.model.generate(
            **input_embedding,
            max_new_tokens=2048,
        )

        output_text = self.tokenizer.batch_decode(output_embedding, skip_special_tokens=True)[0]

        return output_text.split('</think>')[-1].strip()

