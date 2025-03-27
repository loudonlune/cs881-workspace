
from tool.llm.base import LLMBackend

from transformers import BitsAndBytesConfig, pipeline

class LocalLLMBackend(LLMBackend):
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name

    def initialize(self, bits_and_bytes_cfg: dict = {"load_in_4bit": True}):
        quantization = BitsAndBytesConfig(**bits_and_bytes_cfg)
        self.pipeline = pipeline(
            'text2text-generation',
            model=self.base_model_name,
            model_kwargs={"quantization_config": quantization},
        )
    
    def train(self, _, **__):
        """
        Run training on this model.
        """
        pass
    
    def query(self, prompt: str) -> str:
        """
        Query the LLM.
        """

        return self.pipeline(prompt)

