
from tool.llm.base import LLMBackend

from transformers import Trainer, AutoTokenizer, AutoModel

class TrainedLocalLLMBackend(LLMBackend):
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name

    def initialize(self):
        self.model = AutoModel.from_pretrained(self.base_model_name)
        self.trainer = Trainer(
            model=self.model,
        )

        raise NotImplementedError()

    
    def train(self, _, **__):
        """
        Run training on this model.
        """
        raise NotImplementedError()
    
    def query(self, _: str) -> str:
        """
        Query the LLM.
        """
        raise NotImplementedError()