
from huggingface_hub import login
import os

def login_to_huggingface():
    huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
    login(huggingface_token)

class LLMBackend(object):
    """
    Interface for whatever LLM we are using to implement the task.
    """

    def initialize(self):
        raise NotImplementedError()
    
    def train(self, _, **__):
        """
        Run training on this model.
        """
        raise NotImplementedError()
    
    def query(self, *args, **kwargs) -> str:
        """
        Query the LLM.
        """
        raise NotImplementedError()
