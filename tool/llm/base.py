
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
    
    def query(self, _: str) -> str:
        """
        Query the LLM.
        """
        raise NotImplementedError()
