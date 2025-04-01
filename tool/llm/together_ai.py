
import pandas
import os
from queue import Queue
import time
from together import Together
from together.types.chat_completions import ChatCompletionResponse

from typing import Optional

from tool.llm.base import LLMBackend

FREE_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

def get_together_client() -> Optional[Together]:
    if api_key := os.environ.get('TOGETHER_API_KEY'):
        return Together(api_key=api_key)
    else:
        print("Failed to get the API key, which is required for the operation. Set TOGETHER_API_KEY in the environment.")
        return None


def together_prompt(model: str, prompt: str) -> str:
    if client := get_together_client():
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content
    else:
        print("Error: Failed to get the client.")
        return ""

class TogetherLLMBackend(LLMBackend):
    """
    LLM backend for LLMs provided by the Together AI service.
    Can make up to 10 requests per minute with the free model (set as default).
    There could be support for training, but it is not implemented, and it costs a considerable amount of money.
    """

    # Configure the throttling (max 10 requests per minute)
    THROTTLE_TIME: float = 60.0
    THROTTLE_MAX:  int = 10

    # This queue is used to throttle queries to Together.
    _tq: Queue
    _model: str

    together_client: Together
    
    def __init__(self, model: str = FREE_MODEL):
        self._tq = Queue(TogetherLLMBackend.THROTTLE_MAX)
        self._model = model

    def initialize(self, *args, **kwargs):
        self.together_client = get_together_client()
        if not self.together_client:
            raise ResourceWarning("Failed to create Together client.")

    def train(self, _: pandas.DataFrame):
        pass

    def query(self, querytext: str, system_prompt: Optional[str] ) -> str:
        # Throttling. Wait such that the oldest request occurred over a minute ago if we have issued
        #   10 requests and the oldest of those occurred less than a minute ago.
        if self._tq.qsize() >= TogetherLLMBackend.THROTTLE_MAX:
            last_time: float = self._tq.get_nowait()
            delta: float = time.monotonic() - last_time
            if delta < TogetherLLMBackend.THROTTLE_TIME:
                time.sleep(60.0 - delta)
        if system_prompt is None:
            message=[{
                "role":"system", "content": system_prompt,
                "role": "user", "content": querytext,
            }]
        else:
            message=[{
                "role": "user", "content": querytext,
            }]

        # Put in the time we're running the query into the queue.
        self._tq.put_nowait(time.monotonic())
        response: ChatCompletionResponse = self.together_client.chat.completions.create(
            model=self._model,
            messages=message
        )

        if type(response) is not ChatCompletionResponse:
            raise NotImplementedError("Together API returned completion chunks.")
        
        if len(response.choices) > 1:
            print("More than one choice was given.")
        elif len(response.choices) == 0:
            raise Exception("Got zero choices")

        return response.choices[0].message.content
