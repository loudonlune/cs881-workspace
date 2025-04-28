
from tool.llm.base import LLMBackend

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class LocalSeq2SeqLLMBackend(LLMBackend):
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name

    def initialize(self, use_flash: bool = False, use_4bit_quant: bool = True, additional_model_args: dict = {}):
        if use_flash:
            additional_model_args['attn_implementation'] = 'flash_attention_2'

        if use_4bit_quant:
            additional_model_args['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_storage=torch.bfloat16,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            **additional_model_args,
        )
    
    def train(self, _, **__):
        """
        Run training on this model.
        """
        pass
    
    def query(self,
              prompt: str,
              system_prompt: str =  "You create objective, verifiable summaries."
                                    "Prioritize summarizing the objective parts of the text."
                                    "Keep some key details, such as proper nouns, in your response.") -> str:
        """
        Query the LLM.
        """

        # messages = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": prompt},
        # ]

        # prompt_text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )

        #prompt_text = f"{system_prompt}\n{prompt}\n<think>"
        prompt_text = prompt

        input_embedding = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)

        output_embedding = self.model.generate(
            **input_embedding,
            max_new_tokens=8192,
            #temperature=1.1,
        )

        output_text = self.tokenizer.batch_decode(output_embedding, skip_special_tokens=True)[0]

        return output_text.split('</think>')[-1].strip()

class LocalCausalLLMBackend(LLMBackend):
    def __init__(self, base_model_name: str, system_prompt: str = "You create objective, verifiable summaries."
                                    "Prioritize summarizing the objective parts of the text."
                                    "Keep some key details, such as proper nouns, in your response."):
        self.base_model_name = base_model_name
        self.system_prompt = system_prompt

    def initialize(self, use_flash: bool = False, use_4bit_quant: bool = True, additional_model_args: dict = {}):
        if use_flash:
            additional_model_args['attn_implementation'] = 'flash_attention_2'

        if use_4bit_quant:
            additional_model_args['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_storage=torch.bfloat16,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            **additional_model_args,
        )
    
    def train(self, _, **__):
        """
        Run training on this model.
        """
        pass
    
    def query(self,
              prompt: str) -> str:
        """
        Query the LLM.
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
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
            max_new_tokens=8192,
            #temperature=1.1,
        )

        output_text = self.tokenizer.batch_decode(output_embedding, skip_special_tokens=True)[0]

        return output_text.split('</think>')[-1].strip()

