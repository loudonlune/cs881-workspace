
from math import trunc
from tool.llm.base import LLMBackend

from transformers import Trainer, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, BitsAndBytesConfig
from datasets import Dataset
import evaluate
import numpy
import torch

from peft import LoraConfig, TaskType, get_peft_model

class TrainedLocalLLMBackend(LLMBackend):
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

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = get_peft_model(AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            **additional_model_args,
        ), lora_cfg)
    
    def train(self, training_dataset: Dataset, name: str = "checkthat_task2_finetune", test_size: float = 0.2, shuffle_seed: int = 0xABCD, batch_size: int = 250):
        """
        Run training on this model.
        """
        split_datasets = training_dataset.train_test_split(test_size=test_size, shuffle=True, seed=shuffle_seed)

        def train_preprocessor(ds):
            model_inputs = self.tokenizer(ds["input"], max_length=512, truncation=True)
            labels = self.tokenizer(ds["output"], max_length=512, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        tokenized_datasets = split_datasets.map(train_preprocessor, batched=True, batch_size=batch_size)
        collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.base_model_name)

        rouge = evaluate.load("rouge")

        def compute_metrics(plt):
            predictions, labels = plt
            decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            filtered_labels = numpy.where(labels != -100, labels, self.tokenizer.pad_token_type_id)
            decoded_labels = self.tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

            result = rouge.compute(predictions=decoded_predictions, references=decoded_labels, use_stemmer=True)
            result["gen_len"] = numpy.mean(
                [numpy.count_nonzero(pred != self.tokenizer.pad_token_type_id) for pred in predictions]
            )

            return {k: round(v, 4) for k, v in result.items()}

        train_args = Seq2SeqTrainingArguments(
            output_dir=name,
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_eval_batch_size=16,
            per_device_train_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=4,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=train_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            processing_class=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
    
    def query(self, _: str) -> str:
        """
        Query the LLM.
        """
        raise NotImplementedError()