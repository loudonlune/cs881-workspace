
from random import shuffle
import tqdm
from tool.llm.base import LLMBackend

from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, default_data_collator, get_linear_schedule_with_warmup
from datasets import Dataset
import evaluate
import numpy
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW

from peft import LoraConfig, TaskType, get_peft_model

from typing import List, Optional

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
        self.model = get_peft_model(AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            **additional_model_args,
        ), lora_cfg)
    
    # Inspiration taken from here: https://huggingface.co/spaces/PEFT/causal-language-modeling/blob/main/prompt_tuning_clm.ipynb
    # Modified this a bit and added some comments to guide my understanding.
    def train(self, training_dataset: Dataset,
              name: str = "checkthat_task2_finetune",
              cap: Optional[int] = None,
              num_epochs: int = 5,
              test_size: float = 0.2,
              shuffle_seed: int = 0xABCD,
              batch_size: int = 16,
              max_length: int = 128):
        """
        Run training on this model.
        """
        if cap is not None and cap > 0:
                training_dataset = training_dataset.select(range(cap))

        if len(training_dataset) * test_size < 35.0:
            print("WARNING: There is not much data in this dataset. The resulting model will be poor.")

        split_datasets = training_dataset.train_test_split(test_size=test_size, shuffle=True, seed=shuffle_seed)

        def dataset_preprocessor(ds):
            # Get the size of the current batch.
            current_batch_size: int = len(ds["input"])
            # Apply chat template to the inputs.
            inputs: List = [ self.tokenizer.apply_chat_template(
                {"system": "Extract the verifiable claim as one sentence from the user input.", "user": input},
                tokenize=False,
            ) for input in ds["input"] ]

            # Tokenize the inputs.
            model_inputs = self.tokenizer(inputs)

            # Tokenize the desired outputs.
            labels = self.tokenizer(ds["output"], add_special_tokens=False)

            # Process the embeddings for each input and label.
            for i in range(current_batch_size):
                input_eids = model_inputs["input_ids"][i]
                input_elen = len(input_eids)
                label_eids = labels["input_ids"][i]
                label_elen = len(label_eids)

                model_inputs["input_ids"][i] = torch.tensor(
                    (
                        ([self.tokenizer.pad_token_type_id] * (max_length - input_elen)) + input_eids
                    )[:max_length]
                )

                model_inputs["attention_mask"][i] = torch.tensor(
                    (
                        ([0] * (max_length * input_elen)) + model_inputs["attention_mask"][i]
                    )[:max_length]
                )

                labels["input_ids"][i] = torch.tensor(
                    (
                        ([-100] * (max_length - label_elen)) + labels["input_ids"][i]
                    )[:max_length]
                )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Create tokenized datasets.
        train_ds = split_datasets["train"].map(
            dataset_preprocessor,
            batched=True,
            num_proc=1,
            batch_size=batch_size,
            remove_columns=split_datasets["train"].column_names,
            load_from_cache_file=False,
            desc="Tokenizing the datasets."
        )

        eval_ds = split_datasets["test"].map(
            dataset_preprocessor,
            batched=True,
            num_proc=1,
            batch_size=batch_size,
            remove_columns=split_datasets["test"].column_names,
            load_from_cache_file=False,
            desc="Tokenizing the datasets."
        )

        # def test_preprocessor(ds):
        #     # Get the size of the current batch.
        #     current_batch_size: int = len(ds["input"])
        #     # Apply chat template to the inputs.
        #     inputs: List = [ self.tokenizer.apply_chat_template(
        #         {"system": "Extract the verifiable claim as one sentence from the user input.", "user": input},
        #         tokenize=False,
        #     ) for input in ds["input"] ]

        #     # Tokenize the inputs.
        #     model_inputs = self.tokenizer(inputs)

        #     # Process the embeddings for each input and label.
        #     for i in range(current_batch_size):
        #         input_eids = model_inputs["input_ids"][i]
        #         input_elen = len(input_eids)

        #         model_inputs["input_ids"][i] = torch.tensor(
        #             (
        #                 ([self.tokenizer.pad_token_type_id] * (max_length - input_elen)) + input_eids
        #             )[:max_length]
        #         )

        #         model_inputs["attention_mask"][i] = torch.tensor(
        #             (
        #                 ([0] * (max_length * input_elen)) + model_inputs["attention_mask"][i]
        #             )[:max_length]
        #         )
        #     return model_inputs
        
        # test_ds = split_datasets["test"]

        # test_dataloader = test_ds.map(
        #     test_preprocessor,
        #     batched=True,
        #     num_proc=1,
        #     batch_size=batch_size,
        #     remove_columns=train_ds.column_names,
        #     load_from_cache_file=False,
        #     desc="Tokenizing the test dataset."
        # )

        # print(f"Test dataset length: {len(test_dataloader)}")
        # print(f"First row: {next(iter(test_dataloader))}")

        trainp, allp = self.model.get_nb_trainable_parameters()
        print(f"Model trainable params: {trainp}")
        print(f"Model all parameters: {allp}")

        # rouge = evaluate.load("rouge")

        # def compute_metrics(plt):
        #     predictions, labels = plt
        #     decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        #     filtered_labels = numpy.where(labels != -100, labels, self.tokenizer.pad_token_type_id)
        #     decoded_labels = self.tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

        #     result = rouge.compute(predictions=decoded_predictions, references=decoded_labels, use_stemmer=True)
        #     result["gen_len"] = numpy.mean(
        #         [numpy.count_nonzero(pred != self.tokenizer.pad_token_type_id) for pred in predictions]
        #     )

        #     return {k: round(v, 4) for k, v in result.items()}

        trainargs = TrainingArguments(
            output_dir=name,
            num_train_epochs=num_epochs,
            save_total_limit=5,
            per_device_train_batch_size=8,
            warmup_steps=10,
            weight_decay=0.01,
            dataloader_drop_last=True,
            bf16=True,
            logging_steps=10,
            learning_rate=0.0001,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={ "use_reentrant": False },
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=trainargs,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=default_data_collator,
        )

        opt = trainer.train()

        print(opt)


    
    def query(self, _: str) -> str:
        """
        Query the LLM.
        """
        raise NotImplementedError()