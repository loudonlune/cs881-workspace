
from random import shuffle
import tqdm
from tool.llm.base import LLMBackend

from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, default_data_collator, get_linear_schedule_with_warmup
from datasets import Dataset
import evaluate
import numpy
import torch

from peft import TaskType, get_peft_model, PromptTuningConfig, PromptTuningInit, AutoPeftModelForCausalLM

from typing import List, Optional

class TrainedLocalLLMBackend(LLMBackend):
    base_model_name: str
    local_model_name: str
    system_prompt: str
    training_mode: bool

    def __init__(self, base_model_name: str, system_prompt: str = "Extract the verifiable claim as one sentence from the user input.", local_model_name: str = "checkthat_task2_finetune"):
        print("Base model:", base_model_name)
        print("Local model:", local_model_name)
        self.base_model_name = base_model_name
        self.local_model_name = local_model_name
        self.system_prompt = system_prompt
        self.training_mode = False

    def initialize(self, use_flash: bool = False, use_4bit_quant: bool = True, skip_train: bool = False, additional_model_args: dict = {}):
        if use_flash:
            additional_model_args['attn_implementation'] = 'flash_attention_2'

        if use_4bit_quant:
            additional_model_args['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_storage=torch.bfloat16,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        # Always use the same tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.training_mode = not skip_train

        # Load the trained model if we're not training.
        if skip_train:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.local_model_name,
                device_map="auto",
                **additional_model_args
            )
        # Otherwise, load the base model.
        else:
            pt_cfg = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=16,
                prompt_tuning_init_text=self.system_prompt,
                tokenizer_name_or_path=self.base_model_name,
            )


            self.model = get_peft_model(AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map="auto",
                **additional_model_args,
            ), pt_cfg)
    
    # Inspiration taken from here: https://huggingface.co/spaces/PEFT/causal-language-modeling/blob/main/prompt_tuning_clm.ipynb
    # Modified this a bit and added some comments to guide my understanding.
    def train(self, training_dataset: Dataset,
              cap: Optional[int] = None,
              num_epochs: int = 5,
              test_size: float = 0.2,
              shuffle_seed: int = 0xABCD,
              batch_size: int = 8,
              warmup_steps: int = 10,
              learning_rate: float = 0.0001,
              weight_decay: float = 0.01,
              max_length: int = 512):
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
                {"system": self.system_prompt, "user": input},
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
            output_dir=self.local_model_name,
            num_train_epochs=num_epochs,
            save_total_limit=5,
            per_device_train_batch_size=8,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            dataloader_drop_last=True,
            bf16=True,
            logging_steps=10,
            learning_rate=learning_rate,
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

        trainer.train()
        trainer.save_model(self.local_model_name)
    
    def query(self, prompt: str, temperature: float = 0.0) -> str:
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
            temperature=temperature,
        )

        output_text = self.tokenizer.batch_decode(output_embedding, skip_special_tokens=True)[0]

        return output_text.split('</think>')[-1].strip()
