import logging
import os
import sys
import ast
import csv
import json

import torch
from datasets import load_dataset, load_from_disk, Dataset
from peft import (
    LoraConfig, TaskType,
    get_peft_model, prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, TrainerCallback
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class CustomTensorBoardCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logging.info(f"Step {state.global_step}: {logs}")  # Print logs to console

class ModelTuner:
    def __init__(self, config):
        self.model = None
        self.config = config

        if not self.set_modelpaths():
            raise ValueError("paths for models is missing from the tuner config")

        if not self.set_dataset():
            raise ValueError("'dataset' is missing from the tuner config")

    def set_modelpaths(self) -> bool:
        if self.config.get("model_path", None) is None:
            logging.error("'model_path' is missing from the tuner config")
            return False
        self.model_path = self.config['model_path']

        if self.config.get("finetuned_model", None) is None:
            logging.error("'finetuned_model' is missing from the tuner config")
            return False
        self.finetuned_model = self.config['finetuned_model']
        return True

    def transform_scraped_data(self, data):
        jsonl_data = data + ".jsonl"
        csv_data = data + ".csv"

        with open(jsonl_data, "r", encoding="utf-8") as infile, open(csv_data, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)

            writer.writerow(['source', 'data'])

            for line in infile:
                record = json.loads(line.strip())
                data = str([record["input"], record["output"]])
                writer.writerow(["scraped", data])
        
        self.scraped_data = csv_data


    def set_dataset(self) -> bool:
        if self.config.get("dataset", None) is None:
            logging.error("'dataset' is missing from the tuner config")
            return False
        
        if self.config.get("tokenized_dataset", None) is None:
            logging.error("'tokenized_dataset' is missing from the tuner config")
            return False
        
        if self.config.get("scraped_data", None) is None:
            logging.error("'scraped_data' is missing from the tuner config")
            return False
        
        self.transform_scraped_data(self.config['scraped_data'])

        self.tokenized_dataset_path = self.config['tokenized_dataset']

        tbl_df = pd.read_csv(self.config['dataset'])
        scraped_df = pd.read_csv(self.scraped_data)

        dataset_df = pd.concat([tbl_df, scraped_df], ignore_index=True)

        dataset = Dataset.from_pandas(dataset_df)
        dataset = dataset.shuffle(seed=10)

        dataset = dataset.filter(lambda x: x["source"] in ["ultrachat", "scraped"])
        dataset = dataset.remove_columns(["combined", "filename", "data_type", "cause_clean", "bert_summary" ,"bert_topic", "topic_assignment", "source"])

        self.dataset = dataset.select(range(5000))
        self.eval_dataset = dataset.select(range(10000, 10500))

        return True

    def get_device_map(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def get_bnb_config(self) -> BitsAndBytesConfig:
        if self.get_device_map() == "cuda":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        return None

    def setup_tokenizer(self) -> AutoTokenizer:
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup_model(self):
        bnb_config = self.get_bnb_config()
        device = self.get_device_map()
   
        if device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto"
            )
        
        logging.info(f"Model : {self.model}")
    
    @staticmethod
    def _prompt_formatter(sample):
        try:
            sample_obj = ast.literal_eval(sample['data'])
        except Exception as e:
            logging.error(f"Error while literal evaluation of {sample['data']}")
            return ""

        bos_token = "<s>"
        eos_token = "</s>"
        system_message = "You are a travel planner and guide. Your name is Smart Traveler"
        full_prompt = ""
        full_prompt += bos_token
        full_prompt += "### Instruction:"
        full_prompt += "\n" + system_message
        full_prompt += "\n\n### User:"
        full_prompt += "\n" + sample_obj[0]
        full_prompt += "\n\n### Travel AI:"
        full_prompt += "\n" + sample_obj[1]
        full_prompt += eos_token

        return full_prompt

    def generate_and_tokenize_prompt(self, data):
        result = self.tokenizer(
            ModelTuner._prompt_formatter(data),
            truncation=True,
            max_length=512,
            padding="max_length"
        )

        result["labels"] = result["input_ids"].copy()
        return result

    def get_tokenized_dataset(self):
        tokenized_eval_dataset = self.eval_dataset.map(self.generate_and_tokenize_prompt)

        if os.path.exists(self.tokenized_dataset_path):  
            logging.info(f"Loading tokenized dataset from {self.tokenized_dataset_path}")
            return load_from_disk(self.tokenized_dataset_path), tokenized_eval_dataset
        
        tokenized_dataset = self.dataset.map(self.generate_and_tokenize_prompt)
        tokenized_dataset.save_to_disk(self.tokenized_dataset_path)

        return tokenized_dataset, tokenized_eval_dataset


    def setup_lora(self):
        logging.debug("Preparing model for kbit training!")
        self.model = prepare_model_for_kbit_training(model=self.model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            lora_dropout=0.1,  # Conventional
            task_type=TaskType.CAUSAL_LM,
        )

        logging.debug("Updating model with LoRA config")
        self.model = get_peft_model(model=self.model, peft_config=lora_config)

        def print_trainable_params():
            trainable_params = 0
            all_param = 0
            for _, param in self.model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            logging.info(f"trainable params: {trainable_params} || "
                f"all params: {all_param} || "
                f"trainable%: {100 * trainable_params / all_param}")
        print_trainable_params()
    
    def setup_trainer(self, tokenized_data, tokenized_eval_dataset):
        self.trainer = Trainer(
            model=self.model,
            train_dataset=tokenized_data,
            eval_dataset=tokenized_eval_dataset,
            args=TrainingArguments(
                output_dir=self.finetuned_model,
                warmup_steps=50,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                optim="adamw_torch",
                learning_rate=1e-5,        # Want a small lr for finetuning
                fp16=True,
                logging_steps=5,            # When to start reporting loss
                logging_dir="./logs",        # Directory for storing logs
                report_to=["tensorboard"],    # Enable TensorBoard logging
                save_strategy="steps",       # Save the model checkpoint every logging step
                save_steps=500,               # Save checkpoints every 50 steps
                eval_strategy="steps",       # Evaluate the model every logging step
                eval_steps=500,               # Evaluate and save checkpoints every 50 steps
                do_eval=True,                # Perform evaluation at the end of training
                push_to_hub=False,
            ),
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

    def plot_trainer_data(self):
        # Initialize lists
        train_loss = []
        eval_loss = []
        learning_rates = []
        steps = []
        perplexities = []

        for log in tuner.trainer.state.log_history:
            if "loss" in log:
                train_loss.append(log["loss"])
                steps.append(log["step"])
            if "eval_loss" in log:
                eval_loss.append(log["eval_loss"])
                perplexities.append(np.exp(log["eval_loss"]))  # Calculate perplexity
            if "learning_rate" in log:
                learning_rates.append(log["learning_rate"])

        # Plot Training Loss
        plt.figure(figsize=(10, 5))
        plt.plot(steps, train_loss, label="Training Loss", marker="o")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.grid()
        plt.savefig("training_loss.png")
        plt.show()

        # Plot Evaluation Loss
        if eval_loss:
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(eval_loss)), eval_loss, label="Evaluation Loss", marker="o", color="red")
            plt.xlabel("Evaluation Steps")
            plt.ylabel("Loss")
            plt.title("Evaluation Loss Over Time")
            plt.legend()
            plt.grid()
            plt.savefig("evaluation_loss.png")
            plt.show()

        # Plot Learning Rate
        if learning_rates:
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(learning_rates)), learning_rates, label="Learning Rate", marker="o", color="green")
            plt.xlabel("Steps")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.legend()
            plt.grid()
            plt.savefig("learning_rate.png")
            plt.show()

        # Plot Perplexity
        if perplexities:
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(perplexities)), perplexities, label="Perplexity", marker="o", color="purple")
            plt.xlabel("Evaluation Steps")
            plt.ylabel("Perplexity")
            plt.title("Model Perplexity Over Time")
            plt.legend()
            plt.grid()
            plt.savefig("perplexity.png")
            plt.show()


def init_logging():
    logging.basicConfig(
        handlers = [
            logging.FileHandler(filename="tuner.log"),
            logging.StreamHandler()
        ],
        level = logging.INFO,
        format = "%(asctime)s - %(levelname)s - %(message)s"
    )

if __name__ == "__main__":

    init_logging()

    tuner_config = {
        "model_path": "C:/Users/vedmp/smart-travel/fine-tuning/models/mistral",
        "finetuned_model": "C:/Users/vedmp/smart-travel/fine-tuning/models/mistral-finetuned-test",
        "dataset": "C:/Users/vedmp/smart-travel/fine-tuning/data/travel_convo.csv",
        "scraped_data": "C:/Users/vedmp/smart-travel/fine-tuning/data/scraped_data",
        "tokenized_dataset": "C:/Users/vedmp/smart-travel/fine-tuning/data/tokenized_data"
    }
    tuner = ModelTuner(config=tuner_config)

    if tuner is None:
        logging.fatal(f"Error while initializing ModelTuner")
        sys.exit(0)

    tuner.setup_tokenizer()
    tuner.setup_model()

    tokenized_dataset, tokenized_eval_dataset = tuner.get_tokenized_dataset()
    logging.info(f"Tokenized Dataset : {tokenized_dataset}")
    
    tuner.setup_lora()
    logging.info(f"Model after setting up LoRA : {tuner.model}")

    tuner.setup_trainer(tokenized_dataset, tokenized_eval_dataset)

    tuner.model.config.use_cache = False

    logging.info("Starting model training!")
    try:
        tuner.trainer.train()
    except Exception as e:
        logging.fatal(f"Failure during training : {e}")

    logging.info(f"Saving fine tuned model at {tuner.finetuned_model}")
    tuner.model.save_pretrained(tuner.finetuned_model)
    logging.info(f"Saving tokens for fine tuned model at {tuner.finetuned_model}")
    tuner.tokenizer.save_pretrained(tuner.finetuned_model)

    tuner.plot_trainer_data()