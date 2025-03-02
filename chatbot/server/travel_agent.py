import logging
import torch
from queue import Queue
from threading import Thread

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

class TravelAgent:
    def __init__(self, config):
        self.model = None
        self.tokenizer = None

        self.config = config

        if not self.set_modelpaths():
            raise ValueError("paths for models is missing from the tuner config")
        
        self.device = TravelAgent._set_device()
        self.load_tokenizers()
        self.load_model()
        self.load_base_model()


    def set_modelpaths(self) -> bool:
        if self.config.get("model_path", None) is None:
            logging.error("'model_path' is missing from the tuner config")
            return False
        self.model_path = self.config['model_path']

        if self.config.get("base_model_path", None) is None:
            logging.error("'base_model_path' is missing from the tuner config")
            return False
        self.base_model_path = self.config['base_model_path']
        return True
    
    @staticmethod
    def _set_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_tokenizers(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
    
    def load_model(self):
        print("Loading finetuned model")
        if self.device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map={"": self.device},
                use_cache=False
            )
        
        self.model.to(self.device)
        # Move the model to inference state. Without this it is still in training mode
        self.model.eval()
    
    def load_base_model(self):
        print("Loading base model")
        if self.device == "mps":
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map="auto"
            )
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # Change from bf16 to fp16
                bnb_4bit_use_double_quant=True,  # Secondary quantization for efficiency
                bnb_4bit_quant_type="nf4"
            )

            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map={"": self.device},
                quantization_config=quant_config,
                use_cache=False
            )
        
        self.base_model.to(self.device)
        # Move the model to inference state. Without this it is still in training mode
        self.base_model.eval()
    
    def _generate(self, model, tokenizer, prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = model.generate(**inputs, max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response
    
    def _generate_stream(self, model, tokenizer, streamer, prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=500)

        generator = Thread(target=model.generate, kwargs=generation_kwargs)
        generator.start()


    def generate(self, prompt, stream=False):
        if not stream:
            return self._generate(self.model, self.tokenizer, prompt)
        streamer = TextIteratorStreamer(self.tokenizer)
        self._generate_stream(self.model, self.tokenizer, streamer, prompt)
        return streamer


    def generate_base(self, prompt, streamer=None):
        if not streamer:
            return self._generate(self.base_model, self.base_tokenizer, prompt)
        streamer = TextIteratorStreamer(self.tokenizer)
        self._generate_stream(self.model, self.tokenizer, streamer, prompt)
        return streamer
