from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import logging

class TravelAgent:
    def __init__(self, config):
        self.model = None
        self.tokenizer = None

        self.config = config

        if not self.set_modelpaths():
            raise ValueError("paths for models is missing from the tuner config")
        
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
    
    def get_device_map(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_tokenizers(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
    
    def load_model(self):
        device = self.get_device_map()
        print("Loading finetuned model")
        if device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map={"": device},
                use_cache=False
            )
        
        self.model.to(device)
        # Move the model to inference state. Without this it is still in training mode
        self.model.eval()
    
    def load_base_model(self):
        device = self.get_device_map()
        print("Loading base model")
        if device == "mps":
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
                device_map={"": device},
                quantization_config=quant_config,
                use_cache=False
            )
        
        self.base_model.to(device)
        # Move the model to inference state. Without this it is still in training mode
        self.base_model.eval()
    
    def generate(self, prompt):
        device = self.get_device_map()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs, max_length=750)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response
    
    def generate_base(self, prompt):
        device = self.get_device_map()
        inputs = self.base_tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.base_model.generate(**inputs, max_length=750)
        response = self.base_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response