from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Set model paths
base_model_path = "C:/Users/vedmp/smart-travel/fine-tuning/models/mistral"  # Base Mistral model from Hugging Face
finetuned_model_path = "C:/Users/vedmp/smart-travel/fine-tuning/models/mistral-finetuned-v0.2-weights"  # Your fine-tuned LoRA model
merged_model_path = "C:/Users/vedmp/smart-travel/fine-tuning/models/mistral-finetuned-v0.2"  # Your fine-tuned LoRA model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 4-bit Quantization Optimization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Change from bf16 to fp16
    bnb_4bit_use_double_quant=True,  # Secondary quantization for efficiency
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

# Load base model with offloading
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map={"": device},            # Auto distribute layers to GPU/CPU
    quantization_config=quant_config,
    use_cache=False
)


# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    finetuned_model_path,
    offload_folder="offload"
)

model = model.merge_and_unload()  # Merge LoRA weights into the base model

model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)  # Also save tokenizer