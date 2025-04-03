from pathlib import Path
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig  # ✅ Updated Import
import torch

# Define Paths
MODEL_PATH = Path("/workspace/quantization/models/Qwen2.5-3B").resolve()
QUANTIZED_MODEL_PATH = Path("/workspace/quantization/models/Qwen2.5-3B-GPTQ").resolve()

# Load Tokenizer
# Load Tokenizer (FIX: Set local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH.as_posix(), 
    trust_remote_code=True, 
    local_files_only=True  # ✅ Force loading from local directory
)

# Define Quantization Config using GPTQModel
quant_config = QuantizeConfig(
    bits=4,              # 4-bit quantization
    group_size=128,      # Group size
    desc_act=True        # Act-order (helps with accuracy)
)

# Load Model with Quantization Config
model = GPTQModel.from_pretrained(
    MODEL_PATH.as_posix(), 
    quantize_config=quant_config, 
    device_map="cuda"
)

# Example Inputs
example_texts = ["Hello, how are you?", "This is a test sentence for quantization."]
example_inputs = tokenizer(example_texts, padding=True, truncation=True, return_tensors="pt")

# Move inputs to CUDA
example_inputs = {k: v.to("cuda") for k, v in example_inputs.items()}

# Run Quantization
input_ids_list = example_inputs["input_ids"].tolist()  # Convert tensor to list
model.quantize(input_ids_list)  # Pass as a list

# Save Quantized Model
model.save_quantized(QUANTIZED_MODEL_PATH.as_posix())
tokenizer.save_pretrained(QUANTIZED_MODEL_PATH.as_posix())

print(f"✅ Quantization complete! Model saved at {QUANTIZED_MODEL_PATH}")
