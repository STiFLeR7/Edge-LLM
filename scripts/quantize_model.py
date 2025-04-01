from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch

# Convert Windows-style paths to POSIX (forward slashes)
model_path = Path("D:/Edge-LLM/models/Qwen2.5-3B").resolve()
quantized_model_path = Path("D:/Edge-LLM/models/Qwen2.5-3B-GPTQ").resolve()

# Load Model & Tokenizer with corrected path
tokenizer = AutoTokenizer.from_pretrained(model_path.as_posix(), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path.as_posix(), torch_dtype=torch.float16, trust_remote_code=True)

# Move the model to GPU (CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define Quantization Config
quant_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=True)

# Quantize Model
quantized_model = AutoGPTQForCausalLM.from_pretrained(model_path.as_posix(), quantize_config=quant_config)

# Move quantized model to GPU
quantized_model = quantized_model.to(device)

# Prepare example inputs for quantization (e.g., a batch of tokenized text)
example_inputs = tokenizer("Example input for quantization", return_tensors="pt").input_ids.to(device)

# Perform quantization with the provided examples
quantized_model.quantize(examples=example_inputs)

# Save Quantized Model
quantized_model.save_quantized(quantized_model_path.as_posix())
tokenizer.save_pretrained(quantized_model_path.as_posix())

print(f"✅ Quantization complete! Model saved at {quantized_model_path}")
