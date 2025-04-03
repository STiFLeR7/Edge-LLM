import torch
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

# Path to the quantized model
quantized_model_path = "/workspace/quantization/models/Qwen2.5-3B-GPTQ"

# Force CUDA usage
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(0)

print(f"Using device: {device}")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)

# Load the quantized model
print("\nLoading quantized model...")
model = GPTQModel.from_quantized(quantized_model_path, device=device)
print("✅ Model loaded successfully!")

# Prepare input prompt
prompt = "Explain quantum entanglement in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response
print("\nGenerating response...")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)

# Decode output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n🔹 **Generated Response:**")
print(decoded_output)
