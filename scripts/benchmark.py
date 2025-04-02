import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

# Path to quantized model
quantized_model_path = "/workspace/quantization/models/Qwen2.5-3B-GPTQ"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, trust_remote_code=True)

# Load model
model = AutoGPTQForCausalLM.from_quantized(quantized_model_path, device="cuda" if torch.cuda.is_available() else "cpu")

# Test prompt
prompt = "What is the capital of France?"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Benchmark inference time
start_time = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
end_time = time.time()

# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print results
print(f"Response: {response}")
print(f"Inference Time: {end_time - start_time:.4f} seconds")
