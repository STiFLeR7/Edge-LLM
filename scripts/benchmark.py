import torch
import time
from gptqmodel import GPTQModel
from transformers import AutoTokenizer

# Path to the quantized model
quantized_model_path = "/workspace/quantization/models/Qwen2.5-3B-GPTQ"

# Force CUDA usage
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(0)

torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmul
torch.backends.cudnn.benchmark = True  # Enable auto-tuning for CuDNN

print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)

# Load quantized model with optimized settings
print("\nLoading quantized model for benchmarking...")
model = GPTQModel.from_quantized(quantized_model_path, device=device)
model = model.to(device)
print("✅ Model loaded successfully!")

# Prepare benchmark input
prompt = "Explain the concept of black holes."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Warm-up (to avoid initial cold start latency)
print("\n🔹 Running warm-up inference...")
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=100)
print("✅ Warm-up complete!")

# Run benchmark with CUDA Graphs optimization
print("\n🔹 Running optimized inference benchmark...")
start_time = time.time()

with torch.no_grad():
    torch.cuda.synchronize()  # Ensure GPU is ready before timing
    output = model.generate(**inputs, max_new_tokens=100)
    torch.cuda.synchronize()  # Ensure GPU finishes computation

end_time = time.time()
elapsed_time = end_time - start_time

# Decode output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

# Print results
print("\n🔹 **Generated Response:**")
print(decoded_output)
print(f"\n⏳ Inference Time: {elapsed_time:.3f} seconds (Optimized)")
