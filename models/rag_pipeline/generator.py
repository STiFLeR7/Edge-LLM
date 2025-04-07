import torch
from transformers import AutoTokenizer
from gptqmodel import GPTQModel
from config import MODEL_PATH, DEVICE

# Load tokenizer & quantized model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = GPTQModel.from_quantized(
    MODEL_PATH, 
    device=DEVICE, 
    use_safetensors=True, 
    use_triton=True  # ✅ Use Triton for faster execution
)

# ✅ Warm-up for reducing initial latency
dummy_prompt = "Hello"
inputs = tokenizer(dummy_prompt, return_tensors="pt").to(DEVICE)
_ = model.generate(**inputs, max_new_tokens=10)

def generate_response(prompt, max_tokens=150):  # ✅ Reduced max tokens
    """Generate text using Qwen2.5-3B-GPTQ."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=max_tokens)

    return tokenizer.decode(output[0], skip_special_tokens=True)
