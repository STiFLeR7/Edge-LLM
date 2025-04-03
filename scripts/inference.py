import torch
from transformers import AutoTokenizer
from gptqmodel import GPTQModel

# Define model path
model_path = "D:/Edge-LLM/models/Qwen2.5-3B-GPTQ"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Check if CUDA is available, else use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = GPTQModel.from_quantized(model_path, device=device, use_safetensors=True)

def generate_response(prompt, max_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_tokens)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test the function
query = "What is retrieval-augmented generation?"
print(generate_response(query))
