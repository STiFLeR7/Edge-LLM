from pathlib import Path
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch

# Define Paths
MODEL_PATH = Path("models/Qwen2.5-3B").resolve()
QUANTIZED_MODEL_PATH = Path("models/Qwen2.5-3B-GPTQ").resolve()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH.as_posix(), trust_remote_code=True)

# Define Quantization Config
quant_config = BaseQuantizeConfig(
    bits=4,               # 4-bit quantization
    group_size=128,       # Group size
    desc_act=True         # Act-order
)

# Load Model with Quantization Config
model = AutoGPTQForCausalLM.from_pretrained(
    MODEL_PATH.as_posix(), 
    quantize_config=quant_config, 
    device_map="cuda"
)

# Run Quantization
model.quantize()

# Save Quantized Model
model.save_quantized(QUANTIZED_MODEL_PATH.as_posix())
tokenizer.save_pretrained(QUANTIZED_MODEL_PATH.as_posix())

print(f"✅ Quantization complete! Model saved at {QUANTIZED_MODEL_PATH}")
