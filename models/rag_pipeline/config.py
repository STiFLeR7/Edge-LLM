import torch

# Model path for quantized Qwen2.5-3B-GPTQ
MODEL_PATH = "D:/Edge-LLM/models/Qwen2.5-3B-GPTQ"

# ChromaDB storage path
CHROMA_DB_PATH = "./local_rag_db"

# Check for CUDA availability
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ✅ Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 enabled
torch.backends.cudnn.benchmark = True        # Optimized cuDNN kernel selection
