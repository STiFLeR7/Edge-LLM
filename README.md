
# Edge-LLM: Optimized Qwen2.5-3B with GPTQ

## 📌 Overview
Edge-LLM is a lightweight and optimized deployment of Qwen2.5-3B, quantized using GPTQ for efficient inference on edge devices. This project achieves significant size reduction and improved inference speed while maintaining high-quality responses.

## 🔥 Key Highlights
Model: Qwen2.5-3B

Quantization Method: GPTQ (4-bit)

Size Reduction: 🔽 66.5% (5.75GB ➝ 1.93GB)

Inference Speed Improvement: ⏳ Reduced inference time from 7.29s to 5.99s

Optimized for Edge Devices: Runs efficiently on consumer GPUs

## 🚀 Setup & Installation
1. Clone the Repository: 
```
git clone https://github.com/STiFLeR7/Edge-LLM.git
cd Edge-LLM
```
2. Create a virtual environment and install dependencies:
```
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```
3. Download the quantized model:
```
git lfs install
git clone https://huggingface.co/<your-hf-repo> models/Qwen2.5-3B-GPTQ
```

## 📊 Benchmark Results
Metric | Pre-Quantization | Post-Quantization
--- | --- | ---
Model Size | 5.75GB | 1.93GB
Inference Time | 7.29s | 5.99s (18% faster)

## 🏃 Running Inference
Use the optimized script to test the quantized model:
```
python scripts/benchmark.py
```
Expected Output:
```
🔹 **Generated Response:**
Black holes are regions of space where gravity is so strong that nothing, not even light, can escape...
⏳ Inference Time: ~5.99s
```

## 📢 Contributing
Feel free to open issues or submit pull requests to improve Edge-LLM!

## 📜 License
This project is licensed under the MIT License.