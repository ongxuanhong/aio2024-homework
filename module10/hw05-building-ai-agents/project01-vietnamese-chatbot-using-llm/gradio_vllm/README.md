# Gradio and VLLM Serving for Vietnamese LLM

## Installation

```bash
conda create -n rlhf-serve python=3.12 --y
conda activate rlhf-serve
pip install -r requirements.txt
cd gradio_vllm
```

## Usage

### Starting the vLLM server

```bash
vllm serve thuanan/Llama-3.2-1B-RLHF-2k-vi-alpaca \
  --api-key aio2025 \
  --compilation-config '{"cache_dir": "../cache"}' \
  --port 8000 \
  --quantization bitsandbytes \
  --enable-prefix-caching \
  --swap-space 16 \
  --gpu-memory-utilization 0.9 \
  --disable-log-requests \
  --max-model-len 2048
```

### Running the Gradio Interface

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:7860` to interact with the chatbot.
