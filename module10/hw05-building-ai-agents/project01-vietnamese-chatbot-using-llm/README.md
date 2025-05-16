# AIO2024 Chatbot LLMs

This repository contains resources, scripts, and notebooks for training, fine-tuning, and evaluating large language models (LLMs) with Reinforcement Learning from Human Feedback (RLHF), using the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) framework.

## Features

- End-to-end RLHF pipeline with support for SFT, DPO, PPO, REINFORCE++, and more.
- Integration with [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for scalable, distributed training (Ray, vLLM, ZeRO-3, HuggingFace Transformers).
- Example scripts and notebooks for:
  - Supervised fine-tuning (SFT)
  - Direct Preference Optimization (DPO)
  - Reward model training
  - PPO/RLHF with Ray and vLLM
  - Model conversion (e.g., HuggingFace to GGUF)
- Utilities for pushing models to Hugging Face Hub and model conversion.

## Repository Structure

- `OpenRLHF/` – OpenRLHF framework (submodule or local copy)
- `notebooks/` – Example Jupyter notebooks for training, evaluation, and conversion
- `openwebui_ollama/` – Integration with Ollama LLMs and Open WebUI
- `gradio_vllm/` – Integration with Gradio and vLLM

## Quick Start

1. **Install dependencies**  
   See [OpenRLHF/README.md](OpenRLHF/README.md) for detailed installation instructions.

2. **Run example notebooks**  
   Explore the `notebooks/` directory for end-to-end RLHF workflows.

3. **Train your own model**  
   Use the provided scripts or adapt the notebooks for your dataset and model.

## Documentation

- [OpenRLHF Documentation](https://openrlhf.readthedocs.io/)
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)
