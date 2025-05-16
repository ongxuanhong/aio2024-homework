# Ollama LLM Integration

This repository contains tools and configurations for running LLM models with Ollama in Docker.

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed on your system
- At least 8GB RAM for running smaller models

### Installation

1. Start the Ollama container:

    ```bash
    docker compose up -d
    ```

2. Access the Ollama container shell:

    ```bash
    docker exec -it ollama bash
    ```

## Using Models

### Running a Fine-tuned Model

To run a specific fine-tuned model:

```bash
ollama run llama3.2:1b
ollama run llama3.2:1b-instruct-fp16
ollama run hf.co/thuanan/Llama-3.2-1B-Instruct-Chat-sft-gguf
ollama run hf.co/thuanan/Llama-3.2-1B-RLHF-2k-vi-alpaca-gguf
```

### Other Useful Commands

#### List available models

```bash
ollama list
```

#### Pull a model

```bash
ollama pull llama3
```

#### Remove a model

```bash
ollama rm <model-name>
```

## Troubleshooting

### Common Issues

- If you encounter memory errors, try using a smaller model or increasing Docker's memory allocation
- Connection issues may require restarting the Docker container

### Logs

View container logs with:

```bash
docker logs ollama
```

## References

- [Ollama Official Documentation](https://ollama.com/documentation)
- [Hugging Face Model Repository](https://huggingface.co/)
