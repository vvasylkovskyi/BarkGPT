FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    torch \
    transformers

# Create model directory
RUN mkdir -p /models

# Download BarkGPT model from Hugging Face
RUN curl -L -o /models/bark_model.pt \
    https://huggingface.co/vvasylkovskyi/barkgpt/resolve/main/bark_model.pt

# Expose model path
ENV MODEL_PATH=/models/bark_model.pt

# Copy app code
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]