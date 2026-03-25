FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /models

RUN curl -L -o /models/bark_model.pt \
    https://huggingface.co/vvasylkovskyi/barkgpt/resolve/main/bark_model.pt

ENV MODEL_PATH=/models/bark_model.pt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]