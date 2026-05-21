FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy pyproject.toml and install dependencies with uv
COPY pyproject.toml ./

# Install dependencies with uv, using unsafe-best-match to handle PyTorch CPU index
RUN uv pip install --system --no-cache --index-strategy unsafe-best-match . --extra-index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /models

# RUN curl -L -o /models/bark_model.pt \
#     https://huggingface.co/vvasylkovskyi/barkgpt/resolve/main/bark_model.pt

ENV MODEL_PATH=/models/bark_model.pt

# Copy remaining files
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9999"]