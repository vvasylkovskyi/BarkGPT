.PHONY: clean install dev-install lint format check

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Install dependencies
install:
	@echo "Installing dependencies..."
	uv sync

# Run linting checks
lint:
	@echo "Running linting checks..."
	uv run python -m ruff check .

lock:
	@echo "Locking dependencies..."
	uv lock

# Format code
format:
	@echo "Formatting code..."
	uv run python -m ruff format .

# Run all checks
check: lint test-coverage
	@echo "All checks completed!"

	.PHONY: run
run:
	@echo "Starting the development server..."
	uv run uvicorn app.main:app --host 0.0.0.0 --port 80 --reload

prepare_dataset:
	@echo "Preparing dataset..."
	uv run python bark_gpt/train/prepare_dataset.py

train: install
	@echo "Training model..."
	uv run python -m bark_gpt.train.train

test: train test_model
	
test_model:
	@echo "Testing model..."
	uv run python -m bark_gpt.test.test_pytorch_gen

test_hf_model:
	@echo "Testing Hugging Face integration..."
	uv run python -m bark_gpt.test.test_hf_gen

build_hf_model:
	@echo "Building Hugging Face model..."
	uv run python -m bark_gpt.build_hf_model

test_model_hf:
	@echo "Testing Hugging Face integration..."
	uv run python -m bark_gpt.test_model_hf

train_rnn: install
	@echo "Training RNN model..."
	uv run python -m bark_rnn.train.train

test_rnn_model:
	@echo "Testing RNN model..."
	uv run python -m bark_rnn.test.test

load_datasets:
	@echo "Loading datasets..."
	uv run python datasets/load_dataset.py

train_bark_gpt_2:
	@echo "Training BarkGPT 2 model..."
	uv run python -m bark_gpt_2.train.train

test_bark_gpt_2:
	@echo "Testing BarkGPT 2 model..."
	uv run python -m bark_gpt_2.test.test_gen