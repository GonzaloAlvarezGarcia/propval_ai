# Makefile for PropVal AI Project

# Variables
PYTHON = python
PIP = pip
UVICORN = uvicorn
APP_MODULE = src.app:app

.PHONY: install train run clean docker-build

# 1. Installation
install:
	$(PIP) install -r requirements.txt

# 2. Training Pipeline
train:
	$(PYTHON) src/train_pipeline.py

# 3. Run API locally
run:
	$(UVICORN) $(APP_MODULE) --reload --host 0.0.0.0 --port 8000

# 4. Docker Build & Run (Test container locally)
docker-build:
	docker build -t propval-ai .

docker-run:
	docker run -p 8000:8000 propval-ai

# 5. Clean up junk files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf venv