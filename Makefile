# Makefile for Industrial IoT Anomaly Detection
# Author: Lucas William Junges
# Date: December 2024

.PHONY: help install test docker-build docker-run clean train-nasa train-synthetic api lint format

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

help: ## Display this help message
	@echo "$(BLUE)Industrial IoT Anomaly Detection - Makefile$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: install ## Install development dependencies
	@echo "$(BLUE)Installing dev dependencies...$(NC)"
	pip install black flake8 isort mypy pylint
	@echo "$(GREEN)✓ Dev dependencies installed$(NC)"

##@ Training

train-nasa: ## Train models on NASA bearing data (RECOMMENDED)
	@echo "$(BLUE)Training on NASA bearing dataset...$(NC)"
	python train_nasa.py
	@echo "$(GREEN)✓ NASA models trained$(NC)"

train-synthetic: ## Train models on synthetic data
	@echo "$(BLUE)Training on synthetic data...$(NC)"
	python train_simple.py
	@echo "$(GREEN)✓ Synthetic models trained$(NC)"

##@ Evaluation

evaluate-nasa: ## Evaluate NASA-trained models
	@echo "$(BLUE)Evaluating NASA models...$(NC)"
	python evaluate_nasa.py
	@echo "$(GREEN)✓ Evaluation complete$(NC)"

evaluate-synthetic: ## Evaluate synthetic-trained models
	@echo "$(BLUE)Evaluating synthetic models...$(NC)"
	python evaluate_simple.py
	@echo "$(GREEN)✓ Evaluation complete$(NC)"

##@ Testing

test: ## Run test suite with coverage
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Tests passed$(NC)"

test-fast: ## Run tests without coverage (faster)
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests passed$(NC)"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	pytest-watch tests/ -v

coverage: ## Generate coverage report
	@echo "$(BLUE)Generating coverage report...$(NC)"
	pytest tests/ --cov=src --cov-report=html
	@echo "$(GREEN)✓ Coverage report: htmlcov/index.html$(NC)"

##@ Code Quality

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	flake8 src/ tests/ api/ || true
	pylint src/ tests/ api/ || true
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/ api/
	isort src/ tests/ api/
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check code formatting without changing
	@echo "$(BLUE)Checking code format...$(NC)"
	black --check src/ tests/ api/
	isort --check-only src/ tests/ api/
	@echo "$(GREEN)✓ Format check complete$(NC)"

##@ Docker

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run training in Docker
	@echo "$(BLUE)Running training in Docker...$(NC)"
	docker-compose run anomaly-detection python train_nasa.py
	@echo "$(GREEN)✓ Docker training complete$(NC)"

docker-test: ## Run tests in Docker
	@echo "$(BLUE)Running tests in Docker...$(NC)"
	docker-compose run anomaly-detection pytest tests/ -v
	@echo "$(GREEN)✓ Docker tests complete$(NC)"

docker-shell: ## Open shell in Docker container
	@echo "$(BLUE)Opening Docker shell...$(NC)"
	docker-compose run anomaly-detection /bin/bash

docker-jupyter: ## Start Jupyter in Docker
	@echo "$(BLUE)Starting Jupyter in Docker...$(NC)"
	@echo "$(YELLOW)Access at: http://localhost:8888$(NC)"
	docker-compose up jupyter

docker-clean: ## Remove Docker images and containers
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker-compose down
	docker system prune -f
	@echo "$(GREEN)✓ Docker cleaned$(NC)"

##@ API

api: ## Start FastAPI server
	@echo "$(BLUE)Starting FastAPI server...$(NC)"
	@echo "$(YELLOW)Access docs at: http://localhost:8000/docs$(NC)"
	cd api && python main.py

api-docker: ## Start API in Docker
	@echo "$(BLUE)Starting API in Docker...$(NC)"
	@echo "$(YELLOW)Access docs at: http://localhost:8000/docs$(NC)"
	docker-compose up api

##@ Examples

example: ## Run inference example
	@echo "$(BLUE)Running inference example...$(NC)"
	python examples/inference_example.py
	@echo "$(GREEN)✓ Example complete$(NC)"

demo: train-nasa example api ## Full demo: train → example → API

##@ Cleanup

clean: ## Clean generated files
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-models: ## Remove trained models
	@echo "$(RED)Removing trained models...$(NC)"
	rm -rf models/*.pkl
	rm -rf models/*/
	@echo "$(GREEN)✓ Models removed$(NC)"

clean-data: ## Remove generated data
	@echo "$(RED)Removing generated data...$(NC)"
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.csv
	@echo "$(GREEN)✓ Data removed$(NC)"

clean-all: clean clean-models clean-data docker-clean ## Clean everything
	@echo "$(GREEN)✓ Complete cleanup done$(NC)"

##@ Git

git-status: ## Show git status
	@git status

git-commit: ## Commit with message (use MSG="your message")
	@if [ -z "$(MSG)" ]; then \
		echo "$(RED)Error: Please provide commit message with MSG='...'$(NC)"; \
		exit 1; \
	fi
	@git add -A
	@git commit -m "$(MSG)"
	@echo "$(GREEN)✓ Committed: $(MSG)$(NC)"

git-push: ## Push to GitHub
	@git push origin main
	@echo "$(GREEN)✓ Pushed to GitHub$(NC)"

##@ CI/CD

ci: lint test docker-build ## Run full CI pipeline locally
	@echo "$(GREEN)✓ CI pipeline passed$(NC)"

##@ Info

info: ## Show project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo ""
	@echo "$(GREEN)Repository:$(NC) /home/lucas-junges/Documents/material_estudo/projetos/projeto 2"
	@echo "$(GREEN)Python version:$(NC) $(shell python --version)"
	@echo "$(GREEN)Git branch:$(NC) $(shell git branch --show-current)"
	@echo "$(GREEN)Last commit:$(NC) $(shell git log -1 --pretty=format:'%h - %s')"
	@echo ""
	@echo "$(GREEN)Trained models:$(NC)"
	@ls -lh models/ 2>/dev/null | grep -v total || echo "  No models found"
	@echo ""
	@echo "$(GREEN)Generated data:$(NC)"
	@du -sh data/ 2>/dev/null || echo "  No data found"
	@echo ""
