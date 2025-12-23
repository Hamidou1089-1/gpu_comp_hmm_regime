# ==============================================================================
# Makefile Wrapper for GPU HMM Regime Detection
# ==============================================================================


.PHONY: help build build-cuda clean test test-cpu test-gpu benchmark info data venv cluster-submit cluster-status
.DEFAULT_GOAL := help

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# Configuration
BUILD_DIR := build
BUILD_TYPE := Release

help: ## Show this help message
	@echo "$(BLUE)GPU HMM Regime Detection - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ==============================================================================
# Build Targets
# ==============================================================================

build: ## Build project (CPU only)
	@echo "$(BLUE)Building project (CPU only)...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DBUILD_CUDA=OFF \
		-DBUILD_TESTS_CPU=ON \
		-DBUILD_TESTS_CUDA=OFF \
		.. && make -j$$(nproc)
	@echo "$(GREEN)✓ Build complete!$(NC)"

build-cuda: ## Build project with CUDA support
	@echo "$(BLUE)Building project (CPU + CUDA)...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DBUILD_CUDA=ON \
		-DBUILD_TESTS_CPU=ON \
		-DBUILD_TESTS_CUDA=ON \
		.. && make -j$$(nproc)
	@echo "$(GREEN)✓ Build complete!$(NC)"

build-debug: ## Build in debug mode
	@echo "$(BLUE)Building (Debug mode)...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Debug \
		-DBUILD_TESTS_CPU=ON \
		.. && make -j$$(nproc)
	@echo "$(GREEN)✓ Debug build complete!$(NC)"

clean: ## Clean build directory
	@echo "$(YELLOW)Cleaning build directory...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✓ Clean complete!$(NC)"

rebuild: clean build ## Clean and rebuild

# ==============================================================================
# Test Targets
# ==============================================================================

test: build ## Build and run all CPU tests
	@echo "$(BLUE)Running all CPU tests...$(NC)"
	@cd $(BUILD_DIR) && ctest --output-on-failure
	@echo "$(GREEN)✓ Tests complete!$(NC)"

test-cpu: ## Run CPU tests only
	@echo "$(BLUE)Running CPU tests...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DBUILD_TESTS_CPU=ON -DBUILD_TESTS_CUDA=OFF .. && make -j$$(nproc)
	@cd $(BUILD_DIR) && ctest -R "cpu" --output-on-failure
	@echo "$(GREEN)✓ CPU tests complete!$(NC)"

test-gpu: ## Run GPU tests only (requires CUDA)
	@echo "$(BLUE)Running GPU tests...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DBUILD_CUDA=ON -DBUILD_TESTS_CPU=OFF -DBUILD_TESTS_CUDA=ON .. && make -j$$(nproc)
	@cd $(BUILD_DIR) && ctest -R "cuda" --output-on-failure
	@echo "$(GREEN)✓ GPU tests complete!$(NC)"

test-quick: ## Run quick tests only (no stress/benchmark)
	@echo "$(BLUE)Running quick tests...$(NC)"
	@cd $(BUILD_DIR) && ctest -E "stress|benchmark" --output-on-failure
	@echo "$(GREEN)✓ Quick tests complete!$(NC)"

# ==============================================================================
# Benchmark Targets
# ==============================================================================

benchmark-cpu: ## Run CPU benchmarks (WARNING: may take hours)
	@echo "$(YELLOW)Running CPU benchmarks (this may take hours)...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DBUILD_TESTS_CPU=ON -DBUILD_TESTS_BENCHMARK=ON .. && make -j$$(nproc)
	@cd $(BUILD_DIR) && ctest -R "benchmark.*cpu" --output-on-failure
	@echo "$(GREEN)✓ CPU benchmarks complete!$(NC)"

benchmark-gpu: ## Run GPU benchmarks
	@echo "$(BLUE)Running GPU benchmarks...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DBUILD_CUDA=ON -DBUILD_TESTS_CUDA=ON -DBUILD_TESTS_BENCHMARK=ON .. && make -j$$(nproc)
	@cd $(BUILD_DIR) && ctest -R "benchmark.*cuda" --output-on-failure
	@echo "$(GREEN)✓ GPU benchmarks complete!$(NC)"

benchmark-all: ## Run ALL benchmarks (CPU + GPU) - VERY LONG
	@echo "$(RED)WARNING: This will take a VERY long time!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C within 5 seconds to cancel...$(NC)"
	@sleep 5
	@$(MAKE) benchmark-cpu
	@$(MAKE) benchmark-gpu

# ==============================================================================
# Run Targets
# ==============================================================================

run: build ## Build and run main program
	@echo "$(BLUE)Running main program...$(NC)"
	@$(BUILD_DIR)/src/hmm_main

run-cuda: build-cuda ## Build and run with CUDA
	@echo "$(BLUE)Running main program (CUDA)...$(NC)"
	@$(BUILD_DIR)/src/hmm_main


# ==============================================================================
# Development Tools
# ==============================================================================

format: ## Format code with clang-format
	@echo "$(BLUE)Formatting code...$(NC)"
	@find src include tests -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -i
	@echo "$(GREEN)✓ Code formatted!$(NC)"

info: ## Show project information
	@echo "$(BLUE)=====================================$(NC)"
	@echo "$(BLUE)  GPU HMM Project Information$(NC)"
	@echo "$(BLUE)=====================================$(NC)"
	@echo "  Build dir   : $(BUILD_DIR)"
	@echo "  Build type  : $(BUILD_TYPE)"
	@echo "  CMake       : $$(command -v cmake >/dev/null && cmake --version | head -n1 || echo 'Not found')"
	@echo "  GCC         : $$(command -v gcc >/dev/null && gcc --version | head -n1 || echo 'Not found')"
	@echo "  CUDA        : $$(command -v nvcc >/dev/null && nvcc --version | grep release || echo 'Not found')"
	@echo "  Python      : $$(command -v python3 >/dev/null && python3 --version || echo 'Not found')"
	@echo "$(BLUE)=====================================$(NC)"


# ==============================================================================
# Python Environment
# ==============================================================================

venv: ## Setup Python virtual environment (run once)
	@if [ ! -d ".venv" ]; then \
		echo "$(BLUE)Setting up Python environment...$(NC)"; \
		bash scripts/python/setup_venv.sh; \
	else \
		echo "$(GREEN)✓ Virtual environment already exists$(NC)"; \
	fi

venv-clean: ## Remove virtual environment
	@echo "$(YELLOW)Removing virtual environment...$(NC)"
	@rm -rf .venv
	@echo "$(GREEN)✓ Done$(NC)"

# ==============================================================================
# Data Preparation
# ==============================================================================

data: ## Prepare financial data (download + process)
	@echo "$(BLUE)Preparing financial data...$(NC)"
	@.venv/bin/python scripts/python/prepare_data.py
	@echo "$(GREEN)✓ Data ready in data/processed/$(NC)"

# ==============================================================================
# Cluster Targets (Ensicompute via VPN Ensimag)
# ==============================================================================

cluster-submit: ## Submit GPU job to ensicompute
	@echo "$(BLUE)Submitting GPU job...$(NC)"
	@bash scripts/cluster/submit_gpu_job.sh

cluster-submit-test: ## Submit specific test (usage: make cluster-submit-test TEST=test_hmm_cuda)
	@echo "$(BLUE)Submitting test: $(TEST)...$(NC)"
	@bash scripts/cluster/submit_gpu_job.sh $(TEST)

cluster-status: ## Check job status
	@bash scripts/cluster/check_status.sh

cluster-logs: ## Download logs from cluster
	@bash scripts/cluster/download_logs.sh

cluster-cancel: ## Cancel all jobs (or specific: make cluster-cancel JOB=12345)
	@bash scripts/cluster/cancel_jobs.sh $(if $(JOB),$(JOB),all)

cluster-interactive: ## Start interactive GPU session (for debugging)
	@bash scripts/cluster/interactive_gpu.sh

cluster-results: ## Download all results (logs + benchmarks)
	@echo "$(BLUE)Downloading all results...$(NC)"
	@rsync -avz dialloh@mash.ensimag.fr:~/gpu_comp_hmm_regime/results/ ./results_cluster/
	@echo "$(GREEN)✓ Results in results_cluster/$(NC)"

cluster-clean: ## Clean build directory on cluster
	@echo "$(YELLOW)Cleaning build on cluster...$(NC)"
	@ssh dialloh@mash.ensimag.fr "cd gpu_comp_hmm_regime && rm -rf build"
	@echo "$(GREEN)✓ Clean complete$(NC)"