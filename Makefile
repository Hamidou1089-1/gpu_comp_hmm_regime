# ==============================================================================
# Makefile Wrapper for GPU HMM Regime Detection
# ==============================================================================


.PHONY: help build build-cuda clean test test-cpu test-gpu benchmark info data venv cluster-build cluster-submit cluster-status cluster-logs cluster-clean profile-cpu profile-analyze profile-all analyze submit-gpu download-results clean-results venv
.DEFAULT_GOAL := help

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# Configuration
BUILD_DIR := build
RESULTS_DIR := results
DATA_DIR := data
SCRIPTS_DIR := scripts
BUILD_TYPE := Release

REMOTE_USER := dialloh
REMOTE_HOST := nash.ensimag.fr
PROJECT_NAME := gpu_comp_hmm_regime



help: ## Show this help message
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║    GPU HMM REGIME DETECTION - PROFILING PIPELINE             ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Data Preparation:$(NC)"
	@grep -E '^data.*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)CPU Profiling (Local):$(NC)"
	@grep -E '^profile-cpu.*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)GPU Profiling (Cluster):$(NC)"
	@grep -E '^(profile-gpu|submit-gpu|download).*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Analysis:$(NC)"
	@grep -E '^analyze.*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Full Pipeline:$(NC)"
	@grep -E '^(profile-all|pipeline).*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
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

test-quick: ## Run quick tests only (no stress/benchmark)
	@echo "$(BLUE)Running quick tests...$(NC)"
	@cd $(BUILD_DIR) && ctest -E "stress|benchmark" --output-on-failure
	@echo "$(GREEN)✓ Quick tests complete!$(NC)"





# ==============================================================================
# Development Tools
# ==============================================================================

format: ## Format code with clang-format
	@echo "$(BLUE)Formatting code...$(NC)"
	@find src include tests -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -i
	@echo "$(GREEN)✓ Code formatted!$(NC)"




info: ## Show project information
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║    PROJECT INFORMATION                                       ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "Local:"
	@echo "  Build dir:    $(BUILD_DIR)"
	@echo "  Results dir:  $(RESULTS_DIR)"
	@echo "  Data dir:     $(DATA_DIR)"
	@echo ""
	@echo "Cluster:"
	@echo "  Host:         $(REMOTE_HOST)"
	@echo "  User:         $(REMOTE_USER)"
	@echo "  Project:      $(PROJECT_NAME)"
	@echo ""
	@echo "Tools:"
	@echo "  CMake:        $$(command -v cmake >/dev/null && cmake --version | head -n1 || echo 'Not found')"
	@echo "  GCC:          $$(command -v gcc >/dev/null && gcc --version | head -n1 || echo 'Not found')"
	@echo "  Python:       $$(command -v python3 >/dev/null && python3 --version || echo 'Not found')"
	@echo "  perf:         $$(command -v perf >/dev/null && echo 'Available' || echo 'Not found')"
	@echo ""

sync-to-cluster: ## Sync local code to cluster
	@echo "$(BLUE)📤 Syncing to cluster...$(NC)"
	@rsync -avz --progress \
		--exclude 'build/' \
		--exclude '.venv/' \
		--exclude 'data/raw/' \
		--exclude '__pycache__/' \
		./ $(REMOTE_USER)@$(REMOTE_HOST):~/$(PROJECT_NAME)/
	@echo "$(GREEN)✓ Sync complete$(NC)"

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

data-clean: ## Remove processed data
	@rm -rf $(DATA_DIR)/processed/*
	@echo "$(YELLOW)Cleaned processed data$(NC)"

# ==============================================================================
# Cluster Targets (Ensicompute via VPN Ensimag)
# ==============================================================================

# Build localement avant de submit (recommandé)
cluster-build:
	@echo "🔨 Building project locally first..."
	mkdir -p build && cd build && \
	cmake -DCMAKE_BUILD_TYPE=Release \
	      -DBUILD_CUDA=OFF \
	      -DBUILD_TESTS_CPU=ON \
	      -DBUILD_TESTS_CUDA=ON \
	      .. && \
	make -j4
	@echo "✅ Local build successful"

# Submit job (avec build préalable)
cluster-submit: cluster-build
	@echo "🚀 Submitting job to cluster..."
	./scripts/cluster/submit_gpu_job.sh $(TEST)

cluster-submit-finance: 
	$(MAKE) cluster-submit TEST=test_finance_gpu

# Clean remote build
cluster-clean:
	@ssh dialloh@nash.ensimag.fr 'cd gpu_comp_hmm_regime && rm -rf build/'
	@echo "🧹 Remote build directory cleaned"


profile-gpu: submit-gpu-timing ## Alias for submit-gpu-timing

submit-gpu-timing: ## Submit GPU timing benchmark to cluster
	@echo "$(BLUE)🚀 Submitting GPU timing job...$(NC)"
	@bash $(SCRIPTS_DIR)/cluster/submit_profiling_job.sh timing all

submit-gpu-nsight: ## Submit GPU nsight profiling to cluster (slow)
	@echo "$(BLUE)🚀 Submitting GPU nsight job...$(NC)"
	@bash $(SCRIPTS_DIR)/cluster/submit_profiling_job.sh nsight real

submit-gpu-full: ## Submit full GPU profiling (timing + nsight)
	@echo "$(BLUE)🚀 Submitting full GPU profiling job...$(NC)"
	@bash $(SCRIPTS_DIR)/cluster/submit_profiling_job.sh full all

download-results: ## Download profiling results from cluster
	@echo "$(BLUE)📥 Downloading results from cluster...$(NC)"
	@bash $(SCRIPTS_DIR)/cluster/download_profiling_results.sh

cluster-status: ## Check cluster job status
	@ssh $(REMOTE_USER)@$(REMOTE_HOST) 'squeue -u $(REMOTE_USER)'

cluster-logs: ## Show latest cluster log
	@ssh $(REMOTE_USER)@$(REMOTE_HOST) 'tail -50 $(PROJECT_NAME)/results/logs/*.out 2>/dev/null | tail -100'


# ==============================================================================
# ANALYSIS
# ==============================================================================

# Complete workflow helper
profile-help: ## Show profiling workflow
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)           PROFILING WORKFLOW                                  $(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)Step 1: Generate benchmark data$(NC)"
	@echo "  make generate-data"
	@echo ""
	@echo "$(YELLOW)Step 2: Run local profiling (CPU + hmmlearn)$(NC)"
	@echo "  make profile-local"
	@echo ""
	@echo "$(YELLOW)Step 3: Submit GPU job to cluster$(NC)"
	@echo "  make submit-profiling           # Default (timing, all tests)"
	@echo "  make submit-profiling-quick     # Quick test"
	@echo "  make submit-profiling-full      # Full profiling with nsight"
	@echo ""
	@echo "$(YELLOW)Step 4: Check job status$(NC)"
	@echo "  make cluster-status"
	@echo ""
	@echo "$(YELLOW)Step 5: Download results and analyze$(NC)"
	@echo "  make profile-analyze-all"
	@echo ""
	@echo "$(YELLOW)Or run individual steps:$(NC)"
	@echo "  make download-profiling         # Download GPU results"
	@echo "  make analyze-benchmarks         # Generate plots"
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"

# Generate benchmark data (run once)
generate-data:  ## Generate synthetic benchmark datasets
	@echo "$(BLUE)Generating benchmark data...$(NC)"
	@mkdir -p data/bench
	@python3 scripts/python/generate_benchmark_data.py
	@echo "$(GREEN)✓ Data generated in data/bench/$(NC)"

# CPU Profiling (local)
profile-cpu-robust: build ## Run robust CPU profiling on benchmark data
	@echo "$(BLUE)Running CPU profiling (robust)...$(NC)"
	@mkdir -p results/benchmarks
	@./build/test/test_profile_cpu_robust data/bench results/benchmarks/cpu_benchmark all
	@echo "$(GREEN)✓ CPU profiling complete$(NC)"
	@echo "Results: results/benchmarks/cpu_benchmark_results.csv"

profile-cpu-finance: build ## Run robust CPU profiling on benchmark data
	@echo "$(BLUE)Running CPU finance (SP&500)...$(NC)"
	@mkdir -p results/finance
	@./build/test/test_finance_cpu 
	@echo "$(GREEN)✓ CPU finance complete$(NC)"
	@echo "Results: results/benchmarks/cpu_benchmark_results.csv"

profile-cpu-perf:  ## Run CPU profiling with perf metrics
	@echo "$(BLUE)Running CPU profiling with perf...$(NC)"
	@chmod +x scripts/run_cpu_profiling.sh
	@./scripts/run_cpu_profiling.sh data/bench results/benchmarks/cpu_benchmark perf
	@echo "$(GREEN)✓ CPU profiling with perf complete$(NC)"

# hmmlearn baseline
profile-hmmlearn: ## Run hmmlearn baseline benchmark
	@echo "$(BLUE)Running hmmlearn benchmark...$(NC)"
	@mkdir -p results/benchmarks
	@python3 scripts/python/benchmark_hmmlearn.py data/bench results/benchmarks/hmmlearn_benchmark
	@echo "$(GREEN)✓ hmmlearn benchmark complete$(NC)"
	@echo "Results: results/benchmarks/hmmlearn_benchmark_results.csv"

# Analyze all results
analyze-benchmarks: ## Analyze benchmark results and generate plots
	@echo "$(BLUE)Analyzing benchmark results...$(NC)"
	@mkdir -p results/figures
	@python3 scripts/python/analyze_benchmark_results.py \
		--cpu-csv results/benchmarks/cpu_benchmark_results.csv \
		--gpu-csv results/benchmarks/gpu_benchmark_results.csv \
		--hmmlearn-csv results/benchmarks/hmmlearn_benchmark_results.csv \
		--output-dir results/figures
	@echo "$(GREEN)✓ Analysis complete$(NC)"
	@echo "Figures saved in: results/figures/"

# ==============================================================================
# CLUSTER PROFILING TARGETS
# ==============================================================================

# Submit GPU profiling job
submit-profiling: ## Submit GPU profiling job to cluster
	@echo "$(BLUE)Submitting GPU profiling job...$(NC)"
	@chmod +x scripts/cluster/submit_profiling_job.sh
	@./scripts/cluster/submit_profiling_job.sh $(MODE) $(TEST)

submit-profiling-quick: ## Submit quick GPU profiling job (small datasets)
	@$(MAKE) submit-profiling MODE=timing TEST=quick

submit-profiling-full: ## Submit full GPU profiling job (timing + nsight)
	@$(MAKE) submit-profiling MODE=full TEST=all

# Download results from cluster
download-profiling: ## Download profiling results from cluster
	@echo "$(BLUE)Downloading profiling results...$(NC)"
	@chmod +x scripts/cluster/download_profiling_results.sh
	@./scripts/cluster/download_profiling_results.sh
	@echo "$(GREEN)✓ Download complete$(NC)"

# ==============================================================================
# COMPLETE PROFILING WORKFLOW
# ==============================================================================

# Full local profiling (CPU + hmmlearn)
profile-local: profile-cpu-robust profile-hmmlearn ## Run complete local profiling (CPU + hmmlearn)
	@echo "$(GREEN)✓ Local profiling complete$(NC)"
	@echo ""
	@echo "Next: Submit GPU job with 'make submit-profiling'"

# Full analysis after GPU results
profile-analyze-all: analyze-benchmarks ## analyze all
	@echo "$(GREEN)✓ Full analysis complete$(NC)"
	@echo "Open results/figures/ to see the plots"

# ==============================================================================
# BUILD TARGETS
# ==============================================================================

build-cpu: ## Build CPU profiling binary
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release \
		-DBUILD_PROFILING=ON \
		-DBUILD_TESTS_CPU=ON \
		.. && make -j$$(nproc) test_profile_cpu_robust
	@echo "$(GREEN)✓ CPU profiling binary built$(NC)"

build-debug: ## Build with debug symbols
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Debug \
		-DBUILD_TESTS_CPU=ON \
		-DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer" \
		.. && make -j$$(nproc)

# ==============================================================================
# CLEANING
# ==============================================================================

clean-results: ## Clean all profiling results
	@rm -rf $(RESULTS_DIR)/cpu_profiling/*
	@rm -rf $(RESULTS_DIR)/benchmarks/*
	@rm -rf $(RESULTS_DIR)/figures/*
	@rm -rf $(RESULTS_DIR)/nsight/*
	@echo "$(YELLOW)Cleaned all results$(NC)"

clean-build: ## Clean build directory
	@rm -rf $(BUILD_DIR)
	@echo "$(YELLOW)Cleaned build directory$(NC)"

clean: clean-build ## Clean everything
