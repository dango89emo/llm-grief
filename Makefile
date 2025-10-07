.PHONY: help generate generate-batch generate-batch-100 generate-multi train train-ddp train-ddp-4 analyze clean

# Default target
help:
	@echo "LLM Grief Analysis - Available Commands:"
	@echo ""
	@echo "  make generate           - Generate diaries for single persona"
	@echo "  make generate-batch     - Generate diaries for multiple personas (batch inference, fast)"
	@echo "  make generate-batch-100 - Generate diaries for 100 personas (batch size=5)"
	@echo "  make generate-multi     - Generate diaries for multiple personas (sequential)"
	@echo "  make train              - Train SAE on collected activations (single GPU)"
	@echo "  make train-ddp          - Train SAE with DDP (2 GPUs)"
	@echo "  make train-ddp-4        - Train SAE with DDP (4 GPUs)"
	@echo "  make analyze            - Analyze SAE features (baseline vs grief)"
	@echo "  make clean              - Clean generated data and results"
	@echo ""

# Python executable (using venv)
PYTHON := .venv/bin/python

# Generate diaries for single persona
generate:
	$(PYTHON) src/data_generation.py config/config.yaml 32

# Generate diaries for multiple personas using batch inference (fast)
generate-batch:
	$(PYTHON) src/data_generation.py config/config_multi_personas.yaml 32 \
		--multiple-personas \
		--batch \
		--personas config/personas.yaml

# Generate diaries for 100 personas using batch inference (batch_size=5)
generate-batch-100:
	$(PYTHON) src/data_generation.py config/config_multi_personas.yaml 32 \
		--multiple-personas \
		--batch \
		--personas personas.yaml

# Generate diaries for multiple personas sequentially (slower)
generate-multi:
	$(PYTHON) src/data_generation.py config/config_multi_personas.yaml 32 \
		--multiple-personas \
		--personas config/personas.yaml

# Train SAE on collected activations (single GPU)
train:
	$(PYTHON) src/train_sae.py activations

# Train SAE with DDP (2 GPUs)
train-ddp:
	$(PYTHON) -m torch.distributed.run --nproc_per_node=2 src/train_sae.py activations --ddp

# Train SAE with DDP (4 GPUs)
train-ddp-4:
	$(PYTHON) -m torch.distributed.run --nproc_per_node=4 src/train_sae.py activations --ddp

# Analyze SAE features
analyze:
	$(PYTHON) src/analyze_sae.py \
		--activations-dir activations \
		--model-path models/sae_final.pt \
		--save-dir results

# Clean generated data and results
clean:
	rm -rf data/ activations/ models/ results/
	@echo "Cleaned all generated data"
