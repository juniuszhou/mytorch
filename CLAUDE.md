# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- **Run a demo**: Execute any Python script in the `src/` directory to run a demonstration.
  Example: `python src/embedding/call-embedding.py`
- **Install dependencies**: The project uses `uv` for dependency management. To sync dependencies:
  ```bash
  uv sync
  ```
- **Activate virtual environment**: A `.venv` directory is already present. Activate it with:
  ```bash
  source .venv/bin/activate
  ```
- **Linting**: No formal linting setup is configured. Consider using `ruff` or `flake8` if needed.
- **Testing**: No test suite is currently implemented. Demos serve as informal verification.

## Project Architecture

The repository is organized as a collection of educational demonstrations covering various PyTorch-related topics. Each directory under `src/` represents a self-contained tutorial or experiment:

- **embedding**: Demonstrations of embedding layers and lookup operations.
- **evaluation**: Scripts for evaluating model performance (e.g., perplexity calculation).
- **ezkl**: Experiments with EZKL (Zero-Knowledge Proofs) integration.
- **gradient**: Implementations and visualizations of gradient computation and backpropagation.
- **inference**: Techniques for efficient inference, including KV-caching and prefix sharing.
- **loss**: Examples of loss function implementations and visualizations.
- **parallel**: Distributed training techniques: Data Parallelism (DP), Tensor Parallelism (TP), Fully Sharded Data Parallelism (FSDP2), and Pipeline Parallelism.
- **pipeline**: End-to-end pipeline demonstrations combining multiple concepts.
- **sft**: Supervised Fine-Tuning (SFT) examples.
- **tokenization**: Tokenization experiments and visualizations.
- **utils**: Utility scripts, including Hugging Face authentication helpers.

### Key Observations
- Each `src/*` directory contains one or more `.py` files that can be run independently to explore a specific concept.
- Many demos require external dependencies (e.g., Hugging Face Transformers) and may need API keys (stored in `.env` for HF authentication).
- The project emphasizes hands-on experimentation over formal software engineering practices (e.g., no unit tests, minimal packaging).
- The `main.py` file is a simple placeholder printing "Hello from mytorch!".

## Notes for Claude Code
- When assisting with modifications, prioritize preserving the educational and demonstrative nature of the code.
- Changes should aim to enhance clarity, correctness, or accessibility of the demonstrations.
- Avoid introducing unnecessary abstractions; the value lies in the direct, readable implementations.
- If adding new features, consider creating a new demo script in the relevant `src/` directory rather than modifying existing ones unless fixing bugs.
- Environment variables (like `HF_KEY`) are expected in a `.env` file at the project root for Hugging Face-related demos.
