# Chunking Methods Experimentation

A simple, cross-platform framework for evaluation of text chunking strategies in IR systems.

## Prerequisites
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/)

## Installation
Clone the repo and open it
```bash
git clone https://github.com/samarrik/chunking.git
cd chunking
```

## Usage

### Running Experiments

Execute the main experiment suite *(take cares of the env)*:
```bash
uv run main.py
```

### Configuration

Experiment parameters can be configured in `configs/config.yaml`. Available parameters include:
- Chunk sizes
- Overlap strategies
- Number of retrieved chunks
- Embedding model selection

### Results

- Detailed report available in `reports/report.pdf`
- Experiment results are saved in the `results` directory: 
    - Actual data from the experiments `results/data/results.csv`
    - Figures depicting data from the experiments `results/figures/`

## Citation
The evaluation dataset, chunkers, and some simple utils are sourced from the Chroma research team's chunking evaluation framework [1].

## References

[1] Smith, B., & Troynikov, A. (2024, July). *Evaluating Chunking Strategies for Retrieval*. Chroma. 
    Retrieved from https://research.trychroma.com/evaluating-chunking
