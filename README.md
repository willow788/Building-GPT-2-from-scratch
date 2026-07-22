# Building GPT-2 from Scratch

A step-by-step, educational implementation of GPT-2-style transformer components in **PyTorch**.

This repository focuses on building the model from first principles with readable, modular code that makes the internals of a GPT-2-like architecture easy to understand, inspect, and extend.

> ⚠️ **Project Status:** Work in progress. The repository currently includes the foundation for multi-head causal self-attention and will continue expanding toward a full GPT-style decoder stack.

---

## What’s Included

### `module/attention.py`
A clean implementation of **multi-head self-attention** with:

- Query / Key / Value projections
- Multi-head reshaping
- Scaled dot-product attention
- Causal masking for autoregressive generation
- Softmax weighting
- Final output projection

The module also includes a small driver snippet to validate tensor shapes.

---

## Current Project Direction

This project is being built to mirror the core design of GPT-2 in a transparent way.

Planned next steps include:

- Token embeddings
- Positional embeddings
- Transformer decoder blocks
- Feed-forward layers
- Layer normalization and residual connections
- Language modeling head
- Training and inference scripts
- Checkpointing
- Tokenization and dataset preprocessing
- Unit tests and validation utilities

---

## Repository Structure

```text
Building-GPT-2-from-scratch/
├── module/
│   └── attention.py
└── README.md