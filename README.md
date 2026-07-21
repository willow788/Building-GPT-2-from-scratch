# Building GPT-2 from Scratch

A clean, educational implementation of GPT-2-style transformer components in **PyTorch**, developed step by step from first principles.

> ⚠️ **Project Status:** This repository is currently a **work in progress** and not yet feature-complete.

---

## Overview

This project aims to recreate the core architecture and training workflow of GPT-2 in a modular and understandable way.  
The goal is to make each component transparent, testable, and extensible for learning and experimentation.

The repository currently includes foundational transformer modules (such as multi-head self-attention) and will continue expanding toward a full GPT-2 training and inference pipeline.

---

## Current Progress

### ✅ Implemented (so far)

- **Multi-Head Self-Attention module** (`module/attention.py`)
  - Query/Key/Value linear projections
  - Multi-head reshaping and head-wise attention
  - Scaled dot-product attention
  - Causal masking (autoregressive behavior)
  - Softmax attention weighting
  - Output projection back to embedding dimension

### 🚧 In Progress

- Additional transformer block components
- End-to-end GPT-style model wiring
- Better testing and validation utilities
- Documentation improvements across modules

---

## Planned Features (Upcoming)

The following items are planned as the project evolves:

- [ ] **Token + positional embeddings**
- [ ] **Feed-forward (MLP) sublayers**
- [ ] **Layer normalization + residual connections**
- [ ] **Stacked transformer decoder blocks**
- [ ] **Language modeling head**
- [ ] **Config system for model hyperparameters**
- [ ] **Training script (single GPU baseline)**
- [ ] **Loss computation and optimization loop**
- [ ] **Text generation / inference script**
- [ ] **Checkpoint save/load support**
- [ ] **Dataset preprocessing + tokenization pipeline**
- [ ] **Evaluation metrics (perplexity, validation loss)**
- [ ] **Support for scaling to larger models**
- [ ] **Unit tests for key modules**
- [ ] **Examples and reproducible experiments**

---

## Repository Structure (Current)

```text
Building-GPT-2-from-scratch/
├── module/
│   └── attention.py   # Multi-head causal self-attention implementation
└── ...                # Additional modules and scripts will be added
```

---

## Technical Direction

This implementation follows the decoder-only transformer approach used by GPT-style models:

1. Convert tokens into embeddings  
2. Add positional information  
3. Pass through stacked masked self-attention + MLP blocks  
4. Project to vocabulary logits  
5. Train with next-token prediction objective

The codebase is being organized for clarity first, then performance optimizations.

---

## Installation

```bash
git clone https://github.com/willow788/Building-GPT-2-from-scratch.git
cd Building-GPT-2-from-scratch
pip install -r requirements.txt
```

> If `requirements.txt` is not yet available, install core dependencies manually:
>
> ```bash
> pip install torch numpy
> ```

---

## Example (Current Module Test)

The current `attention.py` includes a small driver snippet for shape validation:

- Input shape: `(batch, seq_len, embed_dim)`  
- Output shape: `(batch, seq_len, embed_dim)`  
- Attention weights: `(batch, heads, seq_len, seq_len)`

---

## Development Goals

- Keep modules readable and educational
- Stay close to GPT-2 design principles
- Build incrementally with clear milestones
- Add tests and training tooling as components stabilize

---

## Contributing

Contributions, suggestions, and review feedback are welcome as the repository grows.

If you’d like to contribute:
1. Fork the repo
2. Create a feature branch
3. Open a pull request with a clear description

---

## Disclaimer

This project is intended for **learning and research purposes** and is currently under active development.  
APIs, file structure, and implementation details may change as the project matures.

---

## Author

**[willow788](https://github.com/willow788)**

If you find this project useful, consider starring the repository to follow progress.