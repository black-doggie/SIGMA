IGMA Reproduction (DSW Environment)

This repository contains a **reproduction of the paper**:

> SIGMA: Selective Gated Mamba for Sequential Recommendation
> 
> 
> Ziwang Liu et al.
> 

The reproduction is conducted on **Alibaba Cloud Data Science Workshop (DSW)** with GPU support, following the experimental settings reported in the paper as closely as possible.

---

## 1. Purpose of This Repository

The goal of this repository is to:

- Reproduce the **main experimental results** of SIGMA on standard sequential recommendation benchmarks
- Verify that the official implementation can achieve **comparable performance** to the paper
- Serve as a clean and inspectable codebase for **further ablation studies and extensions**

This repository is intended for **research reproduction and academic guidance**, not for production use.

---

## 2. Environment

- **Platform**: Alibaba Cloud Data Science Workshop (DSW)
- **OS**: Linux
- **GPU**: NVIDIA GPU (≈24GB VRAM)
- **Python**: 3.10
- **PyTorch**: 2.1.1 + CUDA 11.8
- **CUDA**: Available (`torch.cuda.is_available() == True`)
- **Key dependencies**:
    - `mamba_ssm == 1.1.4`
    - `causal_conv1d` (verified working)

The environment has been verified to successfully train and evaluate the SIGMA model.

---

## 3. Repository Structure

```
SIGMA/
├── model/
│   ├── gated_mamba.py# Core SIGMA model implementation
│   ├── run.py# Original training script
│   ├── run_noflops.py# Modified training script (FLOPs skipped)
│   ├── config.yaml# Default configuration
│   ├── config_ml-1m.yaml# Dataset-specific config (MovieLens-1M)
│   ├── run_5seeds.sh# Script for multi-seed experiments
│   └── dataset/
│       └── ml-1m/# Processed MovieLens-1M dataset
│
├── .gitignore# Ignore datasets, logs, checkpoints
└── README.md

```

Notes:

- **Raw datasets and large logs are excluded** from the repository.
- Dataset files should be placed locally according to the expected structure.

---

## 4. Dataset Status

### Currently completed:

- **MovieLens-1M (ml-1m)**

### Planned (paper-aligned):

- Amazon Beauty
- Amazon Sports and Outdoors
- Amazon Fashion
- Amazon Video Games

Each dataset will be trained and evaluated independently using the same protocol as described in the paper.

---

## 5. Training Configuration (Example: ML-1M)

Key settings (from `config_ml-1m.yaml`):

- Loss: Cross-Entropy (CE)
- Hidden size: 64
- Layers: 1
- `d_state = 32`
- `d_conv = 4`
- `expand = 2`
- Max sequence length: 200
- Optimizer: Adam
- Learning rate: 1e-3
- Epochs: 200 (early stopping enabled)

To avoid a known **Conv1D shape issue during FLOPs computation**, FLOPs calculation is skipped in `run_noflops.py`.

This does **not affect training or evaluation results**, only logging.

---

## 6. Example Command

```bash
python run_noflops.py --model=SIGMA --dataset=ml-1m

```

---

## 7. Current Reproduction Results (ML-1M)

Best validation performance (single run):

- **Hit@10**: 0.3354
- **NDCG@10**: 0.1925
- **MRR@10**: 0.1489

Corresponding test performance:

- **Hit@10**: 0.3083
- **NDCG@10**: 0.1814
- **MRR@10**: 0.1424

The performance trend is **consistent with the paper**, considering possible differences due to:

- Random seed
- Hardware
- Library versions

---

## 8. Current Progress Summary

- ✅ Environment successfully configured
- ✅ Code runs end-to-end on DSW GPU
- ✅ One dataset (ML-1M) reproduced with reasonable performance
- 🔄 Remaining datasets to be reproduced
- 🔜 Next steps: multi-dataset + multi-seed experiments, result aggregation, and paper-level comparison table

---

## 9. Next Planned Steps

1. Run SIGMA on all five datasets reported in the paper
2. Repeat experiments with multiple random seeds
3. Aggregate results into a comparison table
4. Align results with paper tables
5. Discuss deviations and possible causes
6. Explore optional ablations or extensions (if needed)

---

## 10. Reference

Official paper and codebase:

- Paper: *SIGMA: Selective Gated Mamba for Sequential Recommendation*
- Official repository: [https://github.com/ziwliu8/SIGMA](https://github.com/ziwliu8/SIGMA?utm_source=chatgpt.com)
