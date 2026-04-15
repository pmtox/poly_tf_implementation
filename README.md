# Poly-TF: Multi-Task Learning with Task Prompts

**A simplified implementation of Poly-TF (Polymeric Transformer Framework) for multi-task learning on CIFAR-10.**

This project implements a CNN-based version of the Poly-TF paper, demonstrating how a single model can learn multiple tasks simultaneously using **task prompts** and **adapters** to prevent catastrophic forgetting.

---

## Paper Reference

> **Poly-TF: A polymeric transformer framework for multiple visual tasks at once**  
> *Knowledge-Based Systems, Volume 327, October 2025*  
> [Link to paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705125012080)

This is a **learning implementation** — not an official reproduction. The goal was to understand multi-task learning by building from scratch.

---

## Tasks

| Task | Description | Output |
|------|-------------|--------|
| **Classification** | Identify object type | 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck) |
| **Rotation Detection** | Predict image rotation angle | 4 classes (0°, 90°, 180°, 270°) |

Both tasks are learned by the **same model** simultaneously.

---

## Architecture

Input Image → CNN Backbone → Shared Features → Task Prompts → Adapters → Outputs
↓
Classification Head
Rotation Head

- **CNN Backbone**: Shared convolutional layers for feature extraction
- **Task Prompts**: Small learnable vectors added to input for task specification
- **Adapters**: Lightweight modules for task-specific adjustments
- **Heads**: Separate classification and rotation output layers

---

##  Results

### Training Progress (10 → 40 Epochs)

| Epochs | Classification | Rotation |
|--------|---------------|----------|
| 10 | 78.3% | 72.4% |
| 40 | **85.1%** | **78.7%** |

**Improvement**: +6.8% (classification), +6.3% (rotation)

### Baseline Comparison

| Model | Classification | Rotation |
|-------|---------------|----------|
| Single-task (ResNet18 / RotNet) | 93% | 75-80% |
| **Our Model (40 epochs)** | **85.1%** | **78.7%** |
| **Gap** | **-7.9%** | **-0.7% to +3.7%** |

### Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Prompt Similarity** | -0.1718 | Tasks learn complementary features (good!) |
| **Training epochs** | 40 | Converged with no overfitting |

---

## Learning Curves

*[Insert your learning curve images here]*

The model shows steady improvement with minimal gap between train and test accuracy — no significant overfitting.

---

## Sample Outputs

| Image | Prediction |
|-------|------------|
| Dog | "Object Type: dog (99.4%)", "Rotation: 0° (99.5%)" |
| Horse | "Object Type: horse (100.0%)", "Rotation: 0° (99.9%)" |
| Cat | "Object Type: cat (66.5%)", "Rotation: 0° (83.9%)" |
| Automobile | "Object Type: automobile (84.8%)", "Rotation: 90° (45.5%)" |

---

## Requirements

```bash
pip install torch torchvision matplotlib numpy

Limitations
Classification accuracy (85.1%) still below single-task baseline (93%)

Rotation detection works well, but certain classes (automobile) need improvement

Simplified CNN backbone (not the full transformer from the paper)

Trained only on CIFAR-10 (small, clean images) — real-world performance may vary

What I Learned
Reading and implementing research papers

Multi-task learning with shared representations

Task prompts and adapters to prevent catastrophic forgetting

Training, evaluating, and documenting ML experiments

Interpreting metrics like prompt similarity