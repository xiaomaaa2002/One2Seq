# One2Seq: One-Token Wise Decoder for Efficient Scene Text Recognition (AAAI 2026)

## 📌 Introduction

This repository provides a **PyTorch implementation** of:

> **One2Seq: One-Token Wise Decoder for Efficient Scene Text Recognition (AAAI 2026)**

One2Seq proposes an efficient sequence modeling paradigm for scene text recognition by adopting a **one-token-wise decoding strategy**, which significantly improves inference efficiency while maintaining competitive recognition performance.

---

## 🛠️ Environment Setup

This project is built upon the OpenOCR framework.

Please follow the official guide for environment setup:

👉 https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md

---

## 📂 Dataset Preparation

Datasets should also be prepared according to the OpenOCR instructions:

👉 https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md

---

## ▶️ Training

### 🔹 Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/one2seq/svtr-b_o2s.yml

### 🔹 Single-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_rec.py --c configs/rec/one2seq/svtr-b_o2s.yml

## ▶️ Evaluation

```bash
python tools/eval_rec.py --c configs/rec/one2seq/svtr-b_o2s.yml


