# Monet Style Transfer

This repository contains the code, models, and report for our final project in the course:

Deep Learning and its Applications to Signal and Image Processing and Analysis (361.2.1120) — 2025, Ben-Gurion University

---

## Project Overview

The goal of this project is to translate **real-world photographs into Monet-style paintings** using **unsupervised image-to-image translation**.

We implement and compare two models:

1. **Vanilla CycleGAN**

   * U-Net generators
   * Adversarial loss, cycle-consistency loss, identity loss, and discriminator loss

2. **Improved CycleGAN**

   * Adds **self-attention layers**
   * Uses **perceptual loss** (VGG-based feature similarity)
   * Adopts **LSGAN loss** for more stable adversarial training

Evaluation is performed both:

* **Quantitatively** using the **MiFID** score
* **Qualitatively** with generated visual examples

---

## Dataset

We use the dataset from the Kaggle competition:
**[I’m Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started)**

It contains:

* **7,038** real-world landscape photos
* **300** Monet-style paintings
* **Unpaired images** (no aligned pairs → unpaired image translation task)

---

## Results

* **Vanilla U-Net CycleGAN**: MiFID = **92.92**
* **Improved CycleGAN**: MiFID = **87.14**

Lower MiFID indicates better similarity between generated Monet paintings and real Monet data.

Visual examples (including success and failure cases) are available in the [`results/`](./results) folder.

---

## Related Work

Our implementation was inspired by the following works:

* [CycleGAN: Unpaired Image-to-Image Translation](https://arxiv.org/abs/1703.10593) — Zhu et al., 2017
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
* [Self-Attention GAN](https://arxiv.org/abs/1805.08318) — Zhang et al., 2019
* [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155) — Johnson et al., 2016
* [Least Squares GAN (LSGAN)](https://arxiv.org/abs/1611.04076) — Mao et al., 2017
* Metric: [FID and MiFID](https://arxiv.org/abs/2002.09797)

---

## Folder Structure

```
monet-style-transfer/
├── README.md                      # Project documentation
├── report/
│   └── final_report.pdf            # Full project report
├── models/
│   ├── Monet_CycleGan_VanillaModel.py    # Vanilla U-Net CycleGAN
│   └── Monet_CycleGan_ImprovedModel.py   # Improved CycleGAN with attention + losses
├── data/
│   ├── EDA_monet.ipynb             # Dataset analysis
│   └── data_splits.ipynb           # Train/test split
├── results/
│   ├── MiFID.py                    # MiFID computation
│   ├── Result_comparision.ipynb    # Model comparison
│   └── Ablation_study.ipynb        # Self-Attention ablation study
```

---

## How to Run

### 1. Clone the repository

```bash
!git clone https://github.com/yuvalira/Monet_style_transfer.git
%cd Monet_style_transfer
```

### 2. Install dependencies

```bash
!pip -q install tensorflow==2.15.0 tensorflow-addons==0.23.0 pillow matplotlib kaggle
```

### 3. Download the Kaggle dataset

```python
from google.colab import files
import os

print("Please upload your kaggle.json (download from https://www.kaggle.com/settings/account)")
uploaded = files.upload()

os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "wb") as f:
    f.write(uploaded["kaggle.json"])
os.chmod("/root/.kaggle/kaggle.json", 0o600)

!kaggle competitions download -c gan-getting-started -p ./data -q
!unzip -qo ./data/gan-getting-started.zip -d ./data/gan-getting-started
```

Dataset will be available under:

* `./data/gan-getting-started/monet_tfrec/*.tfrec`
* `./data/gan-getting-started/photo_tfrec/*.tfrec`

### 4. Train the models

* **Vanilla CycleGAN** → run `models/Monet_CycleGan_VanillaModel.ipynb`
* **Improved CycleGAN** → run `models/Monet_CycleGan_ImprovedModel.ipynb`

---

## Authors

* **Shahar Ain Kedem**
* **Yuval Ratzabi**
