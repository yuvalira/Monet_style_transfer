# Monet Style Transfer

This repository contains the code, models, and report for our final project in the course:
**Deep Learning and its Applications to Signal and Image Processing and Analysis (361.2.1120)**

## Project Overview

The goal of this project is to translate real-world photographs into **Monet-style paintings** using unsupervised **image-to-image translation**.

We implemented and compared two models:

1. **Vanilla CycleGAN (U-Net based)** — Uses U-Net generators with adversarial, cycle-consistency, identity, and discriminator losses.
2. **Improved CycleGAN** — Extends the vanilla version with **self-attention layers**, **perceptual loss** (VGG-based feature similarity), and the **LSGAN loss** for more stable adversarial training.

We evaluate both models quantitatively using the **MiFID** score and qualitatively with visual examples.

## Dataset

We use the dataset from the Kaggle competition:
**[I’m Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started)**

It contains:

* 7,038 real-world landscape photos
* 300 Monet-style paintings
* No aligned pairs → an **unpaired image translation** task

## Results

* **Vanilla U-Net CycleGAN**: MiFID = **92.92**
* **Improved CycleGAN (Self-Attention + Perceptual Loss + LSGAN)**: MiFID = **87.14**

Lower MiFID indicates better similarity between generated Monet paintings and real Monet data. Visual examples of success and failure cases are included in the `results/` folder.

## Related Papers

This project was inspired by the following works:

* [CycleGAN: Unpaired Image-to-Image Translation](https://arxiv.org/abs/1703.10593) — Zhu et al., 2017
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
* [Self-Attention GAN](https://arxiv.org/abs/1805.08318) — Zhang et al., 2019
* [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155) — Johnson et al., 2016
* [Least Squares GAN (LSGAN)](https://arxiv.org/abs/1611.04076) — Mao et al., 2017
* Metric: [FID and MiFID](https://arxiv.org/abs/2002.09797)

## Folder Structure

```
monet-style-transfer/
├── README.md                  # This file
├── report/
│   └── final_report.pdf       # Final report
├── models/
│   ├── Monet_CycleGan_VanillaModel.py       # Vanilla U-Net CycleGAN
│   └── Monet_CycleGan_ImprovedModel.py   # Improved CycleGAN with attention + losses
├── data/
│   ├── EDA_monet.ipynb                    # Dataset Analysis
│   └── data_splits.ipynb                  # Data Split for test and train sets
├── results/
│   ├── MiFID.py               # MiFID compute functions
│   └── sample_outputs/        # Example outputs
```

## Project Authors

* **Shahar Ain Kedem**
* **Yuval Ratzabi**

This project is part of the final assignment for the course:
\*\*"Deep Learning and its Applications to Signal and Image Processing and Analysis" — Spring 2025, Ben-Gurion University
