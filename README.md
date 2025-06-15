# Monet Style Transfer 

This repository contains the code, models, and report for our final project in the course:  
**Deep Learning and its Applications to Signal and Image Processing and Analysis (361.2.1120)**

## Project Overview

The goal of this project is to translate real-world photographs into **Monet-style paintings**, using unsupervised **image-to-image translation** methods.

We compare two deep generative models:  
1. **CycleGAN (ResNet-based)** — A vanilla implementation that uses residual blocks in the generator.  
2. **CycleGAN with U-Net Generator** — An improved version using U-Net architecture with skip connections to better preserve structural details during translation.

We evaluate both models quantitatively using the **MiFID** score (memorization-informed Fréchet Inception Distance) and qualitatively with visual examples.

## Dataset

We use the dataset from the Kaggle competition:  
**[I’m Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started)**

It contains:
- ~7,000 real-world landscape photos.
- ~300 Monet-style paintings.
- No aligned pairs → **unpaired image translation** task.

To download the dataset manually:
1. Visit the competition link above and accept the rules.
2. Download the ZIP file and extract it into `./data/`.

## Related Papers

This project is inspired by the following papers studied in the course:
- [CycleGAN: Unpaired Image-to-Image Translation](https://arxiv.org/abs/1703.10593) — Zhu et al., 2017  
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015  
- [GANs: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Goodfellow et al., 2014  
- Metric: [FID and MiFID](https://arxiv.org/abs/2002.09797)

## Folder Structure

```
monet-style-transfer-final/
├── README.md                  # This file
├── report/
│   └── final_report.pdf       # Final report (submitted version)
├── models/
│   ├── base_cyclegan.py       # ResNet-based CycleGAN generator and discriminator
│   └── unet_cyclegan.py       # U-Net-based CycleGAN generator
├── train/
│   ├── train_base.py          # Training script for baseline model
│   └── train_unet.py          # Training script for improved U-Net model
├── data/
│   ├── instructions_to_download.txt # Notes for downloading the dataset
│   └── trainA/, trainB/, testA/, testB/ # Organized dataset
├── results/
│   ├── fid_scores.csv         # MiFID scores for all models
│   └── sample_outputs/        # Visual outputs for both models
└── utils/
    └── metrics.py             # Code for computing MiFID
```

## How to Run

> Environment: Python 3.10, PyTorch, Colab Pro recommended

1. Clone the repo  
   ```bash
   git clone https://github.com/your_username/monet-style-transfer-final.git
   cd monet-style-transfer-final
   ```

2. Download the dataset from Kaggle and place it in `./data/`.

3. Train the baseline model:
   ```bash
   python train/train_base.py
   ```

4. Train the U-Net model:
   ```bash
   python train/train_unet.py
   ```

5. Generate translated images and compute MiFID:
   ```bash
   python utils/metrics.py
   ```

## Evaluation

We evaluate both models on:
- **Quantitative metric:** MiFID (lower is better).
- **Qualitative results:** Success/failure cases shown in the report.
- **Ablation Study:** We evaluate the importance of the U-Net structure by comparing to ResNet generator.

## Project Authors

- Shahar Ain Kedem
- Yuval Ratzabi

This project is part of the final assignment for the course:  
**"Deep Learning and its Applications to Signal and Image Processing and Analysis" — Spring 2025, Ben-Gurion University**

## License

This repository is for educational purposes only.
