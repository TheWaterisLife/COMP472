# ASL Recognition Using Convolutional Neural Networks


We are mainly using Google Drive as a sharing platform for our project. Here is the link to our Google Drive:
https://drive.google.com/drive/folders/1Zs071JSPT88VIHSQXuHdqIvBFeRL0Urx?usp=drive_link

**COMP 472 — Applied Artificial Intelligence | Concordia University | Winter 2026**

A comparative study of CNN architectures for American Sign Language (ASL) hand gesture recognition. Three models are trained and evaluated across three datasets of increasing complexity, with transfer learning applied to the most challenging task.

---

## Datasets

| Dataset | Classes | Images | Description |
|---------|:-------:|:------:|-------------|
| **ASL Commands** | 3 | ~9,000 | Utility gestures: *Space*, *Delete*, *Nothing* (200×200 RGB) |
| **ASL Digits** | 10 | ~2,062 | Digits 0–9 from 218 participants (100×100 RGB) |
| **ASL Alphabets** | 26 | ~39,000 | Letters A–Z, subsampled to ~1,500/class (200×200 RGB) |

All datasets are split **80 / 10 / 10** (train / val / test) with stratified sampling.

**Sources:**
- [ASL Alphabet — Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (Commands & Alphabets)
- [Sign Language Digits — GitHub](https://github.com/ardamavi/Sign-Language-Digits-Dataset)

---

## Model Architectures

| Model | Type | Details |
|-------|------|---------|
| **VGG-16** | From scratch + Transfer Learning | VGG-16 with BatchNorm; ImageNet-pretrained variant for Alphabets |
| **ResNet-10** | Custom from scratch | 4-stage residual network (~10 layers, 64→512 filters) |
| **ResNet-50** | Transfer Learning | ImageNet-pretrained ResNet-50 with frozen backbone (Alphabets only) |
| **MobileNetV2** | From scratch + Transfer Learning | Lightweight architecture; ImageNet-pretrained variant for Alphabets |

All models use:
- **Input**: 224×224 images normalized with ImageNet statistics
- **Augmentation**: Random rotation (±15°), affine translation, horizontal flip, color jitter
- **Training**: Adam optimizer, cross-entropy loss, mixed-precision (AMP), early stopping (patience = 5)
- **Evaluation**: Accuracy, macro F1-score, confusion matrices, classification reports

---

## Project Structure

```
ai-models/
├── notebooks/                  # Training scripts
│   ├── VGG16_ASL.py
│   ├── ResNet10_ASL.py
│   └── MobileNetV2_ASL.py
├── results/                    # Trained models, plots, and reports
│   ├── vgg16/                  #   Confusion matrices, training curves, .pth checkpoints
│   ├── resnet10/
│   ├── mobilenetv2/
│   └── vgg16_v1/               #   Earlier VGG16 experiment
├── datasets/                   # Raw & split datasets (git-ignored)
├── proposal/                   # Project proposal document
├── prepare_datasets.py         # Dataset splitting & subsampling utility
└── test_model.py               # Single-image inference script
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)

### Installation

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow
```

### 1. Prepare Datasets

Download the raw datasets into `datasets/`, then run:

```bash
python prepare_datasets.py
```

This creates stratified train/val/test splits under `datasets/asl_commands/`, `datasets/asl_digits/`, and `datasets/asl_alphabets/`.

### 2. Train Models

```bash
python notebooks/VGG16_ASL.py
python notebooks/ResNet10_ASL.py
python notebooks/MobileNetV2_ASL.py
```

Each script trains on all three datasets and saves results (model checkpoints, training curves, confusion matrices) to `results/<model>/`.

### 3. Test on a Single Image

```bash
python test_model.py --model results/mobilenetv2/alphabets_model.pth --image path/to/hand.jpg
python test_model.py --model results/resnet10/digits_model.pth --image path/to/digit.jpg --top 3
```

---

## References

1. Akash. "ASL Alphabet." Kaggle, 2018. https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Mavi, A. (2020). "A new dataset and proposed convolutional neural network architecture for classification of American Sign Language digits." *arXiv:2011.08927*
3. He, K. et al. (2016). "Deep residual learning for image recognition." *CVPR*
4. Sandler, M. et al. (2018). "MobileNetV2: Inverted residuals and linear bottlenecks." *CVPR*
