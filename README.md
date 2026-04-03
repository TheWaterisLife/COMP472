# ASL Recognition Using Convolutional Neural Networks

**COMP 472 — Applied Artificial Intelligence | Concordia University | Winter 2026**

This repository is the course project deliverable: a PyTorch implementation that compares several convolutional neural networks for American Sign Language (ASL) hand-gesture recognition. Models are trained and evaluated on three tasks of increasing difficulty (commands, digits, full alphabet), including transfer-learning variants where noted.

---

## Repository & course access

Per course instructions, the project should live in a **private** GitHub repository. All team members should commit updates regularly. The assigned **TA** and the **lecturer** must be granted access so progress can be reviewed; individual commit activity may factor into grading.

**Collaborators to add on GitHub (invite with appropriate access per your TA’s instructions):**

| Role        | GitHub username      |
|------------|----------------------|
| Professor  | `TBD`  |
| Lead TA    | `ChrisLeclerc` |



---

## Project overview

- **Goal:** Classify ASL gestures from RGB images using CNNs implemented in **PyTorch**.
- **Tasks:** Three datasets—**ASL Commands** (3 classes), **ASL Digits** (10 classes), **ASL Alphabets** (26 classes)—each split **80% / 10% / 10%** (train / validation / test) with stratified sampling.
- **Models:** **VGG-16** (from scratch + transfer learning on alphabets), **custom ResNet-10** (from scratch; script also trains **ResNet-50** transfer learning on alphabets), **MobileNetV2** (from scratch + transfer learning on alphabets).
- **Training:** Adam, cross-entropy, mixed precision (AMP) when CUDA is available, augmentations (rotation, translation, flip, color jitter), early stopping (patience 5). **Validation** runs each epoch inside the training scripts; metrics include accuracy, macro F1, confusion matrices, and classification reports.
- **Outputs:** Curves, confusion matrices, and text reports under `results/<model>/`; compatible `.pth` checkpoints for inference with `test_model.py`.

---

## Obtaining the dataset

You need two public sources. Download and extract them so the **folder names and nesting** match what `prepare_datasets.py` expects (paths are relative to the repository root).

### 1. ASL Alphabet (Kaggle) — commands & alphabets

- **Download:** [ASL Alphabet dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (download the archive and unzip).
- **Required layout:** Place the material under:

  `datasets/ASL-Alphabet/asl_alphabet_train/asl_alphabet_train/`

  Inside that folder you should have one subdirectory **per class** (e.g. `A`, `B`, …, and `del`, `nothing`, `space` for commands). The preparation script reads class images from those subdirectories.

### 2. Sign Language Digits — digits 0–9

- **Download:** [Sign-Language-Digits-Dataset on GitHub](https://github.com/ardamavi/Sign-Language-Digits-Dataset) (clone or download ZIP).
- **Required layout:** Ensure the digit images live under:

  `datasets/Sign-Language-Digits-Dataset/Dataset/`

  with subfolders `0`, `1`, …, `9`.

### 3. Build train / validation / test splits

From the repository root, after the two sources are in place:

```bash
python prepare_datasets.py
```

This creates:

- `datasets/asl_commands/{train,val,test}/<class>/`
- `datasets/asl_digits/{train,val,test}/<class>/`
- `datasets/asl_alphabets/{train,val,test}/<class>/` (alphabet classes subsampled to 1500 images per class by default)

If any path is wrong, the script prints warnings for missing class folders—fix the directory layout and run again.

---

## Requirements (environment)

- **Python:** 3.10 or newer recommended.
- **Hardware:** CPU runs but is slow; a **CUDA-capable GPU** is strongly recommended for training.
- **Python packages** (install in a virtual environment):

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow numpy
```

Use the [official PyTorch install page](https://pytorch.org/get-started/locally/) if you need a specific CUDA build of `torch` / `torchvision`.

---

## PyTorch source code layout

| Path | Purpose |
|------|---------|
| `prepare_datasets.py` | Stratified split and copy into `datasets/asl_*` |
| `notebooks/VGG16_ASL.py` | Train VGG-16 on all three tasks (+ VGG transfer on alphabets) |
| `notebooks/ResNet10_ASL.py` | Train ResNet-10 on all tasks; ResNet-50 transfer on alphabets; defines `ResNet10` (used by `test_model.py`) |
| `notebooks/MobileNetV2_ASL.py` | Train MobileNetV2 on all tasks (+ transfer on alphabets) |
| `notebooks/MobileNetV2_Optimization.py` | Additional MobileNetV2 experiments |
| `notebooks/generate_tsne_figures.py` | t-SNE visualization helper |
| `test_model.py` | Load a saved `.pth` checkpoint and run inference on one image |
| `results/` | Training plots, reports, and saved checkpoints (e.g. `results/vgg16/`, `results/resnet10/`, `results/mobilenetv2/`) |

---

## How to train and validate

1. Complete **Obtaining the dataset** and run `prepare_datasets.py` so `datasets/asl_commands`, `datasets/asl_digits`, and `datasets/asl_alphabets` exist.
2. From the repository root, run the desired training script (validation is performed automatically each epoch; outputs go to `results/<model>/`):

```bash
python notebooks/VGG16_ASL.py
python notebooks/ResNet10_ASL.py
python notebooks/MobileNetV2_ASL.py
```

3. Optional: `python notebooks/MobileNetV2_Optimization.py` for further MobileNetV2 runs.

Each script prints progress to the console and writes figures and metrics under the corresponding `results/` subdirectory. Checkpoints are usually named `{task}_model.pth` with `task` in `commands`, `digits`, `alphabets`, plus transfer checkpoints such as `alphabets_tl_model.pth`. Confirm filenames in `results/<architecture_folder>/` after a run.

---

## How to run a trained model on sample test images

You need a **checkpoint** (`.pth`) produced by training and **`test_model.py`**. The repository may already contain checkpoints under `results/`; if not, train first as above.

**Example:** run inference on a single image from the prepared **test** split (adjust the checkpoint path to a file that exists on your machine):

```bash
python test_model.py --model results/mobilenetv2/alphabets_model.pth --image datasets/asl_alphabets/test/A/some_image.jpg
```

Other examples:

```bash
python test_model.py --model results/resnet10/digits_model.pth --image datasets/asl_digits/test/3/example.png --top 3
python test_model.py --model results/vgg16/alphabets_tl_model.pth --image datasets/asl_alphabets/test/M/example.jpg
```

**Arguments:**

- `--model` — path to the saved `.pth` file (required).
- `--image` — path to one RGB image file (required).
- `--top` — show top-*N* predictions (default: 5).

The script loads `arch`, `num_classes`, and `class_names` from the checkpoint, rebuilds the network, and prints ranked predictions. If loading fails, verify that the checkpoint matches the code version and that `notebooks/ResNet10_ASL.py` is present (needed for ResNet-10 checkpoints).

---

## Datasets summary

| Dataset | Classes | Approx. scale | Notes |
|---------|:-------:|:-------------:|-------|
| ASL Commands | 3 | ~9,000 | `del`, `nothing`, `space` (sourced from ASL Alphabet train split) |
| ASL Digits | 10 | ~2,062 | Digits 0–9 |
| ASL Alphabets | 26 | ~39,000 → subsampled | A–Z, **1500 images per class** after `prepare_datasets.py` |

---

## References

1. Akash. "ASL Alphabet." Kaggle, 2018. https://www.kaggle.com/datasets/grassknoted/asl-alphabet  
2. Mavi, A. (2020). "A new dataset and proposed convolutional neural network architecture for classification of American Sign Language digits." *arXiv:2011.08927*  
3. He, K. et al. (2016). "Deep residual learning for image recognition." *CVPR*  
4. Sandler, M. et al. (2018). "MobileNetV2: Inverted residuals and linear bottlenecks." *CVPR*
