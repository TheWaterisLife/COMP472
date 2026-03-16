# Project Proposal: American Sign Language Recognition Using Convolutional Neural Networks

**Course:** COMP 472: Applied Artificial Intelligence  
**Institution:** Concordia University  
**Term:** Winter 2026

---

## 1. Problem Statement & Application

American Sign Language (ASL) serves as the primary communication medium for approximately 500,000 deaf and hard-of-hearing individuals in North America. Despite its widespread use, a significant communication barrier persists between ASL users and the hearing population, limiting accessibility in educational, healthcare, and professional settings. Automated ASL recognition systems have the potential to bridge this gap by enabling real-time translation and fostering more inclusive interactions.

This project addresses the **image classification problem of recognizing static ASL hand gestures** using Convolutional Neural Networks (CNNs). The task involves classifying input images of hand signs into their corresponding semantic categories (digits, letters, or phrases).

### Key Challenges

- **Intra-class Variability**: Hand shape, size, and positioning vary significantly across individuals.
- **Environmental Factors**: Variations in lighting conditions, background clutter, and camera angles affect image quality.
- **Inter-class Similarity**: Certain signs (e.g., letters 'M' and 'N', or digits '6' and '9') share visual similarities, complicating classification.
- **Dataset Bias**: Models may overfit to specific skin tones, hand orientations, or acquisition conditions present in training data.

---

## 2. Image Dataset Selection

Three publicly available datasets were selected to represent varying levels of classification complexity while satisfying the course constraint of fewer than 50,000 images per dataset.

| Dataset | Classes | Images | Resolution | Source |
|---------|---------|--------|------------|--------|
| ASL Commands | 3 | ~9,000 | 200×200 RGB | [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| Sign Language Digits | 10 | 2,062 | 100×100 RGB | [GitHub](https://github.com/ardamavi/Sign-Language-Digits-Dataset) |
| ASL Alphabet | 26 | ~39,000* | 200×200 RGB | [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |

*\*Subsampled from ~78,000 letter images to meet the 50,000 image constraint (~1,500 images/class).*

### Dataset 1: ASL Commands Dataset (Low Complexity)

This dataset is a subset of the ASL Alphabet dataset, containing three utility gesture classes: *Space*, *Delete*, and *Nothing*, with approximately 3,000 images per class (~9,000 total). Images are 200×200 pixels in RGB format. The limited number of visually distinct classes makes this suitable for baseline model development and initial CNN experimentation. [1]

### Dataset 2: Sign Language Digits Dataset (Medium Complexity)

Created by students at Turkey Ankara Ayrancı Anadolu High School, this dataset includes 10 classes representing digits 0–9 with contributions from 218 participants. The uniform 100×100 RGB format and balanced class distribution provide a controlled environment for CNN experimentation.

**Citation:** Mavi, A. (2020). "A new dataset and proposed convolutional neural network architecture for classification of American Sign Language digits." *arXiv:2011.08927*

### Dataset 3: ASL Alphabet Dataset (High Complexity)

This widely-used Kaggle dataset by grassknoted originally contains 29 classes. For this dataset, we use only the 26 alphabet letter classes (A–Z), excluding the three utility classes (*space*, *delete*, *nothing*) which are used in Dataset 1. The original letter subset contains ~78,000 images at 200×200 RGB resolution (~3,000 per class). To comply with course requirements, we will perform stratified random subsampling to ~1,500 images per class, yielding approximately 39,000 total images while maintaining class balance. [3]

---

## 3. Proposed Methodology

### Implementation Framework

**PyTorch** will serve as the primary deep learning framework due to its flexibility, dynamic computation graphs, and extensive community support for computer vision tasks.

### Data Preprocessing Pipeline

1. **Resizing**: All images normalized to 224×224 pixels to match pre-trained model input requirements.
2. **Normalization**: Pixel values scaled using ImageNet statistics (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]).
3. **Data Augmentation**: Random horizontal flips, rotation (±15°), and color jittering to improve generalization.
4. **Train/Validation/Test Split**: 70%/15%/15% stratified partitioning to preserve class distributions.

### Model Architectures

| Model | Description |
|-------|-------------|
| **Baseline CNN** | Custom architecture with 3–4 convolutional blocks for establishing performance benchmarks |
| **ResNet-18/ResNet-50** | Residual networks with skip connections to address vanishing gradient issues |
| **MobileNetV2** | Lightweight architecture with depthwise separable convolutions for efficient deployment |

Transfer learning with ImageNet pre-trained weights will be employed, followed by fine-tuning on the target ASL datasets.

### Evaluation Metrics

- **Accuracy**: Overall classification correctness across all classes.
- **Macro F1-Score**: Harmonic mean of precision and recall, averaged across classes to account for potential class imbalance.
- **Confusion Matrix**: Visual analysis of per-class performance and common misclassification patterns.
- **ROC-AUC**: Area under the receiver operating characteristic curve for multi-class evaluation.

---

## References

1. Mitchell, R. E., Young, T. A., Bachleda, B., & Karchmer, M. A. (2006). "How many people use ASL in the United States? Why estimates need updating." *Sign Language Studies*, 6(3), 306–335.

2. Akash. "ASL Alphabet." Kaggle, 2018. https://www.kaggle.com/datasets/grassknoted/asl-alphabet (ASL Commands subset: Space, Delete, Nothing classes)

3. Mavi, A. (2020). "A new dataset and proposed convolutional neural network architecture for classification of American Sign Language digits." *arXiv:2011.08927*. https://github.com/ardamavi/Sign-Language-Digits-Dataset

4. Akash. "ASL Alphabet." Kaggle, 2018. https://www.kaggle.com/datasets/grassknoted/asl-alphabet

5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning for image recognition." *CVPR*, 770–778.

6. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). "MobileNetV2: Inverted residuals and linear bottlenecks." *CVPR*, 4510–4520.

---

## Important Submission Reminders

- **LaTeX is Mandatory**: Use the professor's official LaTeX template for final submission.
- **Gantt Chart**: Include a separate page with the project timeline (see `Gantt_Chart.md`).
- **Email Subject Format**: `[COMP472: YOUR SUBJECT]` when contacting Dr. Mahdi S. Hosseini or Rose Rostami.
