"""
ResNet10_ASL.py — Train custom ResNet-10 on ASL datasets (local GPU)

Trains a custom ResNet-10 on 3 datasets: Commands, Digits, Alphabets.
Includes a transfer-learning variant using pretrained ResNet-50.

Usage:  python notebooks/ResNet10_ASL.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    'commands': {
        'name': 'ASL Commands',
        'path': os.path.join(BASE_DIR, 'datasets', 'asl_commands'),
        'num_classes': 3,
    },
    'digits': {
        'name': 'ASL Digits',
        'path': os.path.join(BASE_DIR, 'datasets', 'asl_digits'),
        'num_classes': 10,
    },
    'alphabets': {
        'name': 'ASL Alphabets',
        'path': os.path.join(BASE_DIR, 'datasets', 'asl_alphabets'),
        'num_classes': 26,
    },
}

IMG_SIZE = 224
BATCH_SIZE = 64
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 5
NUM_WORKERS = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'resnet10')
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
# Data Loading
def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def create_loaders(dataset_path):
    train_ds = datasets.ImageFolder(os.path.join(dataset_path, 'train'), get_transforms(augment=True))
    val_ds   = datasets.ImageFolder(os.path.join(dataset_path, 'val'),   get_transforms())
    test_ds  = datasets.ImageFolder(os.path.join(dataset_path, 'test'),  get_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    print(f"  Train: {len(train_ds)} images | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Classes: {train_ds.classes}")
    return train_loader, val_loader, test_loader, train_ds.classes
# ResNet-10 Architecture
class ResidualBlock(nn.Module):
    """Basic residual block with two 3×3 convs and a skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet10(nn.Module):
    """
    ResNet-10: 4 stages × 1 block each = ~10 layers.
    Stage filters: [64, 128, 256, 512]
    """
    def __init__(self, num_classes):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = ResidualBlock(64,  64,  stride=1)
        self.stage2 = ResidualBlock(64,  128, stride=2)
        self.stage3 = ResidualBlock(128, 256, stride=2)
        self.stage4 = ResidualBlock(256, 512, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def build_resnet10(num_classes):
    return ResNet10(num_classes).to(DEVICE)


def build_resnet50_transfer(num_classes):
    """ResNet-50 with ImageNet weights, frozen base + custom head."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)
# Training
def train_model(model, train_loader, val_loader, lr=LEARNING_RATE,
                epochs=MAX_EPOCHS, patience=EARLY_STOP_PATIENCE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = torch.amp.GradScaler('cuda')

    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        t0 = time.time()

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        elapsed = time.time() - t0
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(f"  Epoch {epoch+1:2d}/{epochs} — "
              f"loss: {train_loss:.4f}  acc: {train_acc:.4f}  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}  "
              f"[{elapsed:.1f}s]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return history
# Plotting & Evaluation
def plot_history(history, title, save_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['accuracy'], label='Train')
    axes[0].plot(history['val_accuracy'], label='Val')
    axes[0].set_title(f'{title} — Accuracy')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['loss'], label='Train')
    axes[1].plot(history['val_loss'], label='Val')
    axes[1].set_title(f'{title} — Loss')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'{save_name}_history.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def evaluate_model(model, test_loader, class_names, title, save_name):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)

    print(f"\n{'='*60}")
    print(f"  Classification Report — {title}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(8, len(class_names)*0.5), max(6, len(class_names)*0.4)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix — {title}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'{save_name}_confusion.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    acc = (y_true == y_pred).mean()
    print(f"\n  Test Accuracy: {acc:.4f}")
    return y_true, y_pred


def save_model(model, class_names, save_name, arch='resnet10', num_classes=None):
    """Save model checkpoint with metadata for later inference."""
    path = os.path.join(RESULTS_DIR, f'{save_name}_model.pth')
    torch.save({
        'arch': arch,
        'num_classes': num_classes or len(class_names),
        'class_names': class_names,
        'state_dict': model.state_dict(),
    }, path)
    print(f"  ✓ Model saved: {path}")
# MAIN

if __name__ == '__main__':

    for ds_key in ['commands', 'digits', 'alphabets']:
        ds = DATASETS[ds_key]
        print(f"\n{'='*60}")
        print(f"  ResNet-10 — {ds['name']}")
        print(f"{'='*60}")

        train_loader, val_loader, test_loader, classes = create_loaders(ds['path'])

        model = build_resnet10(ds['num_classes'])
        history = train_model(model, train_loader, val_loader)

        plot_history(history, f"ResNet-10 — {ds['name']}", f"{ds_key}")
        evaluate_model(model, test_loader, classes,
                       f"ResNet-10 — {ds['name']}", f"{ds_key}")
        save_model(model, classes, ds_key, arch='resnet10', num_classes=ds['num_classes'])

        del model
        torch.cuda.empty_cache()
# Transfer Learning: ResNet-50 (ImageNet) → Alphabets
    ds = DATASETS['alphabets']
    print(f"\n{'='*60}")
    print(f"  ResNet-50 Transfer Learning — {ds['name']}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, classes = create_loaders(ds['path'])

    model_tl = build_resnet50_transfer(ds['num_classes'])
    history_tl = train_model(model_tl, train_loader, val_loader)

    plot_history(history_tl, f"ResNet-50 TL — {ds['name']}", "alphabets_tl")
    evaluate_model(model_tl, test_loader, classes,
                   f"ResNet-50 Transfer Learning — {ds['name']}", "alphabets_tl")
    save_model(model_tl, classes, 'alphabets_tl', arch='resnet50_tl', num_classes=ds['num_classes'])

    print(f"\n✓ All done! Results saved to: {RESULTS_DIR}")
