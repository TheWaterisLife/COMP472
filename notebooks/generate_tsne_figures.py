"""
generate_tsne_figures.py

Train 3 models on ASL Commands and reuse an existing MobileNetV2-Digits
checkpoint, then produce t-SNE visualizations for all four.

Outputs:
    results/tsne/*.png
    figures/*.png

Usage:  python generate_tsne_figures.py
"""

import os, sys, time, shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# Paths
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "tsne"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COMMANDS_PATH = BASE_DIR / "datasets" / "asl_commands"
DIGITS_PATH   = BASE_DIR / "datasets" / "asl_digits"

EXISTING_MOBILENET_DIGITS = (
    BASE_DIR / "results" / "mobilenetv2" / "optimization" / "bs_128_model.pth"
)

# Settings
IMG_SIZE = 224
BATCH_SIZE = 64
MAX_EPOCHS = 15
LEARNING_RATE_DEFAULT = 1e-3
LEARNING_RATE_VGG = 1e-4
EARLY_STOP_PATIENCE = 5
TSNE_MAX_SAMPLES = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def make_grad_scaler(enabled: bool):
    """Compatibility wrapper across PyTorch AMP APIs."""
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return None


def autocast_context(enabled: bool):
    """Compatibility wrapper across PyTorch autocast APIs."""
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ResNet-10 definition
class ResidualBlock(nn.Module):
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
        return self.relu(out)


class ResNet10(nn.Module):
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
        return self.fc(x)


# Model builders
def build_mobilenetv2(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def build_resnet10(num_classes):
    return ResNet10(num_classes)


def _kaiming_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(m.bias)


def build_vgg16bn(num_classes):
    model = models.vgg16_bn(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    _kaiming_init(model)
    return model


# Quick training loop
def train_quick(model, dataset_path, lr, num_classes):
    """Train a model on a dataset and return (model, class_names)."""
    train_ds = datasets.ImageFolder(str(dataset_path / "train"), train_transform)
    val_ds   = datasets.ImageFolder(str(dataset_path / "val"),   test_transform)

    nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=nw, pin_memory=True)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    use_amp = DEVICE.type == "cuda"
    scaler = make_grad_scaler(use_amp)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast_context(use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{MAX_EPOCHS} | "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f} | "
              f"{elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, train_ds.classes


# Embedding extractors (per architecture)
def extract_mobilenetv2(model, loader):
    """Extract 256-d embeddings from the hidden layer of the MobileNetV2 classifier."""
    feats, labs = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            x = model.features(images)
            x = nn.functional.adaptive_avg_pool2d(x, 1)
            x = torch.flatten(x, 1)
            # classifier: Dropout -> Linear(1280,256) -> ReLU -> Dropout -> Linear(256,C)
            x = model.classifier[0](x)   # Dropout
            x = model.classifier[1](x)   # Linear -> 256
            x = model.classifier[2](x)   # ReLU
            feats.append(x.cpu().numpy())
            labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def extract_resnet10(model, loader):
    """Extract 512-d embeddings from ResNet-10 (before fc)."""
    feats, labs = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            x = model.initial(images)
            x = model.stage1(x)
            x = model.stage2(x)
            x = model.stage3(x)
            x = model.stage4(x)
            x = model.pool(x)
            x = x.view(x.size(0), -1)    # 512-d
            feats.append(x.cpu().numpy())
            labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def extract_vgg16bn(model, loader):
    """Extract 256-d embeddings from VGG16-BN (after second hidden layer)."""
    feats, labs = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            x = model.features(images)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            # classifier: Linear->BN->ReLU->Drop->Linear->BN->ReLU->Drop->Linear
            x = model.classifier[0](x)   # Linear(25088, 512)
            x = model.classifier[1](x)   # BN
            x = model.classifier[2](x)   # ReLU
            x = model.classifier[3](x)   # Dropout
            x = model.classifier[4](x)   # Linear(512, 256)
            x = model.classifier[5](x)   # BN
            x = model.classifier[6](x)   # ReLU
            feats.append(x.cpu().numpy())
            labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


# Test-set loader with capped samples
def get_test_loader(dataset_path, max_samples=TSNE_MAX_SAMPLES):
    ds = datasets.ImageFolder(str(dataset_path / "test"), test_transform)
    per_class = max(1, max_samples // len(ds.classes))
    chosen, counts = [], {i: 0 for i in range(len(ds.classes))}
    for i, (_, label) in enumerate(ds.samples):
        if counts[label] < per_class:
            chosen.append(i)
            counts[label] += 1
        if len(chosen) >= max_samples:
            break
    subset = Subset(ds, chosen)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return loader, ds.classes


# t-SNE plotting
def plot_tsne(embeddings, labels, class_names, title, output_path):
    perplexity = min(30, max(5, len(embeddings) // 8))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
    points = tsne.fit_transform(embeddings)

    n_classes = len(class_names)
    cmap_name = "tab10" if n_classes <= 10 else "tab20"
    cmap = plt.cm.get_cmap(cmap_name, n_classes)

    plt.figure(figsize=(7, 5.5))
    for idx, name in enumerate(class_names):
        mask = labels == idx
        plt.scatter(points[mask, 0], points[mask, 1],
                    s=22, alpha=0.8, label=name, color=cmap(idx))
    plt.title(title, fontsize=12)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    ncol = 2 if n_classes <= 10 else 4
    plt.legend(ncol=ncol, fontsize=7, frameon=True, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"  Saved: {output_path}")


# Main
def main():
    models_to_visualize = []

    # 1. Train ResNet-10 on Commands (only model that needs training)
    print("\n[1/1] Training ResNet-10 on Commands …")
    res = build_resnet10(num_classes=3)
    res, res_classes = train_quick(res, COMMANDS_PATH, LEARNING_RATE_DEFAULT, 3)
    models_to_visualize.append({
        "name": "ResNet-10 — Commands",
        "filename": "tsne_resnet10_commands.png",
        "model": res,
        "extractor": extract_resnet10,
        "dataset_path": COMMANDS_PATH,
        "class_names": res_classes,
    })

    # 2–4. Load existing MobileNetV2 Digits checkpoints
    EXISTING_CHECKPOINTS = [
        ("bs_128_model.pth",  "MobileNetV2 — Digits (bs=128)",  "tsne_mobilenetv2_bs128_digits.png"),
        ("lr_1e-3_model.pth", "MobileNetV2 — Digits (lr=1e-3)", "tsne_mobilenetv2_lr1e3_digits.png"),
        ("lr_1e-2_model.pth", "MobileNetV2 — Digits (lr=1e-2)", "tsne_mobilenetv2_lr1e2_digits.png"),
    ]
    for ckpt_name, title, filename in EXISTING_CHECKPOINTS:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        print(f"\n[+] Loading {ckpt_name} …")
        ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
        model = build_mobilenetv2(ckpt["num_classes"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models_to_visualize.append({
            "name": title,
            "filename": filename,
            "model": model,
            "extractor": extract_mobilenetv2,
            "dataset_path": DIGITS_PATH,
            "class_names": ckpt["class_names"],
        })

    # Generate t-SNE for all 4
    print("\nGenerating t-SNE plots")
    for entry in models_to_visualize:
        print(f"\n  {entry['name']}")
        loader, _ = get_test_loader(entry["dataset_path"])
        embeddings, labels = entry["extractor"](entry["model"], loader)

        result_path = RESULTS_DIR / entry["filename"]
        figure_path = FIGURES_DIR / entry["filename"]

        plot_tsne(embeddings, labels, entry["class_names"],
                  f"t-SNE: {entry['name']}", result_path)
        shutil.copy2(result_path, figure_path)

    print("\n✓ Done — 4 t-SNE figures saved to figures/")


if __name__ == "__main__":
    main()
