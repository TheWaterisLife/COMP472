"""
MobileNetV2_Optimization.py — Hyperparameter Tuning on ASL Digits

Varies learning rate, batch size, and loss function one-at-a-time
around the baseline (lr=1e-3, batch=64, CrossEntropyLoss).

Usage:  python notebooks/MobileNetV2_Optimization.py
"""

import os, sys, time, json
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configuration 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'asl_digits')
IMG_SIZE = 224
# On Windows, DataLoader worker processes re-import torch and can easily
# trigger WinError 1455 (paging file too small). Default to 0 for stability.
NUM_WORKERS = 0 if os.name == 'nt' else 4
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()  
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'mobilenetv2', 'optimization')
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Baseline hyperparameters 
BASELINE = {
    'lr': 1e-3,
    'batch_size': 64,
    'loss_fn': 'cross_entropy',
}

# Experiment definitions (one-at-a-time) 
EXPERIMENTS = [
    # Learning rate sweep
    {'name': 'lr_1e-2',  'lr': 1e-2, 'batch_size': 64, 'loss_fn': 'cross_entropy'},
    {'name': 'lr_1e-3',  'lr': 1e-3, 'batch_size': 64, 'loss_fn': 'cross_entropy'},  # baseline
    {'name': 'lr_1e-4',  'lr': 1e-4, 'batch_size': 64, 'loss_fn': 'cross_entropy'},
    {'name': 'lr_1e-5',  'lr': 1e-5, 'batch_size': 64, 'loss_fn': 'cross_entropy'},
    # Batch size sweep
    {'name': 'bs_16',    'lr': 1e-3, 'batch_size': 16,  'loss_fn': 'cross_entropy'},
    {'name': 'bs_32',    'lr': 1e-3, 'batch_size': 32,  'loss_fn': 'cross_entropy'},
    {'name': 'bs_128',   'lr': 1e-3, 'batch_size': 128, 'loss_fn': 'cross_entropy'},
    # Loss function sweep
    {'name': 'label_smoothing_0.1', 'lr': 1e-3, 'batch_size': 64, 'loss_fn': 'label_smoothing'},
]


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


def create_loaders(batch_size):
    train_ds = datasets.ImageFolder(os.path.join(DATASET_PATH, 'train'), get_transforms(augment=True))
    val_ds   = datasets.ImageFolder(os.path.join(DATASET_PATH, 'val'),   get_transforms())
    test_ds  = datasets.ImageFolder(os.path.join(DATASET_PATH, 'test'),  get_transforms())

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_ds.classes


# Model Builder 
def build_mobilenetv2(num_classes=10):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


# Loss Function Factory 
def get_loss_fn(loss_name):
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'label_smoothing':
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# Training Loop 
def train_model(model, train_loader, val_loader, lr, loss_fn_name,
                epochs=MAX_EPOCHS, patience=EARLY_STOP_PATIENCE):
    criterion = get_loss_fn(loss_fn_name)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        t0 = time.time()

        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
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

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
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

        print(f"    Epoch {epoch+1:2d}/{epochs} — "
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
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return history


# Evaluation 
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    report = classification_report(y_true, y_pred, output_dict=True)
    acc = accuracy_score(y_true, y_pred)
    return {
        'accuracy': acc,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'y_true': y_true,
        'y_pred': y_pred,
    }


#  Comparison Plots 
def plot_lr_comparison(all_results):
    """Overlaid training curves for learning rate experiments."""
    lr_exps = [r for r in all_results if r['name'].startswith('lr_')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in lr_exps:
        label = r['name'].replace('lr_', 'lr=')
        axes[0].plot(r['history']['val_accuracy'], label=label)
        axes[1].plot(r['history']['val_loss'], label=label)

    axes[0].set_title('Validation Accuracy — Learning Rate Comparison')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True)

    axes[1].set_title('Validation Loss — Learning Rate Comparison')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'comparison_learning_rate.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_bs_comparison(all_results):
    """Overlaid training curves for batch size experiments."""
    bs_exps = [r for r in all_results if r['name'].startswith('bs_')]

    baseline = next((r for r in all_results if r['name'] == 'lr_1e-3'), None)
    if baseline:
        bs_exps.append({**baseline, 'name': 'bs_64'})
    bs_exps.sort(key=lambda x: int(x['name'].split('_')[1]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in bs_exps:
        label = r['name'].replace('bs_', 'batch=')
        axes[0].plot(r['history']['val_accuracy'], label=label)
        axes[1].plot(r['history']['val_loss'], label=label)

    axes[0].set_title('Validation Accuracy — Batch Size Comparison')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True)

    axes[1].set_title('Validation Loss — Batch Size Comparison')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'comparison_batch_size.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_loss_fn_comparison(all_results):
    """Overlaid training curves for loss function experiments."""
    baseline = next((r for r in all_results if r['name'] == 'lr_1e-3'), None)
    ls_exp = next((r for r in all_results if r['name'] == 'label_smoothing_0.1'), None)
    if not baseline or not ls_exp:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r, label in [(baseline, 'CrossEntropy'), (ls_exp, 'LabelSmoothing(0.1)')]:
        axes[0].plot(r['history']['val_accuracy'], label=label)
        axes[1].plot(r['history']['val_loss'], label=label)

    axes[0].set_title('Validation Accuracy — Loss Function Comparison')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True)

    axes[1].set_title('Validation Loss — Loss Function Comparison')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'comparison_loss_function.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_bar_chart(all_results):
    """Bar chart comparing test accuracy across all experiments."""
    names = [r['name'] for r in all_results]
    accs = [r['metrics']['accuracy'] * 100 for r in all_results]

    # Color code by experiment type
    colors = []
    for name in names:
        if name.startswith('lr_'):
            colors.append('#2196F3')   
        elif name.startswith('bs_'):
            colors.append('#4CAF50')   
        else:
            colors.append('#FF9800')   

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(names)), accs, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('MobileNetV2 on ASL Digits — Hyperparameter Optimization Results')
    ax.set_ylim(max(0, min(accs) - 5), 100)
    ax.grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='Learning Rate'),
        Patch(facecolor='#4CAF50', label='Batch Size'),
        Patch(facecolor='#FF9800', label='Loss Function'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'accuracy_bar_chart.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_best_confusion_matrix(best_result, class_names):
    """Confusion matrix for the best-performing experiment."""
    cm = confusion_matrix(best_result['metrics']['y_true'], best_result['metrics']['y_pred'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — Best Config: {best_result['name']}\n"
              f"(Accuracy: {best_result['metrics']['accuracy']:.4f})")
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'best_confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# Main function 
if __name__ == '__main__':
    all_results = []
    class_names = None

    print(f"\n{'='*70}")
    print(f"  MobileNetV2 Hyperparameter Optimization — ASL Digits")
    print(f"  Baseline: lr=1e-3, batch_size=64, CrossEntropyLoss")
    print(f"{'='*70}")

    for exp in EXPERIMENTS:
        name = exp['name']
        lr = exp['lr']
        bs = exp['batch_size']
        loss_fn = exp['loss_fn']

        print(f"\n{'─'*60}")
        print(f"  Experiment: {name}")
        print(f"  lr={lr}, batch_size={bs}, loss={loss_fn}")
        print(f"{'─'*60}")

        train_loader, val_loader, test_loader, classes = create_loaders(bs)
        if class_names is None:
            class_names = classes

        model = build_mobilenetv2(num_classes=10)
        t_start = time.time()
        history = train_model(model, train_loader, val_loader, lr=lr, loss_fn_name=loss_fn)
        train_time = time.time() - t_start

        metrics = evaluate_model(model, test_loader)

        print(f"\n  >> Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"  >> Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}")
        print(f"  >> Training Time: {train_time:.1f}s")

        # Save model
        model_path = os.path.join(RESULTS_DIR, f'{name}_model.pth')
        torch.save({
            'arch': 'mobilenetv2',
            'experiment': name,
            'hyperparams': {'lr': lr, 'batch_size': bs, 'loss_fn': loss_fn},
            'num_classes': 10,
            'class_names': classes,
            'state_dict': model.state_dict(),
        }, model_path)

        all_results.append({
            'name': name,
            'lr': lr,
            'batch_size': bs,
            'loss_fn': loss_fn,
            'history': history,
            'metrics': metrics,
            'train_time': train_time,
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary Table 
    print(f"\n\n{'='*90}")
    print(f"  OPTIMIZATION SUMMARY — MobileNetV2 on ASL Digits")
    print(f"{'='*90}")
    print(f"  {'Experiment':<25} {'LR':<10} {'Batch':<8} {'Loss Fn':<18} {'Acc':<8} {'F1':<8} {'Time':<8}")
    print(f"  {'─'*85}")
    for r in all_results:
        marker = " <-- baseline" if r['name'] == 'lr_1e-3' else ""
        print(f"  {r['name']:<25} {r['lr']:<10} {r['batch_size']:<8} {r['loss_fn']:<18} "
              f"{r['metrics']['accuracy']:.4f}   {r['metrics']['f1']:.4f}   {r['train_time']:.1f}s{marker}")

    best = max(all_results, key=lambda x: x['metrics']['accuracy'])
    print(f"\n  BEST: {best['name']} — Accuracy: {best['metrics']['accuracy']:.4f}, F1: {best['metrics']['f1']:.4f}")

    # Comparison Plots 
    print(f"\nGenerating comparison plots...")
    plot_lr_comparison(all_results)
    plot_bs_comparison(all_results)
    plot_loss_fn_comparison(all_results)
    plot_accuracy_bar_chart(all_results)
    plot_best_confusion_matrix(best, class_names)

    # Save summary as JSON 
    summary = []
    for r in all_results:
        summary.append({
            'name': r['name'],
            'lr': r['lr'],
            'batch_size': r['batch_size'],
            'loss_fn': r['loss_fn'],
            'test_accuracy': r['metrics']['accuracy'],
            'precision': r['metrics']['precision'],
            'recall': r['metrics']['recall'],
            'f1': r['metrics']['f1'],
            'train_time_seconds': round(r['train_time'], 1),
            'epochs_trained': len(r['history']['loss']),
        })
    summary_path = os.path.join(RESULTS_DIR, 'optimization_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    print(f"\n  All results saved to: {RESULTS_DIR}")
