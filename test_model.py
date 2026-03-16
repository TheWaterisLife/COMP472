"""
test_model.py — Test a trained ASL model on individual images
==============================================================
Load a saved .pth checkpoint and predict the ASL sign from an image.

Usage:
  python test_model.py --model results/mobilenetv2/alphabets_model.pth --image path/to/hand.jpg
  python test_model.py --model results/resnet10/digits_model.pth --image path/to/digit.jpg
  python test_model.py --model results/vgg16/alphabets_tl_model.pth --image path/to/hand.jpg

Options:
  --top N   Show top N predictions (default: 5)
"""

import argparse, os, sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ── Import ResNet10 architecture (needed to rebuild the model) ──────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))
from ResNet10_ASL import ResNet10


# ── Model Builders ──────────────────────────────────────────────────────────
def build_model(arch, num_classes):
    """Rebuild the model architecture given the arch string saved in the checkpoint."""
    if arch in ('mobilenetv2', 'mobilenetv2_tl'):
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    elif arch in ('resnet10',):
        model = ResNet10(num_classes)
    elif arch in ('resnet50_tl',):
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    elif arch in ('vgg16', 'vgg16_tl'):
        model = models.vgg16(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


# ── Image Preprocessing ────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(model, image_path, class_names, device, top_k=5):
    """Run inference on a single image and return top-k predictions."""
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            logits = model(tensor)

    probs = torch.softmax(logits.float(), dim=1).squeeze()
    top_probs, top_idxs = probs.topk(min(top_k, len(class_names)))

    results = []
    for prob, idx in zip(top_probs, top_idxs):
        results.append((class_names[idx.item()], prob.item() * 100))
    return results


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Test a trained ASL model on an image')
    parser.add_argument('--model', required=True, help='Path to saved .pth checkpoint')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--top', type=int, default=5, help='Show top N predictions')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading model: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    arch = checkpoint['arch']
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']

    print(f"Architecture: {arch}")
    print(f"Classes ({num_classes}): {class_names}")

    # Rebuild model and load weights
    model = build_model(arch, num_classes)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    # Run prediction
    print(f"\nPredicting: {args.image}")
    print(f"{'='*40}")

    results = predict(model, args.image, class_names, device, top_k=args.top)

    for i, (label, confidence) in enumerate(results):
        bar = '█' * int(confidence / 2)
        print(f"  {i+1}. {label:>12s}  {confidence:5.1f}%  {bar}")

    top_label, top_conf = results[0]
    print(f"\n  → Predicted: {top_label} ({top_conf:.1f}% confidence)")


if __name__ == '__main__':
    main()
