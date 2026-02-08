"""
Deepfake Detection Training Script
- CommFor (Community-Forensics) model architecture
- Celeb-DF v2, FaceForensics++ C23 datasets
- Focal Loss with label smoothing
"""

import sys
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import ViTClassifier
from dataset import DeepfakeDataset, TrainTransform, ValTransform, load_celeb_df, load_faceforensics
from utils import set_seed, load_face_detector, FocalLoss, calculate_auc


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_datasets(config: dict, base_dir: Path):
    print("[INFO] Loading datasets...")
    all_samples = []

    celeb_path = base_dir / "train_data" / "celeb_df_v2"
    if celeb_path.exists():
        celeb_samples = load_celeb_df(str(celeb_path))
        print(f"  Celeb-DF: {len(celeb_samples)} samples")
        all_samples.extend(celeb_samples)

    ff_path = base_dir / "train_data" / "faceforensics_c23"
    if ff_path.exists():
        ff_samples = load_faceforensics(str(ff_path))
        print(f"  FaceForensics++: {len(ff_samples)} samples")
        all_samples.extend(ff_samples)

    if not all_samples:
        raise ValueError("No training data found!")

    random.shuffle(all_samples)
    val_size = int(len(all_samples) * config.get('val_ratio', 0.1))
    return all_samples[val_size:], all_samples[:val_size]


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", ncols=100)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            outputs_2class = torch.cat([-outputs, outputs], dim=1)
            loss = criterion(outputs_2class, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        correct += ((outputs > 0).long().squeeze() == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch} [Val]", ncols=100):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs_2class = torch.cat([-outputs, outputs], dim=1)
            loss = criterion(outputs_2class, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs).squeeze()
            correct += ((outputs > 0).long().squeeze() == labels).sum().item()
            total += labels.size(0)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
    auc = calculate_auc(np.array(all_probs), np.array(all_labels))
    return total_loss / len(dataloader), 100. * correct / total, auc


def main():
    print("=" * 60)
    print("Deepfake Detection Training")
    print("=" * 60)

    BASE_DIR = Path(__file__).parent
    config = load_config(BASE_DIR / "config" / "config.yaml")
    set_seed(config.get('seed', 42))

    output_dir = BASE_DIR / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Device: {device}")

    face_model = load_face_detector()
    train_samples, val_samples = prepare_datasets(config, BASE_DIR)

    img_size = config.get('img_size', 384)
    train_dataset = DeepfakeDataset(train_samples, face_model, config, TrainTransform(img_size), True)
    val_dataset = DeepfakeDataset(val_samples, face_model, config, ValTransform(img_size), False)

    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print("[INFO] Building model...")
    model = ViTClassifier.from_pretrained('OwensLab/commfor-model-384').to(device)

    train_fake = sum(1 for _, l in train_samples if l == 1)
    weight = torch.tensor([train_fake / len(train_samples), (len(train_samples) - train_fake) / len(train_samples)]).to(device)
    criterion = FocalLoss(gamma=config.get('focal_gamma', 2.0), alpha=weight, label_smoothing=config.get('label_smoothing', 0.1))

    optimizer = AdamW(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=config.get('weight_decay', 0.01))
    epochs = config.get('epochs', 30)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs // 3, T_mult=2)
    scaler = GradScaler()

    best_auc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()
        print(f"\nEpoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, AUC={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({'model': model.state_dict(), 'val_auc': val_auc, 'config': config}, output_dir / 'model.pt')
            print(f"  -> Best model saved! (AUC: {val_auc:.4f})")

    print(f"\nTraining Complete! Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
