"""Internal Evaluation Script"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import load_model
from dataset import CommForImageProcessor, IMAGE_EXTS, VIDEO_EXTS
from utils import load_face_detector, crop_face_yolo, sample_video_frames, aggregate_frame_probs, calculate_auc


def main():
    BASE_DIR = Path(__file__).parent
    with open(BASE_DIR / "config" / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    model = load_model(str(BASE_DIR / "model" / "model.pt"), device)
    face_model = load_face_detector()
    processor = CommForImageProcessor(size=config.get('img_size', 384))

    val_csv = BASE_DIR / "train_data" / "val_labels.csv"
    if not val_csv.exists():
        print("[WARN] No validation CSV found!")
        return

    df = pd.read_csv(val_csv)
    all_probs, all_labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        path = BASE_DIR / "train_data" / row['filename']
        ext = path.suffix.lower()
        try:
            if ext in VIDEO_EXTS:
                frames = sample_video_frames(path, 10)
                probs = [torch.sigmoid(model(processor([crop_face_yolo(f, face_model) or f], mode='test')['pixel_values'].to(device))).item() for f in frames] if frames else [0.0]
                prob = aggregate_frame_probs(probs)
            else:
                img = Image.open(path).convert('RGB')
                face = crop_face_yolo(img, face_model) or img
                prob = torch.sigmoid(model(processor([face], mode='test')['pixel_values'].to(device))).item()
            all_probs.append(prob)
            all_labels.append(row['label'])
        except:
            all_probs.append(0.0)
            all_labels.append(row['label'])

    auc = calculate_auc(np.array(all_probs), np.array(all_labels))
    acc = np.mean((np.array(all_probs) > 0.5) == np.array(all_labels))
    print(f"\nAUC: {auc:.4f}, Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
