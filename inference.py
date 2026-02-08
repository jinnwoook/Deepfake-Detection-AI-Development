"""
Deepfake Detection Inference Script
- Processes images and videos (frame-by-frame independently)
- Uses YOLOv8 for face detection
- Uses CommFor ViT model for classification
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import load_model
from dataset import CommForImageProcessor, IMAGE_EXTS, VIDEO_EXTS
from utils import load_face_detector, crop_face_yolo, sample_video_frames, aggregate_frame_probs


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def predict_single_image(img, model, processor, face_model, device='cuda'):
    face = crop_face_yolo(img, face_model)
    if face is None:
        face = img
    inputs = processor([face], mode='test')
    pixel_values = inputs['pixel_values'].to(device)
    if pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)
    with torch.no_grad():
        logit = model(pixel_values)
        return torch.sigmoid(logit).item()


def predict_video(video_path, model, processor, face_model, n_frames=10, aggregation='mean', device='cuda'):
    frames = sample_video_frames(video_path, n_frames)
    if not frames:
        return 0.0
    probs = [predict_single_image(f, model, processor, face_model, device) for f in frames]
    return aggregate_frame_probs(probs, method=aggregation)


def main():
    BASE_DIR = Path(__file__).parent
    CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
    MODEL_PATH = BASE_DIR / "model" / "model.pt"
    TEST_DATA_DIR = BASE_DIR / "test_data"

    config = load_config(CONFIG_PATH)
    INPUT_CSV = TEST_DATA_DIR / "sample_submission.csv"
    OUTPUT_CSV = BASE_DIR / "submission.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    torch.set_grad_enabled(False)

    print("[INFO] Loading model...")
    model = load_model(str(MODEL_PATH), device)
    print("[INFO] Loading face detector...")
    face_model = load_face_detector()
    processor = CommForImageProcessor(size=config.get('img_size', 384))

    print(f"[INFO] Reading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()
    if 'prob' not in df.columns:
        df['prob'] = 0.0

    print(f"[INFO] Total files: {len(df)}")
    n_frames = config.get('n_frames', 10)
    aggregation = config.get('aggregation', 'mean')

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference", ncols=100):
        filename = row['filename']
        file_path = TEST_DATA_DIR / filename
        ext = file_path.suffix.lower()
        try:
            if not file_path.exists():
                df.at[idx, 'prob'] = 0.0
                continue
            if ext in IMAGE_EXTS:
                img = Image.open(file_path).convert("RGB")
                prob = predict_single_image(img, model, processor, face_model, device)
            elif ext in VIDEO_EXTS:
                prob = predict_video(file_path, model, processor, face_model, n_frames, aggregation, device)
            else:
                prob = 0.0
            df.at[idx, 'prob'] = prob
        except Exception as e:
            print(f"[ERR] {filename}: {e}")
            df.at[idx, 'prob'] = 0.0

    print(f"[INFO] Saving to {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    probs = df['prob'].values
    print(f"\n[Statistics] Mean: {probs.mean():.4f}, Fake(>0.5): {(probs > 0.5).sum()}/{len(probs)}")
    print("Done!")


if __name__ == "__main__":
    main()
