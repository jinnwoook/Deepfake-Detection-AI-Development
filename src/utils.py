"""
Utility Functions for Deepfake Detection
"""

import random
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image
import torch
import yaml


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_face_detector():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    print("[INFO] Loading YOLOv8 face detector...")
    model_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection",
        filename="model.pt"
    )
    return YOLO(model_path)


def crop_face_yolo(img: Union[Image.Image, np.ndarray], face_model, margin: float = 0.3) -> Optional[Image.Image]:
    from supervision import Detections
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img
    output = face_model(img_np, verbose=False)
    detections = Detections.from_ultralytics(output[0])
    if len(detections) == 0:
        return None
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in detections.xyxy]
    idx = int(np.argmax(areas))
    x1, y1, x2, y2 = detections.xyxy[idx].astype(int)
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(img_np.shape[1], x2 + mx), min(img_np.shape[0], y2 + my)
    face_crop = img_np[y1:y2, x1:x2]
    if face_crop.shape[0] < 64 or face_crop.shape[1] < 64:
        return None
    return Image.fromarray(face_crop)


def sample_video_frames(video_path: Union[str, Path], n_frames: int = 10) -> List[Image.Image]:
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return frames
        indices = np.linspace(0, total_frames - 1, num=n_frames, dtype=int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    return frames


def aggregate_frame_probs(probs: List[float], method: str = 'mean') -> float:
    if not probs:
        return 0.0
    probs = np.array(probs)
    if method == 'max':
        return float(np.max(probs))
    elif method == 'top_k':
        k = min(5, len(probs))
        return float(np.mean(sorted(probs, reverse=True)[:k]))
    return float(np.mean(probs))


def calculate_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.5
    auc_sum = sum(np.sum(p > neg_probs) + 0.5 * np.sum(p == neg_probs) for p in pos_probs)
    return auc_sum / (len(pos_probs) * len(neg_probs))


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()
