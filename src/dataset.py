"""
Dataset and Data Processing for Deepfake Detection
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as TF

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


class CommForImageProcessor:
    """Image processor for CommFor model"""

    def __init__(self, size: int = 384):
        self.size = size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, images: List[Image.Image], mode: str = 'test') -> dict:
        processed = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = TF.resize(img, (440, 440))
            img = TF.center_crop(img, (self.size, self.size))
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=self.mean, std=self.std)
            processed.append(img)
        return {'pixel_values': torch.stack(processed)}


class TrainTransform:
    def __init__(self, img_size: int = 384):
        self.img_size = img_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = TF.resize(img, (self.img_size, self.img_size))
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(img)
        if random.random() > 0.7:
            img = TF.gaussian_blur(img, kernel_size=[random.choice([3, 5])] * 2)
        if random.random() > 0.5:
            img = TF.rotate(img, random.uniform(-15, 15))
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.mean, std=self.std)
        return img


class ValTransform:
    def __init__(self, img_size: int = 384):
        self.img_size = img_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = TF.resize(img, (self.img_size, self.img_size))
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.mean, std=self.std)
        return img


class DeepfakeDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], face_detector, config: dict,
                 transform=None, is_train: bool = True):
        self.samples = samples
        self.face_detector = face_detector
        self.config = config
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def crop_face(self, img_np: np.ndarray, margin: float = 0.3) -> np.ndarray:
        try:
            from supervision import Detections
            output = self.face_detector(img_np, verbose=False)
            detections = Detections.from_ultralytics(output[0])
            if len(detections) == 0:
                return img_np
            areas = [(box[2]-box[0])*(box[3]-box[1]) for box in detections.xyxy]
            idx = int(np.argmax(areas))
            x1, y1, x2, y2 = detections.xyxy[idx].astype(int)
            h, w = img_np.shape[:2]
            face_w, face_h = x2 - x1, y2 - y1
            mx, my = int(face_w * margin), int(face_h * margin)
            x1, y1 = max(0, x1 - mx), max(0, y1 - my)
            x2, y2 = min(w, x2 + mx), min(h, y2 + my)
            face_crop = img_np[y1:y2, x1:x2]
            if face_crop.shape[0] < 64 or face_crop.shape[1] < 64:
                return img_np
            return face_crop
        except:
            return img_np

    def sample_frames(self, video_path: str, n_frames: int) -> List[np.ndarray]:
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return frames
            if self.is_train:
                indices = sorted(random.sample(range(total_frames), min(n_frames, total_frames)))
            else:
                indices = np.linspace(0, total_frames - 1, num=n_frames, dtype=int)
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        path = Path(path)
        ext = path.suffix.lower()
        try:
            if ext in VIDEO_EXTS:
                frames = self.sample_frames(path, self.config.get('n_frames_per_video', 8))
                if not frames:
                    img = np.zeros((self.config['img_size'], self.config['img_size'], 3), dtype=np.uint8)
                else:
                    img = random.choice(frames) if self.is_train else frames[len(frames) // 2]
                    img = self.crop_face(img, self.config.get('face_margin', 0.3))
            else:
                img = np.array(Image.open(path).convert('RGB'))
                img = self.crop_face(img, self.config.get('face_margin', 0.3))
            if self.transform:
                img = self.transform(img)
            return img, label
        except:
            img = np.zeros((self.config['img_size'], self.config['img_size'], 3), dtype=np.uint8)
            if self.transform:
                img = self.transform(img)
            return img, label


def load_celeb_df(base_path: str) -> List[Tuple[str, int]]:
    samples = []
    base = Path(base_path)
    for folder in ['Celeb-real', 'YouTube-real']:
        folder_path = base / folder
        if folder_path.exists():
            for video in folder_path.glob('*.mp4'):
                samples.append((str(video), 0))
    folder_path = base / 'Celeb-synthesis'
    if folder_path.exists():
        for video in folder_path.glob('*.mp4'):
            samples.append((str(video), 1))
    return samples


def load_faceforensics(base_path: str) -> List[Tuple[str, int]]:
    samples = []
    base = Path(base_path)
    real_path = base / 'original'
    if real_path.exists():
        for video in real_path.glob('*.mp4'):
            samples.append((str(video), 0))
    for folder in ['Deepfakes', 'FaceSwap', 'FaceShifter', 'Face2Face', 'NeuralTextures']:
        folder_path = base / folder
        if folder_path.exists():
            for video in folder_path.glob('*.mp4'):
                samples.append((str(video), 1))
    return samples
