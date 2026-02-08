# Deepfake Detection - HAI Hecto AI Challenge 2025

**팀명: 바다의노인**
- 이하준 (한양대학교 대학원 컴퓨터 소프트웨어학과)
- 김진욱 (서울과학기술대학교 기계공학과)

---

## Technical Report

<a href="Vision_Transformer_기반_딥페이크_탐지_모델_개발__1_.pdf">
  <img src="https://img.shields.io/badge/Technical%20Report-PDF-red?style=for-the-badge&logo=adobe-acrobat-reader" alt="Technical Report PDF"/>
</a>

**[Vision Transformer 기반 딥페이크 탐지 모델 개발 (PDF)](Vision_Transformer_기반_딥페이크_탐지_모델_개발__1_.pdf)**

---

## Method Summary

### Model Architecture
- **Backbone**: ViT-Small (384×384)
- **Base Model**: CommFor (Community-Forensics) pretrained
- **Output**: Binary classification (Real vs Fake)

### Key Components
1. **Face Detection**: YOLOv8
2. **Preprocessing**: Face crop + normalize
3. **Frame Sampling**: Uniform sampling (N=10)
4. **Aggregation**: Mean pooling (post-processing)

### Training Data
- Celeb-DF v2
- FaceForensics++ C23

## Project Structure

```
your_submission/
├── model/model.pt           # Model weights
├── src/                     # Source modules
├── config/config.yaml       # Configuration
├── env/                     # Docker & requirements
├── train_data/              # Training data (not included in repo)
├── test_data/               # Test data (not included in repo)
├── train.py                 # Training script
├── inference.py             # Inference script
└── README.md
```

> **Note**: `train_data/` and `test_data/` are excluded from this repository due to size constraints. Please download the datasets separately.

## Usage

### Inference
```bash
python inference.py
```

### Training
```bash
python train.py
```

### Docker
```bash
docker build -t deepfake -f env/Dockerfile .
docker run --gpus all deepfake
```

## Competition Rules Compliance

- Single model (no ensemble)
- No TTA
- Frame-by-frame independent processing
- Post-processing aggregation only
- Offline inference capable

## License

This project was developed for the HAI Hecto AI Challenge 2025.
