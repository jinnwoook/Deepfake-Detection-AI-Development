"""
Deepfake Detection Model Definition
- ViT-based classifier using CommFor architecture
- Single backbone model (no ensemble)
"""

import torch
import torch.nn as nn
import timm
from huggingface_hub import PyTorchModelHubMixin


class ViTClassifier(nn.Module, PyTorchModelHubMixin):
    """
    Vision Transformer based Deepfake Classifier
    Based on Community-Forensics (CommFor) model architecture.
    """

    def __init__(
        self,
        model_size: str = "small",
        input_size: int = 384,
        patch_size: int = 16,
        freeze_backbone: bool = False,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        if model_size == "small":
            if input_size == 224:
                model_name = "vit_small_patch16_224.augreg_in21k_ft_in1k"
            else:
                if patch_size == 16:
                    model_name = "vit_small_patch16_384.augreg_in21k_ft_in1k"
                else:
                    model_name = "vit_small_patch32_384.augreg_in21k_ft_in1k"
        else:
            model_name = "vit_tiny_patch16_384.augreg_in21k_ft_in1k"

        self.vit = timm.create_model(model_name, pretrained=True)
        self.vit.head = nn.Linear(
            in_features=self.vit.head.in_features,
            out_features=1,
            bias=True
        )

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.vit.head.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


def load_model(model_path: str = None, device: str = 'cuda') -> ViTClassifier:
    """Load model with optional finetuned weights"""
    model = ViTClassifier.from_pretrained('OwensLab/commfor-model-384')

    if model_path is not None:
        import os
        if os.path.exists(model_path):
            print(f"[INFO] Loading weights from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')

            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "").replace("_orig_mod.", "")
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict, strict=False)
            print("[INFO] Weights loaded successfully!")

    model = model.to(device)
    model.eval()
    return model
