"""Vision Module — CLIP encoder + projection + JEPA contrastive loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPVisionEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 image encoder. Lazy-loaded."""
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self._model_name = model_name
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained(self._model_name)
            self._processor = CLIPProcessor.from_pretrained(self._model_name)
            for p in self._model.parameters():
                p.requires_grad = False
            self._model.eval()
        except ImportError:
            self._model = None

    def encode_image(self, image) -> torch.Tensor:
        self._load()
        if self._model is None:
            return torch.randn(1, 512)
        if hasattr(image, 'mode'):
            inputs = self._processor(images=image, return_tensors="pt")
            with torch.no_grad():
                features = self._model.get_image_features(**inputs)
        else:
            with torch.no_grad():
                features = self._model.get_image_features(pixel_values=image)
        return F.normalize(features, dim=-1)

    def encode_text(self, text: str) -> torch.Tensor:
        self._load()
        if self._model is None:
            return torch.randn(1, 512)
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        return F.normalize(features, dim=-1)


class VisionProjector(nn.Module):
    """Projects CLIP (512) → n_visual_tokens × d_model pseudo-tokens."""
    def __init__(self, clip_dim=512, d_model=1024, n_visual_tokens=4):
        super().__init__()
        self.n_visual_tokens = n_visual_tokens
        self.d_model = d_model
        self.projector = nn.Sequential(
            nn.Linear(clip_dim, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model * n_visual_tokens),
        )
        self.visual_pos = nn.Parameter(torch.randn(1, n_visual_tokens, d_model) * 0.02)

    def forward(self, clip_features):
        projected = self.projector(clip_features)
        visual_tokens = projected.view(-1, self.n_visual_tokens, self.d_model)
        return visual_tokens + self.visual_pos


class JEPAContrastiveLoss(nn.Module):
    """Contrastive loss for image-text alignment (JEPA-inspired)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeds, text_embeds):
        logits = torch.matmul(image_embeds, text_embeds.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2
