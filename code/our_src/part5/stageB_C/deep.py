"""Frozen deep encoders: torchvision CNNs, DINOv2 and CLIP.

``src.deep_models.getModel`` provides CLIP. It has no ResNet-50 or DINOv2 ViT-B/14, which are
loaded here. Images are resized whole to a square (no center crop, so the object is not cut);
CLIP keeps its own ``preprocess``. All outputs are L2-normalised float32 arrays.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from .normalization import l2
from .timing import Timer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _whole_image_transform(input_size: int) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DeepEncoder:
    """Loads one model and encodes images or text.

    Each ``_load_*`` method sets ``self.model``, ``self.preprocess`` and ``self._forward``, which
    maps an image batch to one embedding per image.
    """

    def __init__(self, family: str, name: str, device: str, weights: str | None = None,
                 input_size: int | None = None) -> None:
        self.family, self.name, self.device = family, name, device
        self._weights, self._input_size = weights, input_size
        {"torchvision": self._load_torchvision, "dinov2": self._load_dinov2, "clip": self._load_clip}[family]()
        self.model.eval().requires_grad_(False)
        self.model.to(device)
        self.weights_bytes = sum(t.numel() * t.element_size() for t in self.model.state_dict().values())
        self._warm_up()

    def _load_torchvision(self) -> None:
        self.model = models.get_model(self.name, weights=self._weights)
        self.model.fc = torch.nn.Identity()  # keeps the global-average-pooled features
        self.preprocess = _whole_image_transform(self._input_size)
        self._forward = self.model

    def _load_dinov2(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is")
            self.model = torch.hub.load("facebookresearch/dinov2", self.name)
        self.preprocess = _whole_image_transform(self._input_size)
        self._forward = lambda batch: self.model.forward_features(batch)["x_norm_clstoken"]

    def _load_clip(self) -> None:
        from src.deep_models import getModel  # imported here because it needs the clip package

        self.model, self.preprocess = getModel(self.name, device=self.device)
        self._forward = self.model.encode_image

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    @torch.inference_mode()
    def _warm_up(self) -> None:
        """One dummy pass, so lazy GPU initialisation does not count towards the first query."""
        image = Image.new("RGB", (64, 64))
        self._forward(self.preprocess(image).unsqueeze(0).to(self.device))
        self._sync()

    @torch.inference_mode()
    def _encode_paths(self, paths: list[Path]) -> np.ndarray:
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(self.preprocess(image.convert("RGB")))
        output = self._forward(torch.stack(tensors).to(self.device))
        return output.float().cpu().numpy()

    def encode_corpus(self, paths: list[Path], batch_size: int) -> tuple[np.ndarray, float]:
        """Encode in batches. Returns the embeddings and the total seconds (decoding included)."""
        parts = []
        with Timer(self._sync) as timer:
            for start in range(0, len(paths), batch_size):
                parts.append(self._encode_paths(paths[start:start + batch_size]))
            features = l2(np.vstack(parts).astype(np.float32))
        return features, timer.elapsed

    def encode_queries(self, paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        """Encode one image at a time. Returns the embeddings and the seconds per query."""
        features, seconds = [], []
        for path in paths:
            with Timer(self._sync) as timer:
                features.append(l2(self._encode_paths([path])[0].astype(np.float32)))
            seconds.append(timer.elapsed)
        return np.stack(features), np.asarray(seconds)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """CLIP text encoder, one prompt at a time. Returns the embeddings and the seconds per query."""
        import clip

        features, seconds = [], []
        for text in texts:
            with Timer(self._sync) as timer:
                tokens = clip.tokenize([text]).to(self.device)
                output = self.model.encode_text(tokens).float().cpu().numpy()[0]
                features.append(l2(output.astype(np.float32)))
            seconds.append(timer.elapsed)
        return np.stack(features), np.asarray(seconds)


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def library_versions() -> dict[str, Any]:
    import torchvision

    return {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
