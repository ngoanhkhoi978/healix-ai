from pathlib import Path
import urllib.parse
import requests
import io
import torch
from typing import Optional, Union

from models.mri.TransformerUNet import TransformerUNet
import albumentations
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

DEFAULT_WEIGHT = "models/mri/model.pth"

class SegmentorModel:
    def __init__(self,
                 weights: str = DEFAULT_WEIGHT,
                 cache_dir: str = "weights_cache"):
        self.weights_source = weights
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_weights = self._ensure_local(weights)

        channels = (3, 32, 64, 128, 256, 512)
        self.model = TransformerUNet(
            channels=channels,
            num_heads=4,
            is_residual=True,
            bias=False
        ).to(self._get_device())

        # Load checkpoint flexibly: support wrapped checkpoints and allow partial loads
        state = torch.load(str(self.local_weights), map_location=self._get_device())
        # common checkpoint wrappers may store the state dict under keys like 'state_dict' or 'model_state_dict'
        if isinstance(state, dict) and ("state_dict" in state or "model_state_dict" in state):
            state_dict = state.get("state_dict", state.get("model_state_dict"))
        else:
            state_dict = state

        # If state_dict itself contains a 'model' or similar nested dict, try to find the first dict-like candidate
        if isinstance(state_dict, dict):
            # sometimes checkpoints have prefixes like 'module.'; normalize by removing common prefixes
            def strip_prefix(k, prefixes=("module.", "model.", "state_dict.")):
                for p in prefixes:
                    if k.startswith(p):
                        return k[len(p):]
                return k

            normalized = {strip_prefix(k): v for k, v in state_dict.items()}

            # Filter weights to those matching model parameter names and shapes
            model_state = self.model.state_dict()
            filtered = {}
            for k, v in normalized.items():
                if k in model_state and getattr(v, 'shape', None) == getattr(model_state[k], 'shape', None):
                    filtered[k] = v

            # Load filtered state dict (non-matching params will be left randomly initialized)
            load_result = self.model.load_state_dict(filtered, strict=False)
            missing = getattr(load_result, 'missing_keys', [])
            unexpected = getattr(load_result, 'unexpected_keys', [])
            print(f"MRI model checkpoint: total keys in checkpoint={len(state_dict)}, used keys={len(filtered)}")
            if missing or unexpected:
                print("Warning: MRI model load_state_dict had missing/unexpected keys:", missing, unexpected)
        else:
            # fallback: try loading directly (will raise if incompatible)
            load_result = self.model.load_state_dict(state_dict, strict=False)
            missing = getattr(load_result, 'missing_keys', [])
            unexpected = getattr(load_result, 'unexpected_keys', [])
            if missing or unexpected:
                print("Warning: MRI model load_state_dict had missing/unexpected keys:", missing, unexpected)

        self.model.eval()

        self.transform = albumentations.Compose([
            albumentations.Resize(224, 224),
            albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def _get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _is_url(self, s: str) -> bool:
        p = urllib.parse.urlparse(s)
        return p.scheme in ("http", "https")

    def _ensure_local(self, weights: str) -> Path:
        if self._is_url(weights):
            parsed = urllib.parse.urlparse(weights)
            filename = Path(urllib.parse.unquote(parsed.path)).name or "weights.pth"
            local_path = self.cache_dir / filename
            if not local_path.exists():
                resp = requests.get(weights, stream=True, timeout=60)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return local_path
        else:
            p = Path(weights)
            if not p.exists():
                alt = Path("../..") / p
                if alt.exists():
                    p = alt
                else:
                    raise FileNotFoundError(f"Weights not found at: {weights}")
            return p

    def _load_image(self, image_source: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")

        if isinstance(image_source, Path) or (isinstance(image_source, str) and not self._is_url(image_source)):
            p = Path(image_source)
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {image_source}")
            return Image.open(p).convert("RGB")

        if isinstance(image_source, str) and self._is_url(image_source):
            resp = requests.get(image_source, stream=True, timeout=60)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")

        raise ValueError("Unsupported image_source type. Provide a local path, URL, or PIL.Image.Image.")

    def segment_image(self, image_source: Union[str, Path, Image.Image], threshold: float = 0.5, save_path: Optional[str] = None, save_mask_path: Optional[str] = None):
        original_image = self._load_image(image_source)
        original_size = original_image.size  # (width, height)

        # Transform image for model
        image_np = np.array(original_image)
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(self._get_device())

        # Predict
        with torch.no_grad():
            pred_logits = self.model(image_tensor)
            pred_prob = torch.sigmoid(pred_logits).cpu().squeeze().numpy()
            pred_mask = (pred_prob > threshold).astype(np.uint8)

        # Optional: clean mask (keep largest connected component) if cv2 is available
        try:
            import cv2
            # ensure mask is 2D
            if pred_mask.ndim > 2:
                # if model produced channel dim, take first
                pred_mask = pred_mask.squeeze()
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
            if num_labels > 1:
                # find largest component (ignore background label 0)
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_label = 1 + int(areas.argmax())
                cleaned = (labels == max_label).astype(np.uint8)
            else:
                cleaned = pred_mask
        except Exception:
            # cv2 not available - fallback to using mask as-is
            cleaned = pred_mask

        if save_mask_path:
            # save cleaned mask for debugging
            mask_img = Image.fromarray((cleaned * 255).astype(np.uint8))
            mask_img.save(save_mask_path)

        # Resize mask back to original image size
        pred_mask_resized = Image.fromarray((cleaned * 255).astype(np.uint8)).resize(original_size, Image.NEAREST)
        pred_mask_np = np.array(pred_mask_resized) / 255.0

        # Prepare overlay using proper uint8 blending to avoid color cast artifacts
        original_uint = (np.array(original_image)).astype(np.uint8)
        overlay = original_uint.copy()
        alpha = 0.5
        red = np.array([255, 0, 0], dtype=np.uint8)

        # Create 3-channel boolean mask
        mask2D = pred_mask_np > 0

        # Blend only where mask is True
        overlay = overlay.astype(np.float32)
        # overlay[mask2D] -> shape (num_true_pixels, 3)
        overlay[mask2D] = overlay[mask2D] * (1.0 - alpha) + red.astype(np.float32) * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        overlayed_image = Image.fromarray(overlay)

        if save_path:
            overlayed_image.save(save_path)

        return cleaned, overlayed_image


# if _name_ == "__main_":
#     m = SegmentorModel(weights=DEFAULT_WEIGHT)
#
#     test_image = "path/to/mri_image.jpg"
#
#     pred_mask, overlayed = m.segment_image(test_image, threshold=0.5, save_path="segmented_overlay.jpg")
#     print("Predicted mask shape:", pred_mask.shape)
#     print("Overlay saved to segmented_overlay.jpg")