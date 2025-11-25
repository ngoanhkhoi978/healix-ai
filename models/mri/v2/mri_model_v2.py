import torch
import urllib.parse
import requests
import io
import gzip
import tempfile
import os
import nibabel as nib
from pathlib import Path
from typing import Optional, Union, List
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.mri.TransformerUNet import TransformerUNet

DEFAULT_WEIGHT = "models/mri/v2/model.pth"


class SegmentorModelV2:
    """MRI Brain Tumor Segmentation Model using TransformerUNet with 4 input channels.

    Accepts 4 MRI modality files (.gz or numpy arrays) for t2w, t2f, t1n, t1c.
    Returns segmentation mask and overlay image.
    """

    def __init__(self,
                 weights: str = DEFAULT_WEIGHT,
                 cache_dir: str = "weights_cache"):
        self.weights_source = weights
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_weights = self._ensure_local(weights)

        # Model expects 4 input channels (t2w, t2f, t1n, t1c)
        channels = (4, 32, 64, 128, 256, 512)
        self.model = TransformerUNet(
            channels=channels,
            num_heads=4,
            is_residual=True,
            bias=False
        ).to(self._get_device())

        # Load checkpoint with flexible loading strategy
        state = torch.load(str(self.local_weights), map_location=self._get_device())

        # Handle wrapped checkpoints
        if isinstance(state, dict) and ("state_dict" in state or "model_state_dict" in state):
            state_dict = state.get("state_dict", state.get("model_state_dict"))
        else:
            state_dict = state

        # Normalize keys by stripping common prefixes
        if isinstance(state_dict, dict):
            def strip_prefix(k, prefixes=("module.", "model.", "state_dict.")):
                for p in prefixes:
                    if k.startswith(p):
                        return k[len(p):]
                return k

            normalized = {strip_prefix(k): v for k, v in state_dict.items()}

            # Filter weights to match model parameters
            model_state = self.model.state_dict()
            filtered = {}
            for k, v in normalized.items():
                if k in model_state and hasattr(v, 'shape') and hasattr(model_state[k], 'shape'):
                    if v.shape == model_state[k].shape:
                        filtered[k] = v

            # Load filtered state dict
            load_result = self.model.load_state_dict(filtered, strict=False)
            missing = getattr(load_result, 'missing_keys', [])
            unexpected = getattr(load_result, 'unexpected_keys', [])
            print(f"MRI v2 model checkpoint: total keys={len(state_dict)}, used keys={len(filtered)}")
            if missing:
                print(f"Warning: Missing keys in checkpoint: {len(missing)} keys")
            if unexpected:
                print(f"Warning: Unexpected keys in checkpoint: {len(unexpected)} keys")
        else:
            load_result = self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

        # Transform for preprocessing
        self.transform = A.Compose([
            A.Resize(224, 224),
        ])

    def _get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _is_url(self, s: str) -> bool:
        p = urllib.parse.urlparse(s)
        return p.scheme in ("http", "https")

    def _ensure_local(self, weights: str) -> Path:
        """Download weights if URL, otherwise validate local path."""
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
        """Load PIL Image from path, URL, or PIL Image."""
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

        raise ValueError("Unsupported image_source type. Provide a local path, URL, or PIL.Image.")

    def _pad_to_square(self, image, pad_value=0):
        """Pad image to square with additional 3 pixels on each side like original API."""
        h, w = image.shape

        max_dim = max(h, w)

        pad_h = max_dim - h
        pad_w = max_dim - w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded_image = np.pad(image,
                              pad_width=((pad_top + 3, pad_bottom + 3), (pad_left + 3, pad_right + 3)),
                              mode='constant',
                              constant_values=pad_value)

        return padded_image

    def _read_nifti_data(self, file_bytes: bytes):
        """Helper function to save bytes to temp file, read with nibabel, and delete"""
        # Tạo một file tạm với đuôi .nii.gz
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # Load từ đường dẫn file tạm
            nii = nib.load(tmp_path)
            # Lấy data (đã được load vào RAM)
            data = nii.get_fdata()
            return data
        finally:
            # Xóa file tạm sau khi đọc xong để giải phóng ổ cứng
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _process_nifti_files(self, t2w_bytes: bytes, t2f_bytes: bytes, t1n_bytes: bytes, t1c_bytes: bytes):
        """Process 4 NIfTI files and return all slices like original API"""

        t2w_data = self._read_nifti_data(t2w_bytes)
        t2f_data = self._read_nifti_data(t2f_bytes)
        t1n_data = self._read_nifti_data(t1n_bytes)
        t1c_data = self._read_nifti_data(t1c_bytes)

        num_slices = t2w_data.shape[2]
        processed_slices = []

        for i in range(num_slices):
            slice_channels = []

            slice_2d_t2w = t2w_data[:, :, i]
            cropped_slice_t2w = self._pad_to_square(slice_2d_t2w)
            cropped_slice_t2w = cropped_slice_t2w.astype(np.float32)
            slice_channels.append(cropped_slice_t2w)

            slice_2d_t2f = t2f_data[:, :, i]
            cropped_slice_t2f = self._pad_to_square(slice_2d_t2f)
            cropped_slice_t2f = cropped_slice_t2f.astype(np.float32)
            slice_channels.append(cropped_slice_t2f)

            slice_2d_t1n = t1n_data[:, :, i]
            cropped_slice_t1n = self._pad_to_square(slice_2d_t1n)
            cropped_slice_t1n = cropped_slice_t1n.astype(np.float32)
            slice_channels.append(cropped_slice_t1n)

            slice_2d_t1c = t1c_data[:, :, i]
            cropped_slice_t1c = self._pad_to_square(slice_2d_t1c)
            cropped_slice_t1c = cropped_slice_t1c.astype(np.float32)
            slice_channels.append(cropped_slice_t1c)

            stacked_image = np.stack(slice_channels, axis=0).astype(np.float32)
            processed_slices.append(stacked_image)

        return processed_slices

    def _predict_slice(self, slice_data, threshold=0.5):
        """Predict mask for a single slice - exactly like original API"""

        # Transpose to (H, W, C) for albumentations
        image_vis = np.transpose(slice_data, (1, 2, 0))

        # Normalize each channel
        for i in range(image_vis.shape[-1]):
            channel = image_vis[:, :, i]
            if channel.max() > channel.min():
                image_vis[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())

        transform = A.Compose([A.Resize(224, 224)])
        transformed = transform(image=image_vis)
        image_vis = transformed['image']

        image_tensor = np.transpose(image_vis, (2, 0, 1))
        image_tensor = torch.from_numpy(image_tensor).float().unsqueeze(0).to(self._get_device())

        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            preds = torch.sigmoid(outputs)
            preds = (preds > threshold).float()
            mask = preds.squeeze(0).squeeze(0).cpu().numpy()

        return mask

    def segment_from_files_all_slices(self, t2w_bytes: bytes, t2f_bytes: bytes, t1n_bytes: bytes, t1c_bytes: bytes,
                                     threshold: float = 0.5) -> List[dict]:
        """Process all slices from 4 MRI modality files - exactly like original API.

        Args:
            t2w_bytes: T2-weighted MRI file bytes
            t2f_bytes: T2-FLAIR MRI file bytes
            t1n_bytes: T1-native MRI file bytes
            t1c_bytes: T1-contrast MRI file bytes
            threshold: Threshold for binary mask

        Returns:
            List of dict containing slice results with images and statistics
        """
        print("🔄 Processing NIfTI files...")
        processed_slices = self._process_nifti_files(t2w_bytes, t2f_bytes, t1n_bytes, t1c_bytes)

        print(f"📊 Total slices: {len(processed_slices)}")
        print("🧠 Predicting masks...")

        results = []
        for idx, slice_data in enumerate(processed_slices):
            # slice_data shape: (4, H, W)
            # Get individual channels
            t2w_slice = slice_data[0, :, :]
            t2f_slice = slice_data[1, :, :]
            t1n_slice = slice_data[2, :, :]
            t1c_slice = slice_data[3, :, :]

            # Predict mask
            predicted_mask = self._predict_slice(slice_data, threshold=threshold)

            # Store as dict with numpy arrays (will be converted to base64 in API layer)
            slice_result = {
                "slice_index": idx,
                "images": {
                    "t2w": t2w_slice,
                    "t2f": t2f_slice,
                    "t1n": t1n_slice,
                    "t1c": t1c_slice
                },
                "predicted_mask": predicted_mask,
                "statistics": {
                    "tumor_pixels": int(predicted_mask.sum()),
                    "total_pixels": int(predicted_mask.size),
                    "tumor_percentage": float(predicted_mask.sum() / predicted_mask.size * 100)
                }
            }

            results.append(slice_result)

        print(f"✅ Completed {len(results)} slices")
        return results

    def _load_nifti_slice(self, file_bytes: bytes, slice_idx: Optional[int] = None) -> np.ndarray:
        """Load a slice from a NIfTI .gz file."""
        try:
            # Decompress gzip
            decompressed = gzip.decompress(file_bytes)
            # Load NIfTI
            nii = nib.Nifti1Image.from_bytes(decompressed)
            data = nii.get_fdata()

            # Get middle slice if not specified
            if slice_idx is None:
                slice_idx = data.shape[2] // 2

            # Extract slice (assuming axial view)
            slice_data = data[:, :, slice_idx]

            # Normalize to 0-1
            if slice_data.max() > slice_data.min():
                slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())

            return slice_data.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load NIfTI file: {e}")

    def _rgb_to_4channel(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB image to 4-channel input for the model.

        Strategy:
        - Channel 0 (t2w): Grayscale conversion
        - Channel 1 (t2f): Red channel
        - Channel 2 (t1n): Green channel
        - Channel 3 (t1c): Blue channel
        """
        if rgb_image.ndim == 2:
            # Already grayscale
            gray = rgb_image
            r = rgb_image
            g = rgb_image
            b = rgb_image
        elif rgb_image.shape[2] == 3:
            # Convert to grayscale for first channel
            gray = 0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]
            r = rgb_image[:, :, 0]
            g = rgb_image[:, :, 1]
            b = rgb_image[:, :, 2]
        else:
            raise ValueError(f"Unexpected image shape: {rgb_image.shape}")

        # Stack into 4 channels
        four_channel = np.stack([gray, r, g, b], axis=-1)
        return four_channel

    def segment_from_files(self, t2w_bytes: bytes, t2f_bytes: bytes, t1n_bytes: bytes, t1c_bytes: bytes,
                          slice_idx: Optional[int] = None, threshold: float = 0.5) -> tuple:
        """Segment from 4 MRI modality files (.gz format).

        Args:
            t2w_bytes: T2-weighted MRI file bytes
            t2f_bytes: T2-FLAIR MRI file bytes
            t1n_bytes: T1-native MRI file bytes
            t1c_bytes: T1-contrast MRI file bytes
            slice_idx: Slice index to extract (default: middle slice)
            threshold: Threshold for binary mask

        Returns:
            tuple: (predicted_mask, overlayed_image, original_image)
        """
        # Load all 4 modalities
        t2w = self._load_nifti_slice(t2w_bytes, slice_idx)
        t2f = self._load_nifti_slice(t2f_bytes, slice_idx)
        t1n = self._load_nifti_slice(t1n_bytes, slice_idx)
        t1c = self._load_nifti_slice(t1c_bytes, slice_idx)

        # Stack into 4 channels (H, W, 4)
        four_channel = np.stack([t2w, t2f, t1n, t1c], axis=-1)
        original_shape = four_channel.shape[:2]

        # Apply transform (resize)
        transformed = self.transform(image=four_channel)
        image_resized = transformed['image']

        # Normalize each channel independently
        for i in range(4):
            channel = image_resized[:, :, i]
            if channel.max() > channel.min():
                image_resized[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())

        # Convert to tensor: (H, W, C) -> (C, H, W)
        image_tensor = np.transpose(image_resized, (2, 0, 1))
        image_tensor = torch.from_numpy(image_tensor).float().unsqueeze(0).to(self._get_device())

        # Predict
        with torch.no_grad():
            pred_logits = self.model(image_tensor)
            pred_prob = torch.sigmoid(pred_logits).cpu().squeeze().numpy()
            pred_mask = (pred_prob > threshold).astype(np.uint8)

        # Clean mask using largest connected component
        try:
            import cv2
            if pred_mask.ndim > 2:
                pred_mask = pred_mask.squeeze()
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_label = 1 + int(areas.argmax())
                cleaned = (labels == max_label).astype(np.uint8)
            else:
                cleaned = pred_mask
        except Exception:
            cleaned = pred_mask

        # Resize mask back to original size
        pred_mask_resized = Image.fromarray((cleaned * 255).astype(np.uint8)).resize(
            (original_shape[1], original_shape[0]), Image.NEAREST
        )
        pred_mask_np = np.array(pred_mask_resized) / 255.0

        # Create visualization from T2-weighted (most common for display)
        original_vis = (t2w * 255).astype(np.uint8)
        original_rgb = np.stack([original_vis, original_vis, original_vis], axis=-1)

        # Create overlay
        overlay = original_rgb.astype(np.float32)
        alpha = 0.5
        red = np.array([255, 0, 0], dtype=np.float32)
        mask2D = pred_mask_np > 0
        overlay[mask2D] = overlay[mask2D] * (1.0 - alpha) + red * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        overlayed_image = Image.fromarray(overlay)
        original_image = Image.fromarray(original_rgb)

        return cleaned, overlayed_image, original_image

    def segment_image(self, image_source: Union[str, Path, Image.Image], threshold: float = 0.5,
                     save_path: Optional[str] = None, save_mask_path: Optional[str] = None):
        """Run segmentation on an image and return (mask, overlay_image).

        Args:
            image_source: Path, URL, or PIL Image
            threshold: Threshold for binary mask (default 0.5)
            save_path: Optional path to save overlay image
            save_mask_path: Optional path to save mask

        Returns:
            tuple: (predicted_mask as numpy array, overlayed PIL Image)
        """
        original_image = self._load_image(image_source)
        original_size = original_image.size  # (width, height)

        # Convert to numpy and prepare 4-channel input
        image_np = np.array(original_image).astype(np.float32)

        # Normalize to 0-1 range
        if image_np.max() > 1.0:
            image_np = image_np / 255.0

        # Convert RGB to 4 channels
        four_channel = self._rgb_to_4channel(image_np)

        # Apply transform (resize)
        transformed = self.transform(image=four_channel)
        image_resized = transformed['image']

        # Normalize each channel independently
        for i in range(4):
            channel = image_resized[:, :, i]
            if channel.max() > channel.min():
                image_resized[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())

        # Convert to tensor: (H, W, C) -> (C, H, W)
        image_tensor = np.transpose(image_resized, (2, 0, 1))
        image_tensor = torch.from_numpy(image_tensor).float().unsqueeze(0).to(self._get_device())

        # Predict
        with torch.no_grad():
            pred_logits = self.model(image_tensor)
            pred_prob = torch.sigmoid(pred_logits).cpu().squeeze().numpy()
            pred_mask = (pred_prob > threshold).astype(np.uint8)

        # Clean mask using largest connected component
        try:
            import cv2
            if pred_mask.ndim > 2:
                pred_mask = pred_mask.squeeze()
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_label = 1 + int(areas.argmax())
                cleaned = (labels == max_label).astype(np.uint8)
            else:
                cleaned = pred_mask
        except Exception:
            cleaned = pred_mask

        if save_mask_path:
            mask_img = Image.fromarray((cleaned * 255).astype(np.uint8))
            mask_img.save(save_mask_path)

        # Resize mask back to original size
        pred_mask_resized = Image.fromarray((cleaned * 255).astype(np.uint8)).resize(original_size, Image.NEAREST)
        pred_mask_np = np.array(pred_mask_resized) / 255.0

        # Create overlay with red color for segmented region
        original_uint = np.array(original_image).astype(np.uint8)
        overlay = original_uint.copy().astype(np.float32)

        alpha = 0.5
        red = np.array([255, 0, 0], dtype=np.float32)

        # Create 2D boolean mask
        mask2D = pred_mask_np > 0

        # Blend where mask is True
        overlay[mask2D] = overlay[mask2D] * (1.0 - alpha) + red * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        overlayed_image = Image.fromarray(overlay)

        if save_path:
            overlayed_image.save(save_path)

        return cleaned, overlayed_image


if __name__ == "__main__":
    # Example usage
    m = SegmentorModelV2(weights=DEFAULT_WEIGHT)

    test_image = "path/to/mri_image.jpg"

    pred_mask, overlayed = m.segment_image(test_image, threshold=0.5, save_path="segmented_overlay_v2.jpg")
    print("Predicted mask shape:", pred_mask.shape)
    print("Overlay saved to segmented_overlay_v2.jpg")
