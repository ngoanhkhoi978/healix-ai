from pathlib import Path
import urllib.parse
import requests
import io
import json
from typing import Optional, List, Union

from rfdetr import RFDETRMedium
import supervision as sv
from PIL import Image

DEFAULT_WEIGHT = "models/xray/model.pth"
DEFAULT_COCO = "coco_annotations_val.json"

DEFAULT_CLASSES = [
    "class1", "class2", "class3", "class4", "class5", "class6",
    "class7", "class8", "class9", "class10", "class11", "class12",
    "class13", "class14"
]


class DetectorModel:
    """Wrapper around RFDETR that accepts a local file path or an HTTP/HTTPS URL for weights.

    Methods
    - diagnose_image(image_source, threshold=0.3, save_path=None)
      accepts: local path, URL, or PIL.Image and returns (detections, annotated PIL.Image)
    """

    def __init__(self,
                 weights: str = DEFAULT_WEIGHT,
                 classes: Optional[List[str]] = None,
                 coco_annotations: Optional[str] = None,
                 cache_dir: str = "weights_cache"):
        self.weights_source = weights
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_weights = self._ensure_local(weights)
        self.model = RFDETRMedium(pretrain_weights=str(self.local_weights))

        # load classes: explicit > coco file > default
        if classes is not None:
            self.classes = classes
        else:
            coco_path = coco_annotations or DEFAULT_COCO
            if Path(coco_path).exists():
                try:
                    self.classes = self.load_classes_from_coco(coco_path)
                except Exception:
                    self.classes = DEFAULT_CLASSES
            else:
                self.classes = DEFAULT_CLASSES

    def _is_url(self, s: str) -> bool:
        p = urllib.parse.urlparse(s)
        return p.scheme in ("http", "https")

    def _ensure_local(self, weights: str) -> Path:
        """Return a local Path to weights. If weights is a URL, download and cache it.
        If weights is a local path, validate it exists.
        """
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
                alt = Path("..") / p
                if alt.exists():
                    p = alt
                else:
                    raise FileNotFoundError(f"Weights not found at: {weights}")
            return p

    @staticmethod
    def load_classes_from_coco(annotations_path: str) -> List[str]:
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [c["name"] for c in data.get("categories", [])]

    def _load_image(self, image_source: Union[str, Path, Image.Image]) -> Image.Image:
        """Load a PIL.Image from a path, URL, or return the passed PIL.Image."""
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

    def diagnose_image(self, image_source: Union[str, Path, Image.Image], threshold: float = 0.3, save_path: Optional[str] = None):
        """Run prediction on an image source and return (detections, annotated_image).

        If save_path is provided the annotated image will be saved to that path.
        """
        image = self._load_image(image_source)
        detections = self.model.predict(image, threshold=threshold)

        text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
        color = sv.ColorPalette.from_hex([
            "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
            "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
            "#ff0000", "#008000"
        ])

        bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
        label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK, text_scale=text_scale)

        detections_labels = []
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            try:
                label = self.classes[int(class_id)]
            except Exception:
                label = str(class_id)
            detections_labels.append(f"{label} {float(confidence):.2f}")

        annotated = image.copy()
        annotated = bbox_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, detections_labels)

        # supervision annotators may return a numpy.ndarray — convert to PIL.Image if needed
        if not isinstance(annotated, Image.Image):
            try:
                from PIL import Image as PILImage
                annotated = PILImage.fromarray(annotated)
            except Exception:
                # fallback: return as-is
                pass

        if save_path and isinstance(annotated, Image.Image):
            annotated.save(save_path)

        return detections, annotated


if __name__ == "__main__":
    # example: use local weights or pass a URL to DetectorModel(weights=...)
    m = DetectorModel(weights=DEFAULT_WEIGHT)

    # test with a local image or URL
    test_image = "https://storage.googleapis.com/kagglesdsdata/datasets/1069682/1799839/vinbigdata/test/002a34c58c5b758217ed1f584ccbcfe9.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251109%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251109T035056Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=58f92ac84ec3e4531dde0d277d8ed48e7757cf6255b8cb47b9241ac613e579222304d468a47436ef405adc1e5871653b50bd9cded36f4ace8c0d9ccc676d374a072e827c6440f76d25ca2d9015f7b004d1239dcabbffbbe322ef99cad87940162311e538c7a51e777b09b08d483aaffd8f6185c4ac323447adc6e97b94e83d0b574ce008458652dae847872ee7ddb1ce09940a86e7b02a3fefd8477b69d2ec68b0acbdca3c817dd3350f1a06e6982b7b5e8a56899fd5744fbf116c8ab62091960c5c5e132143d04a95e2366bf7b7eca495e4ba9437de0c797f44723e035ae060b1c9b545c6675135a4fd181b4215a63226d3497f26447cb50f70eed6d5e102ef"
    # Or replace with an accessible URL
    # test_image = "https://example.com/image.png"

    det, ann = m.diagnose_image(test_image, threshold=0.3, save_path="../../diagnosed_image.jpg")
    print("Detections:", det)
    print("Annotated saved to diagnosed_image.jpg")
