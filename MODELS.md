# Healix AI - Các Mô Hình Sử Dụng (Models Documentation)

## Tổng Quan (Overview)

Healix AI sử dụng hai mô hình deep learning chính để phân tích hình ảnh y tế:

1. **RFDETR (Real-time DEtection TRansformer)** - Cho phát hiện bệnh trên ảnh X-quang
2. **TransformerUNet** - Cho phân đoạn ảnh MRI

---

## 1. X-ray Detection Model (RFDETR)

### Mô Tả
Mô hình phát hiện đối tượng dựa trên DETR (Detection Transformer) được tối ưu hóa cho phát hiện bệnh lý trên ảnh X-quang phổi.

### Kiến Trúc
- **Model Type**: RFDETR Medium
- **Framework**: PyTorch
- **Library**: `rfdetr` package
- **Input**: Ảnh X-quang RGB (được chuyển đổi từ grayscale)
- **Output**: Bounding boxes và class predictions cho các bệnh lý phát hiện được

### Tính Năng
- Phát hiện real-time các bệnh lý trên ảnh X-quang phổi
- Hỗ trợ 11 lớp bệnh phổi thường gặp
- Confidence threshold có thể điều chỉnh (mặc định: 0.3)
- Visualization với bounding boxes và labels có màu sắc

### Classes Detected
Model hỗ trợ phát hiện 11 loại bệnh lý phổi khác nhau (được định nghĩa trong `coco_annotations_val.json`):

**Danh sách bệnh được phát hiện:**
1. **Aortic enlargement** (Phình động mạch chủ)
2. **Atelectasis** (Xẹp phổi)
3. **Cardiomegaly** (Tim to)
4. **Consolidation** (Đông đặc phổi)
5. **ILD** (Interstitial Lung Disease - Bệnh phổi kẽ)
6. **Infiltration** (Thâm nhiễm)
7. **Lung Opacity** (Mờ đục phổi)
8. **Other lesion** (Tổn thương khác)
9. **Pleural effusion** (Tràn dịch màng phổi)
10. **Pneumothorax** (Tràn khí màng phổi)
11. **Pulmonary fibrosis** (Xơ phổi)

**Note về classes:** 
- Model architecture hỗ trợ tối đa 14 classes (có fallback DEFAULT_CLASSES trong code)
- COCO annotations dataset hiện tại chỉ định nghĩa 11 bệnh cụ thể
- Model được train với 11 bệnh này từ VinBigData Chest X-ray dataset
- Khi load từ COCO file, model sẽ sử dụng 11 classes thực tế

### File Weights
- **Default Path**: `models/xray/model.pth`
- **Format**: PyTorch checkpoint (.pth)
- Hỗ trợ load từ local file hoặc URL

### Usage
```python
from models.xray.xray_model import DetectorModel

# Khởi tạo model
model = DetectorModel(weights="models/xray/model.pth")

# Dự đoán
detections, annotated_image = model.diagnose_image(
    image_source="path/to/xray.jpg",
    threshold=0.3
)
```

### API Endpoints
- `POST /xray/lung/predict` - Trả về ảnh đã annotate
- `POST /xray/lung/predict_with_json` - Trả về ảnh + JSON detections

---

## 2. MRI Segmentation Model (TransformerUNet)

### Mô Tả
Mô hình phân đoạn ngữ nghĩa (semantic segmentation) kết hợp kiến trúc UNet với Transformer attention mechanisms để phân đoạn các vùng bệnh lý trên ảnh MRI.

### Kiến Trúc

#### TransformerUNet Components

**1. Encoder-Decoder Architecture**
```python
channels = (3, 32, 64, 128, 256, 512)
```

**2. Core Components:**

- **Encoder Layers**: 
  - ConvBlock với residual connections
  - MaxPooling để giảm spatial dimensions
  - Skip connections để giữ thông tin chi tiết

- **Bottleneck**:
  - ConvBlock ở tầng sâu nhất (512 channels)
  - Multi-Head Self-Attention (MHSA)
  - Positional Encoding

- **Decoder Layers**:
  - Multi-Head Cross-Attention (MHCA) 
  - ConvTranspose2d để upsampling
  - Skip connections từ encoder

**3. Attention Mechanisms:**

- **Multi-Head Self-Attention (MHSA)**:
  - Number of heads: 4
  - Áp dụng ở bottleneck
  - Capture global context

- **Multi-Head Cross-Attention (MHCA)**:
  - Cross-attention giữa encoder skip connections và decoder features
  - Weighted feature fusion
  - Gated attention với sigmoid

**4. Positional Encoding**:
  - Sinusoidal position encoding
  - Thêm thông tin vị trí cho attention layers

### Architecture Details

```
Input (3 channels RGB)
    ↓
Encoder Path (with skip connections):
    Conv(3→32) → MaxPool
    Conv(32→64) → MaxPool  
    Conv(64→128) → MaxPool
    Conv(128→256) → MaxPool
    ↓
Bottleneck:
    Conv(256→512)
    + Positional Encoding
    + Multi-Head Self-Attention (4 heads)
    ↓
Decoder Path:
    Cross-Attention + ConvTranspose(512→256)
    Cross-Attention + ConvTranspose(256→128)
    Cross-Attention + ConvTranspose(128→64)
    Cross-Attention + ConvTranspose(64→32)
    ↓
Output:
    Conv(32→1) - Binary segmentation mask
```

### Tính Năng
- **Residual Connections**: Cải thiện gradient flow
- **Batch Normalization**: Ổn định training
- **Kaiming Initialization**: Khởi tạo weights tối ưu
- **Connected Components Analysis**: Loại bỏ noise, giữ vùng phân đoạn lớn nhất
- **Flexible Checkpoint Loading**: Hỗ trợ nhiều định dạng checkpoint

### Input/Output
- **Input Size**: 224x224 RGB (được resize tự động)
- **Normalization**: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
- **Output**: Binary segmentation mask
- **Threshold**: 0.5 (có thể điều chỉnh)

### File Weights
- **Default Path**: `models/mri/model.pth`
- **Format**: PyTorch checkpoint (.pth)
- Hỗ trợ load từ local file hoặc URL
- Flexible checkpoint format (hỗ trợ wrapped checkpoints)

### Usage
```python
from models.mri.mri_model import SegmentorModel

# Khởi tạo model
model = SegmentorModel(weights="models/mri/model.pth")

# Phân đoạn
mask, overlayed_image = model.segment_image(
    image_source="path/to/mri.jpg",
    threshold=0.5
)
```

### API Endpoints
- `POST /mri/predict` - Trả về overlay image
- `POST /mri/predict_with_json` - Trả về overlay + mask metadata
- `POST /mri/debug` - Debug endpoint với raw mask

---

## Technical Stack

### Frameworks & Libraries

**Deep Learning:**
- PyTorch
- rfdetr (DETR implementation)

**Image Processing:**
- PIL/Pillow
- OpenCV (cv2)
- Albumentations
- Supervision (sv)

**API:**
- FastAPI
- Uvicorn

### Device Support
Cả hai model đều hỗ trợ:
- CUDA (GPU acceleration)
- CPU (fallback)

Tự động detect và sử dụng GPU nếu có sẵn:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## Model Loading

### Environment Variables
```bash
MODEL_XRAY_WEIGHTS="models/xray/model.pth"
MODEL_MRI_WEIGHTS="models/mri/model.pth"
```

### Flexible Weight Loading
Cả hai model đều hỗ trợ:
1. **Local file paths**
2. **HTTP/HTTPS URLs** (tự động download và cache)
3. **Automatic caching** trong `weights_cache/` directory

### Example với URLs:
```python
# X-ray model từ URL
xray_model = DetectorModel(
    weights="https://example.com/xray_weights.pth"
)

# MRI model từ URL
mri_model = SegmentorModel(
    weights="https://example.com/mri_weights.pth"
)
```

---

## Model Files Structure

```
models/
├── xray/
│   ├── xray_model.py          # RFDETR wrapper
│   ├── model.pth              # Pretrained weights
│   └── coco_annotations_val.json  # Class definitions
└── mri/
    ├── mri_model.py           # SegmentorModel wrapper
    ├── TransformerUNet.py     # TransformerUNet architecture
    ├── EncoderDecoder.py      # Encoder/Decoder components
    └── model.pth              # Pretrained weights
```

---

## Health Check

API cung cấp health check endpoint để kiểm tra trạng thái models:

```bash
GET /health
```

Response:
```json
{
  "ready": true,
  "models": {
    "xray": true,
    "mri": true
  },
  "errors": {}
}
```

---

## Performance Considerations

### X-ray Model (RFDETR)
- Fast inference (~real-time)
- Lightweight model architecture
- Optimized for edge deployment

### MRI Model (TransformerUNet)
- Medium inference time
- Attention mechanisms add computational cost
- GPU recommended for faster inference
- Input resizing to 224x224 for consistent performance

---

## References

1. **DETR (DEtection TRansformer)**:
   - Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-End Object Detection with Transformers. ECCV 2020.
   - Paper: https://arxiv.org/abs/2005.12872
   - Eliminates need for hand-designed components like NMS

2. **UNet**:
   - Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
   - Paper: https://arxiv.org/abs/1505.04597
   - U-shaped architecture with skip connections

3. **Transformer Attention**:
   - Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.
   - Paper: https://arxiv.org/abs/1706.03762
   - Multi-head attention mechanisms and positional encoding

4. **Medical Imaging Datasets**:
   - VinBigData Chest X-ray Abnormalities Detection: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection
   - BraTS (Brain Tumor Segmentation): http://braintumorsegmentation.org/
