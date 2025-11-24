# Trả Lời: Các Mô Hình Sử Dụng trong Healix AI

## Tóm Tắt Nhanh

Healix AI sử dụng **2 mô hình AI chính**:

### 1. **RFDETR (Real-time DEtection TRansformer)** - Cho X-ray
- **Mục đích**: Phát hiện bệnh lý trên ảnh X-quang phổi
- **Loại model**: Object Detection (Phát hiện đối tượng)
- **Kiến trúc**: Detection Transformer (DETR)
- **Thư viện**: rfdetr package
- **Output**: Bounding boxes + labels cho 14 loại bệnh
- **File weights**: `models/xray/model.pth`

### 2. **TransformerUNet** - Cho MRI  
- **Mục đích**: Phân đoạn vùng bệnh lý trên ảnh MRI
- **Loại model**: Semantic Segmentation (Phân đoạn ngữ nghĩa)
- **Kiến trúc**: UNet + Transformer Attention
- **Components**:
  - Encoder-Decoder với skip connections
  - Multi-Head Self-Attention (4 heads)
  - Multi-Head Cross-Attention
  - Positional Encoding
  - Residual Connections
- **Channels**: (3, 32, 64, 128, 256, 512)
- **Output**: Binary segmentation mask
- **File weights**: `models/mri/model.pth`

## Chi Tiết Kỹ Thuật

### X-ray Model (RFDETR)
```
Input: RGB Image (any size)
  ↓
RFDETR Medium (Detection Transformer)
  ↓
Output: 
  - Bounding boxes (xyxy format)
  - Class IDs (14 classes)
  - Confidence scores
```

**Tính năng nổi bật**:
- Real-time detection
- End-to-end learning (không cần NMS)
- Hỗ trợ 14 loại bệnh lý phổi

### MRI Model (TransformerUNet)
```
Input: RGB Image 224x224
  ↓
Encoder (Conv + MaxPool):
  3 → 32 → 64 → 128 → 256
  ↓
Bottleneck (512 channels):
  + Positional Encoding
  + Multi-Head Self-Attention
  ↓
Decoder (Cross-Attention + ConvTranspose):
  256 → 128 → 64 → 32
  ↓
Output: Binary Mask (1 channel)
```

**Tính năng nổi bật**:
- Transformer attention để capture long-range dependencies
- Cross-attention giữa encoder và decoder
- Residual connections cho stable training
- Connected components analysis để loại bỏ noise

## So Sánh Hai Models

| Feature | RFDETR (X-ray) | TransformerUNet (MRI) |
|---------|----------------|----------------------|
| **Task** | Object Detection | Semantic Segmentation |
| **Input Size** | Variable | 224x224 (fixed) |
| **Output** | Boxes + Labels | Segmentation Mask |
| **Speed** | Fast (real-time) | Medium |
| **Attention** | Built-in DETR | Multi-head (custom) |
| **Classes** | 14 classes | Binary (0/1) |

## Framework và Dependencies

**Deep Learning**:
- PyTorch (core framework)
- rfdetr (cho X-ray detection)
- torch.nn.MultiheadAttention (cho MRI attention)

**Image Processing**:
- Pillow/PIL (load images)
- Albumentations (data augmentation & transforms)
- OpenCV/cv2 (post-processing)
- Supervision (visualization cho X-ray)

**Deployment**:
- FastAPI (REST API)
- Uvicorn (ASGI server)

## Cách Models Được Load

Cả hai models đều:
1. Hỗ trợ load từ local file path
2. Hỗ trợ load từ HTTP/HTTPS URL
3. Tự động download và cache weights
4. Tự động detect CUDA/CPU

```python
# X-ray
xray_model = DetectorModel(
    weights="models/xray/model.pth"  # hoặc URL
)

# MRI  
mri_model = SegmentorModel(
    weights="models/mri/model.pth"  # hoặc URL
)
```

## Tài Liệu Chi Tiết

📚 Xem file **MODELS.md** để biết:
- Kiến trúc chi tiết từng layer
- Code examples
- API endpoints
- Performance considerations
- References và papers

📖 Xem file **README.md** để biết:
- Hướng dẫn cài đặt
- Usage examples
- Project structure
- Configuration options

---

**Tóm lại**: Healix AI sử dụng 2 mô hình state-of-the-art:
1. **RFDETR** - phát hiện bệnh X-ray (detection)
2. **TransformerUNet** - phân đoạn MRI (segmentation)

Cả hai đều sử dụng Transformer attention mechanisms và được tối ưu cho medical imaging.
