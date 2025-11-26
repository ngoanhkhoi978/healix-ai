# Healix AI

Healix AI là một hệ thống phân tích hình ảnh y tế sử dụng deep learning để hỗ trợ chẩn đoán bệnh từ ảnh X-quang và MRI.

## 🎯 Tính Năng Chính

- **Phát hiện bệnh lý trên X-quang phổi**: Sử dụng RFDETR để phát hiện 11 loại bệnh phổi
- **Phân đoạn ảnh MRI**: Sử dụng TransformerUNet để phân đoạn các vùng bệnh lý
- **RESTful API**: FastAPI endpoints để tích hợp dễ dàng
- **Real-time Processing**: Xử lý và trả kết quả nhanh chóng
- **Flexible Weight Loading**: Hỗ trợ local files và URLs

## 🤖 Mô Hình AI

Healix AI sử dụng hai mô hình deep learning tiên tiến:

### 1. X-ray Detection (RFDETR)
- **Kiến trúc**: Real-time Detection Transformer
- **Chức năng**: Phát hiện bệnh lý trên ảnh X-quang phổi
- **Output**: Bounding boxes với class labels và confidence scores
- **Classes**: 11 loại bệnh phổi (Aortic enlargement, Atelectasis, Cardiomegaly, Consolidation, ILD, Infiltration, Lung Opacity, Other lesion, Pleural effusion, Pneumothorax, Pulmonary fibrosis)

### 2. MRI Segmentation (TransformerUNet)
- **Kiến trúc**: UNet với Transformer Attention
- **Chức năng**: Phân đoạn vùng bệnh lý trên ảnh MRI
- **Features**: Multi-head attention, positional encoding, residual connections
- **Output**: Binary segmentation mask

📚 **Chi tiết về mô hình**: Xem [MODELS.md](./MODELS.md) để biết thêm thông tin chi tiết về kiến trúc và cách sử dụng.

## 🚀 Cài Đặt

### Requirements
```bash
# Python 3.8+
pip install torch torchvision
pip install fastapi uvicorn
pip install pillow requests
pip install albumentations
pip install supervision
pip install rfdetr
pip install opencv-python
```

### Chạy API Server
```bash
python -m uvicorn main:app --reload
```

Server sẽ chạy tại: `http://localhost:8000`

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Kiểm tra trạng thái của các models

### X-ray Analysis

**Predict với Image Output:**
```bash
POST /xray/lung/predict
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPEG/PNG)
- threshold: Confidence threshold (default: 0.3)
```

**Predict với JSON Output:**
```bash
POST /xray/lung/predict_with_json
Content-Type: multipart/form-data

Parameters:
- file: Image file
- threshold: Confidence threshold (default: 0.3)

Returns: JSON với base64 image và detection data
```

### MRI Analysis

**Segment với Image Output:**
```bash
POST /mri/predict
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPEG/PNG)
- threshold: Segmentation threshold (default: 0.5)
```

**Segment với JSON Output:**
```bash
POST /mri/predict_with_json
Content-Type: multipart/form-data

Parameters:
- file: Image file
- threshold: Segmentation threshold (default: 0.5)

Returns: JSON với base64 overlay image và mask metadata
```

**Debug Endpoint:**
```bash
POST /mri/debug
Returns: Overlay image, raw mask, và model loading errors
```

## 💻 Usage Examples

### Python Client Example

```python
import requests

# X-ray Analysis
with open("xray_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/xray/lung/predict",
        files={"file": f},
        params={"threshold": 0.3}
    )
    
with open("result.png", "wb") as f:
    f.write(response.content)

# MRI Segmentation
with open("mri_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/mri/predict",
        files={"file": f},
        params={"threshold": 0.5}
    )
    
with open("segmented.png", "wb") as f:
    f.write(response.content)
```

### Direct Model Usage

**X-ray Detection:**
```python
from models.xray.xray_model import DetectorModel

model = DetectorModel(weights="models/xray/model.pth")
detections, annotated = model.diagnose_image(
    "path/to/xray.jpg", 
    threshold=0.3
)
```

**MRI Segmentation:**
```python
from models.mri.mri_model import SegmentorModel

model = SegmentorModel(weights="models/mri/model.pth")
mask, overlay = model.segment_image(
    "path/to/mri.jpg",
    threshold=0.5
)
```

## 📁 Project Structure

```
healix-ai/
├── main.py                    # FastAPI application
├── models/
│   ├── xray/
│   │   ├── xray_model.py     # RFDETR wrapper
│   │   ├── model.pth         # Pretrained weights
│   │   └── coco_annotations_val.json
│   └── mri/
│       ├── mri_model.py      # SegmentorModel wrapper
│       ├── TransformerUNet.py # Model architecture
│       ├── EncoderDecoder.py  # Encoder/Decoder components
│       └── model.pth         # Pretrained weights
├── MODELS.md                  # Detailed model documentation
└── README.md                  # This file
```

## 🔧 Configuration

Model weights có thể được cấu hình qua environment variables:

```bash
export MODEL_XRAY_WEIGHTS="models/xray/model.pth"
export MODEL_MRI_WEIGHTS="models/mri/model.pth"
```

Hoặc sử dụng URLs:
```bash
export MODEL_XRAY_WEIGHTS="https://example.com/xray_weights.pth"
export MODEL_MRI_WEIGHTS="https://example.com/mri_weights.pth"
```

## 🌟 Features

- ✅ CUDA/GPU support tự động
- ✅ Flexible weight loading (local/URL)
- ✅ Automatic caching cho weights từ URLs
- ✅ CORS enabled cho web integration
- ✅ Comprehensive error handling
- ✅ Debug endpoints cho development
- ✅ Health check endpoints
- ✅ JSON và Image response formats

## 📊 Model Performance

### X-ray Model
- **Speed**: Real-time inference
- **Classes**: 11 bệnh phổi
- **Input**: Variable size (tự động resize)

### MRI Model
- **Input Size**: 224x224 (tự động resize)
- **Attention Heads**: 4
- **Channels**: (3, 32, 64, 128, 256, 512)
- **Features**: Transformer attention, residual connections

## 🛠️ Development

### Testing API
Sử dụng file `test_main.http` để test các endpoints:
```http
POST http://localhost:8000/xray/lung/predict
Content-Type: multipart/form-data
```

### Adding New Models
1. Tạo model wrapper trong `models/<model_type>/`
2. Implement `__init__` và inference methods
3. Add vào `main.py` lifespan handler
4. Thêm endpoints tương ứng

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributors here]

## 📧 Contact

[Add contact information here]

---

**Lưu ý**: Để biết thông tin chi tiết về kiến trúc mô hình, cách sử dụng, và API specifications, vui lòng xem [MODELS.md](./MODELS.md).
