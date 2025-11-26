# Healix AI - Documentation Index

## 📚 Trả Lời: "Cho tôi hỏi các mô hình sử dụng trong này là gì"

Healix AI sử dụng **2 mô hình AI chính**: **RFDETR** (X-ray detection) và **TransformerUNet** (MRI segmentation).

---

## 📖 Tài Liệu / Documentation

### 🚀 Bắt Đầu Nhanh / Quick Start
- **[README.md](./README.md)** - Tổng quan dự án, hướng dẫn cài đặt và sử dụng
  - Cài đặt dependencies
  - Chạy API server
  - Ví dụ sử dụng
  - Cấu trúc dự án

### 🎯 Câu Trả Lời Ngắn Gọn / Quick Answer (Vietnamese)
- **[MODELS_SUMMARY_VI.md](./MODELS_SUMMARY_VI.md)** - Tóm tắt nhanh về các mô hình (Tiếng Việt)
  - Mô hình nào được sử dụng?
  - Chức năng của từng mô hình
  - So sánh 2 mô hình
  - Framework và công nghệ

### 📘 Chi Tiết Kỹ Thuật / Technical Details
- **[MODELS.md](./MODELS.md)** - Tài liệu kỹ thuật đầy đủ
  - Kiến trúc chi tiết RFDETR
  - Kiến trúc chi tiết TransformerUNet
  - Input/Output specifications
  - API endpoints
  - Usage examples
  - References và citations

### 🎨 Sơ Đồ Kiến Trúc / Architecture Diagrams
- **[ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md)** - Sơ đồ trực quan
  - Luồng dữ liệu (Data flow)
  - Kiến trúc mô hình (Model architecture)
  - Attention mechanisms
  - So sánh mô hình (Model comparison)
  - Technology stack

---

## 🤖 Tóm Tắt Mô Hình / Models Summary

### 1. RFDETR - X-ray Lung Disease Detection
```
📍 Location: models/xray/xray_model.py
🎯 Purpose: Phát hiện bệnh lý trên X-quang phổi
🏗️ Architecture: Detection Transformer (Real-time DETR)
📊 Output: Bounding boxes + 11 lung diseases
🏥 Diseases: Aortic enlargement, Atelectasis, Cardiomegaly, Consolidation, 
            ILD, Infiltration, Lung Opacity, Other lesion, Pleural effusion, 
            Pneumothorax, Pulmonary fibrosis
⚡ Speed: Real-time (~fast)
📦 Weight: models/xray/model.pth
```

### 2. TransformerUNet - MRI Tumor Segmentation
```
📍 Location: models/mri/mri_model.py
🎯 Purpose: Phân đoạn khối u trên ảnh MRI
🏗️ Architecture: UNet + Transformer Attention
📊 Output: Binary segmentation mask
⚡ Speed: Medium (~1-2 seconds)
📦 Weight: models/mri/model.pth
🧠 Features: 4-head attention, positional encoding, residual connections
```

---

## 🔍 Tìm Thông Tin Theo Chủ Đề / Find Information by Topic

### Nếu bạn muốn biết...

#### "Mô hình nào được sử dụng?"
→ Đọc: [MODELS_SUMMARY_VI.md](./MODELS_SUMMARY_VI.md) - Phần "Tóm Tắt Nhanh"

#### "Kiến trúc mô hình như thế nào?"
→ Đọc: 
- [MODELS.md](./MODELS.md) - Technical details
- [ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md) - Visual diagrams

#### "Cách sử dụng models?"
→ Đọc: 
- [README.md](./README.md) - Usage examples
- [MODELS.md](./MODELS.md) - Detailed usage

#### "API endpoints là gì?"
→ Đọc: 
- [README.md](./README.md) - API Endpoints section
- [MODELS.md](./MODELS.md) - Complete API documentation

#### "Cài đặt và chạy như thế nào?"
→ Đọc: [README.md](./README.md) - Installation section

#### "References và papers?"
→ Đọc: [MODELS.md](./MODELS.md) - References section

---

## 📊 Thống Kê Tài Liệu / Documentation Stats

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| README.md | 255 | 6.4 KB | Project overview |
| MODELS_SUMMARY_VI.md | 140 | 3.8 KB | Quick Vietnamese answer |
| MODELS.md | 320 | 8.2 KB | Technical documentation |
| ARCHITECTURE_DIAGRAMS.md | 350 | 18 KB | Visual diagrams |
| **Total** | **1,065** | **~36 KB** | Complete documentation |

---

## 🛠️ Code Files

### X-ray Model Files
```
models/xray/
├── xray_model.py              # RFDETR wrapper
├── model.pth                  # Pretrained weights
└── coco_annotations_val.json  # Class definitions
```

### MRI Model Files
```
models/mri/
├── mri_model.py          # SegmentorModel wrapper
├── TransformerUNet.py    # Main architecture
├── EncoderDecoder.py     # Encoder/Decoder components
└── model.pth             # Pretrained weights
```

### API Server
```
main.py                   # FastAPI application
test_main.http           # API testing file
```

---

## 🌐 API Endpoints Quick Reference

### Health Check
```
GET /health
```

### X-ray Detection
```
POST /xray/lung/predict                 # Returns annotated image
POST /xray/lung/predict_with_json       # Returns image + JSON data
```

### MRI Segmentation
```
POST /mri/predict                       # Returns overlay image
POST /mri/predict_with_json             # Returns image + mask metadata
POST /mri/debug                         # Debug info + raw mask
```

---

## 💡 Recommended Reading Order

### For Beginners:
1. [MODELS_SUMMARY_VI.md](./MODELS_SUMMARY_VI.md) - Hiểu nhanh về models
2. [README.md](./README.md) - Hướng dẫn sử dụng
3. [ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md) - Xem sơ đồ trực quan

### For Developers:
1. [README.md](./README.md) - Setup và API usage
2. [MODELS.md](./MODELS.md) - Technical specifications
3. Source code: `models/xray/`, `models/mri/`

### For Researchers:
1. [MODELS.md](./MODELS.md) - Architecture và references
2. [ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md) - Detailed diagrams
3. [MODELS_SUMMARY_VI.md](./MODELS_SUMMARY_VI.md) - Quick comparison

---

## 📞 Additional Resources

- **Source Code**: Check `models/` directory for implementation
- **API Testing**: Use `test_main.http` for endpoint testing
- **Main Application**: See `main.py` for FastAPI setup

---

**Last Updated**: 2025-11-24

**Language**: Vietnamese (Tiếng Việt) + English
