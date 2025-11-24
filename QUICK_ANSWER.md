# QUICK ANSWER / TRẢ LỜI NHANH

## Câu Hỏi: "Cho tôi hỏi các mô hình sử dụng trong này là gì?"

## Trả Lời Ngắn Gọn:

Healix AI sử dụng **2 mô hình AI**:

### 🩺 1. RFDETR - Phát Hiện Bệnh X-quang
**Chức năng:** Phát hiện bệnh lý trên ảnh X-quang phổi

**11 Bệnh Được Phát Hiện:**
1. 🫁 Aortic enlargement - Phình động mạch chủ
2. 🫁 Atelectasis - Xẹp phổi  
3. ❤️ Cardiomegaly - Tim to
4. 🫁 Consolidation - Đông đặc phổi
5. 🫁 ILD - Bệnh phổi kẽ
6. 🫁 Infiltration - Thâm nhiễm
7. 🫁 Lung Opacity - Mờ đục phổi
8. 🔍 Other lesion - Tổn thương khác
9. 💧 Pleural effusion - Tràn dịch màng phổi
10. 💨 Pneumothorax - Tràn khí màng phổi
11. 🫁 Pulmonary fibrosis - Xơ phổi

**Công nghệ:** Detection Transformer (DETR) - Real-time  
**Input:** Ảnh X-quang (bất kỳ kích thước)  
**Output:** Bounding boxes + tên bệnh + độ tin cậy

---

### 🧠 2. TransformerUNet - Phân Đoạn MRI
**Chức năng:** Phân đoạn vùng bệnh lý (khối u) trên ảnh MRI

**Công nghệ:** UNet + Transformer Attention  
**Đặc điểm:** 
- 4 attention heads
- Positional encoding
- Residual connections
- Connected components analysis

**Input:** Ảnh MRI 224×224  
**Output:** Segmentation mask (vùng bệnh lý được tô màu đỏ)

---

## So Sánh 2 Models:

| | RFDETR (X-ray) | TransformerUNet (MRI) |
|---|---|---|
| **Nhiệm vụ** | Phát hiện (Detection) | Phân đoạn (Segmentation) |
| **Loại bệnh** | 11 bệnh phổi | Khối u não |
| **Output** | Boxes + Labels | Pixel mask |
| **Tốc độ** | Nhanh (real-time) | Trung bình (1-2s) |

---

## Đọc Thêm:

📄 **MODELS_SUMMARY_VI.md** - Tóm tắt chi tiết (Tiếng Việt)  
📄 **MODELS.md** - Tài liệu kỹ thuật đầy đủ  
📄 **README.md** - Hướng dẫn sử dụng  
📄 **ARCHITECTURE_DIAGRAMS.md** - Sơ đồ kiến trúc  
📄 **DOCUMENTATION_INDEX.md** - Danh mục tài liệu  

---

## Sử Dụng Nhanh:

### X-ray Detection API:
```bash
POST http://localhost:8000/xray/lung/predict
# Upload ảnh X-quang, nhận về ảnh đã đánh dấu bệnh
```

### MRI Segmentation API:
```bash
POST http://localhost:8000/mri/predict
# Upload ảnh MRI, nhận về ảnh đã phân đoạn khối u
```

---

## Tóm Lại:

✅ **2 mô hình AI** cho phân tích hình ảnh y tế  
✅ **RFDETR** phát hiện **11 bệnh phổi** từ X-quang  
✅ **TransformerUNet** phân đoạn **khối u** từ MRI  
✅ Cả 2 đều dùng **Transformer** - công nghệ AI hiện đại  
✅ API đơn giản, dễ tích hợp  

---

**Ngày tạo:** 2024-11-24  
**Ngôn ngữ:** Tiếng Việt 🇻🇳
