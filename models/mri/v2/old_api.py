
Điểm nổi bật trong thư mục
Tài liệu hỗ trợ ứng dụng FastAPI phục vụ phân đoạn não bằng mô hình TransformerUNet trên các tệp NIfTI.

test_api.py
import os
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from models.TransformerUNet import TransformerUNet
import albumentations
from albumentations.pytorch import ToTensorV2
from PIL import Image
import io
import base64
import numpy as np
from torch import nn
import nibabel as nib
import tempfile
import shutil
import albumentations as A

# Thiết lập device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
channels = (4, 32, 64, 128, 256, 512)
model = TransformerUNet(
    channels=channels,
    num_heads=4,
    is_residual=True,
    bias=False
).to(device)

# Load best weights nếu có
model_path = "models/best_brats_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded best model weights.")
else:
    print("No best model weights found, using untrained model.")

model.eval()

# Transform cho ảnh input
transform = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

def pad_to_square(image, pad_value=0):
    h, w = image.shape

    max_dim = max(h, w)

    if max_dim != 218:
        print("Warning: Unexpected image dimension:", (h, w))

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


def read_nifti_data(file_bytes):
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

def process_nifti_files(t2w_bytes, t2f_bytes, t1n_bytes, t1c_bytes):
    """Process 4 NIfTI files - WITHOUT padding, just resize like training"""

    t2w_data = read_nifti_data(t2w_bytes)
    t2f_data = read_nifti_data(t2f_bytes)
    t1n_data = read_nifti_data(t1n_bytes)
    t1c_data = read_nifti_data(t1c_bytes)

    num_slices = t2w_data.shape[2]
    processed_slices = []

    for i in range(num_slices):
        slice_channels = []
        slice_2d_t2w = t2w_data[:, :, i]
        cropped_slice_t2w = pad_to_square(slice_2d_t2w)
        cropped_slice_t2w = cropped_slice_t2w.astype(np.float16)
        slice_channels.append(cropped_slice_t2w)

        slice_2d_t2f = t2f_data[:, :, i]
        cropped_slice_t2f = pad_to_square(slice_2d_t2f)
        cropped_slice_t2f = cropped_slice_t2f.astype(np.float16)
        slice_channels.append(cropped_slice_t2f)

        slice_2d_t1n = t1n_data[:, :, i]
        cropped_slice_t1n = pad_to_square(slice_2d_t1n)
        cropped_slice_t1n = cropped_slice_t1n.astype(np.float16)
        slice_channels.append(cropped_slice_t1n)

        slice_2d_t1c = t1c_data[:, :, i]
        cropped_slice_t1c = pad_to_square(slice_2d_t1c)
        cropped_slice_t1c = cropped_slice_t1c.astype(np.float16)
        slice_channels.append(cropped_slice_t1c)

        stacked_image = np.stack(slice_channels, axis=0).astype(np.float16)
        processed_slices.append(stacked_image)

    return processed_slices


def predict_slice(slice_data, threshold=0.5):
    """Predict mask for a single slice"""

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
    image_tensor = torch.from_numpy(image_tensor).float().unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        preds = torch.sigmoid(outputs)
        preds = (preds > threshold).float()
        mask = preds.squeeze(0).squeeze(0).cpu().numpy()

    return mask

app = FastAPI()

def array_to_base64(array, normalize=True):
    """Convert numpy array to base64 PNG"""
    if normalize and array.max() > array.min():
        array = (array - array.min()) / (array.max() - array.min())

    img = Image.fromarray((array * 255).astype(np.uint8), mode='L')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@app.post("/segment_brain")
async def segment_brain(
        t2w: UploadFile = File(...),
        t2f: UploadFile = File(...),
        t1n: UploadFile = File(...),
        t1c: UploadFile = File(...),
        threshold: float = 0.5
):
    try:
        print("📥 Receiving files...")

        # Read file bytes
        t2w_bytes = await t2w.read()
        t2f_bytes = await t2f.read()
        t1n_bytes = await t1n.read()
        t1c_bytes = await t1c.read()

        print("🔄 Processing NIfTI files...")
        processed_slices = process_nifti_files(t2w_bytes, t2f_bytes, t1n_bytes, t1c_bytes)

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
            predicted_mask = predict_slice(slice_data, threshold=threshold)

            # Convert all to base64
            slice_result = {
                "slice_index": idx,
                "images": {
                    "t2w": array_to_base64(t2w_slice),
                    "t2f": array_to_base64(t2f_slice),
                    "t1n": array_to_base64(t1n_slice),
                    "t1c": array_to_base64(t1c_slice)
                },
                "predicted_mask": array_to_base64(predicted_mask, normalize=False),
                "statistics": {
                    "tumor_pixels": int(predicted_mask.sum()),
                    "total_pixels": int(predicted_mask.size),
                    "tumor_percentage": float(predicted_mask.sum() / predicted_mask.size * 100)
                }
            }

            results.append(slice_result)

        print(f"✅ Completed {len(results)} slices")

        response = {
            "total_slices": len(results),
            "threshold": threshold,
            "slices": results
        }

        return JSONResponse(content=response)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/")
async def root():
    return {
        "message": "Brain Tumor Segmentation API",
        "endpoints": {
            "/segment_brain": "POST - Upload 4 NIfTI files for full brain segmentation"
        }
    }
DDict Logo