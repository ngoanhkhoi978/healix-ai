from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import io
from PIL import Image
import os
import numpy as np
from contextlib import asynccontextmanager
from typing import List

from models.xray.xray_model import DetectorModel
from models.mri.v1.mri_model import SegmentorModel
from models.mri.v2.mri_model_v2 import SegmentorModelV2



app = FastAPI(title="Healix XRAY API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model weights locations can be provided via env vars
MODEL_XRAY_WEIGHTS = os.environ.get("MODEL_XRAY_WEIGHTS", "models/xray/model.pth")
MODEL_MRI_WEIGHTS = os.environ.get("MODEL_MRI_WEIGHTS", "models/mri/v1/model.pth")
MODEL_MRI_V2_WEIGHTS = os.environ.get("MODEL_MRI_V2_WEIGHTS", "models/mri/v2/model.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: load models at startup and keep them on app.state for request handlers."""
    app.state.models = {}
    errors = {}
    try:
        app.state.models['xray'] = DetectorModel(weights=MODEL_XRAY_WEIGHTS)
    except Exception as e:
        app.state.models['xray'] = None
        errors['xray'] = str(e)

    try:
        app.state.models['mri'] = SegmentorModel(weights=MODEL_MRI_WEIGHTS)
    except Exception as e:
        app.state.models['mri'] = None
        errors['mri'] = str(e)

    try:
        app.state.models['mri_v2'] = SegmentorModelV2(weights=MODEL_MRI_V2_WEIGHTS)
    except Exception as e:
        app.state.models['mri_v2'] = None
        errors['mri_v2'] = str(e)

    app.state.errors = errors
    yield
    # optionally cleanup models here


# recreate app with lifespan to ensure handler is used
app.router.routes.clear()
app = FastAPI(title="Healix XRAY API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    """Return readiness status per model."""
    models = getattr(app.state, 'models', {})
    status = {name: (mdl is not None) for name, mdl in models.items() if name != 'errors'}
    errors = getattr(app.state, 'errors', {})
    return {"ready": all(status.values()) if status else False, "models": status, "errors": errors}


@app.post("/xray/lung/predict")
async def xray_inference(file: UploadFile = File(...), threshold: float = 0.3):
    """Accept an uploaded X-ray image file, run X-ray model inference, and return the annotated image."""
    if not hasattr(app.state, "models") or app.state.models.get('xray') is None:
        raise HTTPException(status_code=503, detail="X-ray model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    try:
        detections, annotated = app.state.models['xray'].diagnose_image(pil_image, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/xray/lung/predict_with_json")
async def xray_inference_with_json(file: UploadFile = File(...), threshold: float = 0.3):
    """Return both annotated X-ray image and JSON detections (base64 + metadata)."""
    import base64

    if not hasattr(app.state, "models") or app.state.models.get('xray') is None:
        raise HTTPException(status_code=503, detail="X-ray model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    try:
        detections, annotated = app.state.models['xray'].diagnose_image(pil_image, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    try:
        class_ids = [int(x) for x in detections.class_id]
        confidences = [float(x) for x in detections.confidence]
        boxes = [list(map(float, b)) for b in detections.xyxy]
    except Exception:
        class_ids = str(detections.class_id)
        confidences = str(detections.confidence)
        boxes = str(getattr(detections, 'xyxy', ''))

    return JSONResponse(content={
        "image_base64": b64,
        "detections": {
            "class_id": class_ids,
            "confidence": confidences,
            "boxes_xyxy": boxes,
        }
    })


@app.post("/mri/predict")
async def mri_inference(file: UploadFile = File(...), threshold: float = 0.5):
    """Accept an uploaded MRI image file, run MRI segmentation, and return the overlay image."""
    if not hasattr(app.state, "models") or app.state.models.get('mri') is None:
        raise HTTPException(status_code=503, detail="MRI model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    try:
        pred_mask, overlayed = app.state.models['mri'].segment_image(pil_image, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    buf = io.BytesIO()
    overlayed.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/mri/predict_with_json")
async def mri_inference_with_json(file: UploadFile = File(...), threshold: float = 0.5):
    """Return annotated MRI overlay and a minimal mask metadata as JSON (base64 image + mask shape)."""
    import base64

    if not hasattr(app.state, "models") or app.state.models.get('mri') is None:
        raise HTTPException(status_code=503, detail="MRI model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    try:
        pred_mask, overlayed = app.state.models['mri'].segment_image(pil_image, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    buf = io.BytesIO()
    overlayed.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    # prepare minimal mask metadata
    try:
        mask_shape = list(pred_mask.shape)
    except Exception:
        mask_shape = None

    return JSONResponse(content={
        "image_base64": b64,
        "mask_shape": mask_shape,
    })


@app.post("/mri/debug")
async def mri_debug(file: UploadFile = File(...), threshold: float = 0.5):
    """Return overlay image and raw mask (both base64) plus startup model load errors so you can inspect mask and verify checkpoint load."""
    import base64

    if not hasattr(app.state, "models") or app.state.models.get('mri') is None:
        raise HTTPException(status_code=503, detail="MRI model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    try:
        pred_mask, overlayed = app.state.models['mri'].segment_image(pil_image, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # overlay base64
    buf = io.BytesIO()
    overlayed.save(buf, format="PNG")
    buf.seek(0)
    overlay_b64 = base64.b64encode(buf.read()).decode('utf-8')


    # mask base64 (convert to PNG)
    try:
        mask_img = Image.fromarray((pred_mask * 255).astype('uint8'))
    except Exception:
        # fallback if pred_mask is not numpy array
        mask_img = Image.fromarray((np.array(pred_mask) * 255).astype('uint8'))
    buf2 = io.BytesIO()
    mask_img.save(buf2, format="PNG")
    buf2.seek(0)
    mask_b64 = base64.b64encode(buf2.read()).decode('utf-8')

    errors = getattr(app.state, 'errors', {})

    return JSONResponse(content={
        "overlay_base64": overlay_b64,
        "mask_base64": mask_b64,
        "errors": errors
    })


# ==================== MRI V2 ENDPOINTS ====================

@app.post("/mri/v2/predict")
async def mri_v2_predict(
    t2w: UploadFile = File(...),
    t2f: UploadFile = File(...),
    t1n: UploadFile = File(...),
    t1c: UploadFile = File(...),
    threshold: float = 0.5
):
    """Accept 4 MRI modality files (.gz format) and return full segmentation results exactly like original API."""
    import base64

    if not hasattr(app.state, "models") or app.state.models.get('mri_v2') is None:
        raise HTTPException(status_code=503, detail="MRI v2 model not loaded")

    try:
        print("📥 Receiving files...")

        # Read all 4 files
        t2w_bytes = await t2w.read()
        t2f_bytes = await t2f.read()
        t1n_bytes = await t1n.read()
        t1c_bytes = await t1c.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read files: {e}")

    try:
        # Process all slices
        results = app.state.models['mri_v2'].segment_from_files_all_slices(
            t2w_bytes, t2f_bytes, t1n_bytes, t1c_bytes,
            threshold=threshold
        )

        # Convert numpy arrays to base64
        def array_to_base64(array, normalize=True):
            """Convert numpy array to base64 PNG"""
            if normalize and array.max() > array.min():
                array = (array - array.min()) / (array.max() - array.min())

            img = Image.fromarray((array * 255).astype(np.uint8), mode='L')
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"

        # Convert all numpy arrays to base64
        formatted_results = []
        for slice_result in results:
            formatted_slice = {
                "slice_index": slice_result["slice_index"],
                "images": {
                    "t2w": array_to_base64(slice_result["images"]["t2w"]),
                    "t2f": array_to_base64(slice_result["images"]["t2f"]),
                    "t1n": array_to_base64(slice_result["images"]["t1n"]),
                    "t1c": array_to_base64(slice_result["images"]["t1c"])
                },
                "predicted_mask": array_to_base64(slice_result["predicted_mask"], normalize=False),
                "statistics": slice_result["statistics"]
            }
            formatted_results.append(formatted_slice)

        response = {
            "total_slices": len(formatted_results),
            "threshold": threshold,
            "slices": formatted_results
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")


@app.post("/mri/v2/predict_with_json")
async def mri_v2_inference_with_json(file: UploadFile = File(...), threshold: float = 0.5):
    """Return annotated MRI v2 overlay and mask metadata as JSON (base64 image + mask shape)."""
    import base64

    if not hasattr(app.state, "models") or app.state.models.get('mri_v2') is None:
        raise HTTPException(status_code=503, detail="MRI v2 model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    try:
        pred_mask, overlayed = app.state.models['mri_v2'].segment_image(pil_image, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    buf = io.BytesIO()
    overlayed.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    # prepare minimal mask metadata
    try:
        mask_shape = list(pred_mask.shape)
        mask_sum = int(pred_mask.sum())
    except Exception:
        mask_shape = None
        mask_sum = None

    return JSONResponse(content={
        "image_base64": b64,
        "mask_shape": mask_shape,
        "mask_pixel_count": mask_sum,
    })


@app.post("/mri/v2/debug")
async def mri_v2_debug(file: UploadFile = File(...), threshold: float = 0.5):
    """Return overlay image and raw mask (both base64) for MRI v2 plus startup model load errors."""
    import base64

    if not hasattr(app.state, "models") or app.state.models.get('mri_v2') is None:
        raise HTTPException(status_code=503, detail="MRI v2 model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    try:
        pred_mask, overlayed = app.state.models['mri_v2'].segment_image(pil_image, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # overlay base64
    buf = io.BytesIO()
    overlayed.save(buf, format="PNG")
    buf.seek(0)
    overlay_b64 = base64.b64encode(buf.read()).decode('utf-8')

    # mask base64 (convert to PNG)
    try:
        mask_img = Image.fromarray((pred_mask * 255).astype('uint8'))
    except Exception:
        # fallback if pred_mask is not numpy array
        mask_img = Image.fromarray((np.array(pred_mask) * 255).astype('uint8'))
    buf2 = io.BytesIO()
    mask_img.save(buf2, format="PNG")
    buf2.seek(0)
    mask_b64 = base64.b64encode(buf2.read()).decode('utf-8')

    errors = getattr(app.state, 'errors', {})

    return JSONResponse(content={
        "overlay_base64": overlay_b64,
        "mask_base64": mask_b64,
        "errors": errors,
        "model_version": "v2"
    })
