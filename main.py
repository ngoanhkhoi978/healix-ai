from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import io
from PIL import Image
import os
from contextlib import asynccontextmanager

from models.xray.xray_model import DetectorModel
from models.mri.mri_model import SegmentorModel



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
MODEL_MRI_WEIGHTS = os.environ.get("MODEL_MRI_WEIGHTS", "models/mri/model.pth")


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
