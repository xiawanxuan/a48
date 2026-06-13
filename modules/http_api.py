import io
import os
import logging
import base64
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_app = None
_detector = None
_class_names = None


class DetectResponse(BaseModel):
    detections: list
    num_defects: int
    image_size: list


def create_app(model_path=None, num_classes=4, class_names=None,
               device="auto", score_threshold=0.5, nms_threshold=0.5):
    global _app, _detector, _class_names

    if class_names is None:
        class_names = ["scratch", "dent", "crack"]
    _class_names = class_names

    app = FastAPI(
        title="Defect Detection API",
        description="Precision parts defect detection inference service",
        version="1.0.0",
    )

    @app.on_event("startup")
    async def startup():
        global _detector
        if model_path and os.path.exists(model_path):
            from modules.inference import Detector
            _detector = Detector(
                model_path=model_path,
                num_classes=num_classes,
                device=device,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
            )
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning("No model path provided or file not found. API will return errors until model is loaded.")

    @app.post("/detect", response_model=DetectResponse)
    async def detect_image(file: UploadFile = File(...)):
        if _detector is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

        results = _detector.detect(img)

        return DetectResponse(
            detections=results,
            num_defects=len(results),
            image_size=list(img.size),
        )

    @app.post("/detect/base64")
    async def detect_base64(data: dict):
        if _detector is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field")

        try:
            image_data = base64.b64decode(data["image"])
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

        results = _detector.detect(img)

        return {
            "detections": results,
            "num_defects": len(results),
            "image_size": list(img.size),
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model_loaded": _detector is not None,
            "device": str(_detector.device) if _detector else "N/A",
            "class_names": _class_names,
        }

    @app.post("/reload")
    async def reload_model(data: dict):
        global _detector

        if "model_path" not in data:
            raise HTTPException(status_code=400, detail="Missing 'model_path' field")

        path = data["model_path"]
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {path}")

        try:
            from modules.inference import Detector
            _detector = Detector(
                model_path=path,
                num_classes=num_classes,
                device=device,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
            )
            return {"status": "reloaded", "model_path": path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    _app = app
    return app
