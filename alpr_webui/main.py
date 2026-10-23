"""ALPR demo: fixed plate detector and selectable OCR model generations.

Start with: python -m uvicorn main:app --host 127.0.0.1 --port 8000
"""

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

BASE = Path(__file__).resolve().parent
PLATE_WEIGHTS = BASE / "models" / "plate_detector.pt"

# Only these preset local checkpoints can be selected by API callers.
# Generation 3 retains the original filename for backwards compatibility.
OCR_MODELS = {
    "gen1": {
        "name": "Generation 1 — Spanish",
        "path": BASE / "models" / "character_reader_gen1.pt",
    },
    "gen2": {
        "name": "Generation 2 — Spanish + Danish",
        "path": BASE / "models" / "character_reader_gen2.pt",
    },
    "gen3": {
        "name": "Generation 3 — Spanish + Danish + German",
        "path": BASE / "models" / "character_reader.pt",
    },
}
DEFAULT_OCR_MODEL = "gen3"
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
INFERENCE_LOCK = Lock()  # Serialize calls using shared YOLO instances.

app = FastAPI(title="ALPR Image Recognition", version="1.1.0")


@lru_cache(maxsize=1)
def get_plate_detector():
    if not PLATE_WEIGHTS.is_file():
        raise HTTPException(status_code=503, detail="Missing plate_detector.pt in models/.")
    from ultralytics import YOLO
    return YOLO(str(PLATE_WEIGHTS))


@lru_cache(maxsize=len(OCR_MODELS))
def get_ocr_reader(model_id: str):
    # This is a fixed whitelist, not a filename supplied by the browser.
    entry = OCR_MODELS.get(model_id)
    if entry is None:
        raise HTTPException(status_code=422, detail="Unknown OCR model generation.")
    weights = entry["path"]
    if not weights.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Missing weights for {entry['name']}: {weights.name}",
        )
    from ultralytics import YOLO
    return YOLO(str(weights))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/models")
def list_models():
    """Expose the permitted OCR generations and which weights are installed."""
    return {
        "default_model": DEFAULT_OCR_MODEL,
        "models": [
            {
                "id": model_id,
                "name": entry["name"],
                "available": entry["path"].is_file(),
            }
            for model_id, entry in OCR_MODELS.items()
        ],
    }


@app.post("/api/recognize")
def recognize(file: UploadFile = File(...), ocr_model: str = Form(DEFAULT_OCR_MODEL)):
    # Old API clients can omit 'ocr_model' and continue using Generation 3.
    if ocr_model not in OCR_MODELS:
        raise HTTPException(status_code=422, detail="Unknown OCR model generation.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Use JPEG, PNG or WebP.")

    content = file.file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Maximum file size is 10 MB.")

    try:
        with Image.open(BytesIO(content)) as uploaded:
            if uploaded.width * uploaded.height > MAX_PIXELS:
                raise HTTPException(
                    status_code=413, detail="Maximum image resolution is 20 megapixels."
                )
            image = ImageOps.exif_transpose(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid or corrupted image.")

    width, height = image.size
    plates = []

    with INFERENCE_LOCK:
        detector = get_plate_detector()
        reader = get_ocr_reader(ocr_model)
        detected = detector.predict(image, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]

        for box in detected.boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # A small border helps avoid cropping off the first/last character.
            padx = max(2, int((x2 - x1) * 0.02))
            pady = max(2, int((y2 - y1) * 0.05))
            crop = image.crop(
                (max(0, x1 - padx), max(0, y1 - pady),
                 min(width, x2 + padx), min(height, y2 + pady))
            )
            found = reader.predict(crop, imgsz=480, conf=0.25,
                                   device="cpu", verbose=False)[0]
            chars = []
            for char_box in found.boxes:
                chars.append({
                    "x": float(char_box.xyxy[0][0]),
                    "value": str(reader.names[int(char_box.cls[0])]),
                    "confidence": round(float(char_box.conf[0]), 4),
                })
            chars.sort(key=lambda char: char["x"])
            plates.append({
                "text": "".join(char["value"] for char in chars),
                "confidence": round(float(box.conf[0]), 4),
                "box": [x1, y1, x2, y2],
                "characters": [
                    {"value": char["value"], "confidence": char["confidence"]}
                    for char in chars
                ],
            })

    return {
        "image_width": width,
        "image_height": height,
        "ocr_model": ocr_model,
        "ocr_model_name": OCR_MODELS[ocr_model]["name"],
        "plates": plates,
    }
