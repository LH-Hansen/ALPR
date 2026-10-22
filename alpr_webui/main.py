"""License plate recognition demo: FastAPI + two trained YOLO models.

Start with: python -m uvicorn main:app --host 127.0.0.1 --port 8000
"""

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

BASE = Path(__file__).resolve().parent
PLATE_WEIGHTS = BASE / "models" / "plate_detector.pt"
OCR_WEIGHTS = BASE / "models" / "character_reader.pt"
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
INFERENCE_LOCK = Lock()  # Use one inference at a time with shared YOLO instances.

app = FastAPI(title="ALPR Image Recognition", version="1.0.0")


@lru_cache(maxsize=1)
def get_models():
    """Load the two trained model weights once, on first recognition request."""
    if not PLATE_WEIGHTS.is_file() or not OCR_WEIGHTS.is_file():
        raise HTTPException(
            status_code=503,
            detail="Trained model weights missing. See README.md for model paths.",
        )
    from ultralytics import YOLO

    return YOLO(str(PLATE_WEIGHTS)), YOLO(str(OCR_WEIGHTS))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.post("/api/recognize")
def recognize(file: UploadFile = File(...)):
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
    detector, reader = get_models()
    plates = []

    # PIL RGB images are supported directly by Ultralytics YOLO.
    with INFERENCE_LOCK:
        detected = detector.predict(image, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
        for box in detected.boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Extra border reduces the chance of cutting off edge characters.
            padx = max(2, int((x2 - x1) * 0.02))
            pady = max(2, int((y2 - y1) * 0.05))
            crop = image.crop((max(0, x1 - padx), max(0, y1 - pady),
                               min(width, x2 + padx), min(height, y2 + pady)))
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

    return {"image_width": width, "image_height": height, "plates": plates}
