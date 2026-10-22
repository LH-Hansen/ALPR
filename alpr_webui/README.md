# ALPR web UI + FastAPI

A local Windows 11 application that serves a simple image-upload page and a reusable JSON API.
It uses **two YOLO models that you already trained** with UC3M-LP.

## 1. Put the trained weights in the models folder

Copy your two training output files:

| Source | Destination |
|---|---|
| `C:\ALPR\runs\plate_detector\weights\best.pt` | `models\plate_detector.pt` |
| `C:\ALPR\runs\character_reader\weights\best.pt` | `models\character_reader.pt` |

Do not rename weights unless you also update the paths in `main.py`.
Training is not part of this web application.

## 2. Install dependencies (PowerShell, Windows 11)

Use the same venv as your YOLO training setup, or create a new one:

```powershell
cd C:\ALPR\alpr_webui
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `C:\ALPR\.venv` already has YOLO/PyTorch installed, you can use that interpreter and install the packages in `requirements.txt` there instead.

## 3. Start the server

```powershell
cd C:\ALPR\alpr_webui
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open:
- **UI**: http://127.0.0.1:8000/
- **Interactive API docs**: http://127.0.0.1:8000/docs

## 4. Test the JSON API

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/recognize -F "file=@C:\ALPR\test_car.jpg"
```

Example response (illustrative):

```json
{
  "image_width": 1280,
  "image_height": 720,
  "plates": [
    {
      "text": "AB12345",
      "confidence": 0.94,
      "box": [425, 379, 680, 443],
      "characters": [
        {"value": "A", "confidence": 0.92},
        {"value": "B", "confidence": 0.88}
      ]
    }
  ]
}
```

The example abbreviated the character list; in real output, it contains one item per recognized character.
`confidence` on a plate means **plate-detection confidence**, not confidence that the whole string is correct.

## 5. Open in Visual Studio 2022

Install the **Python development** workload in Visual Studio Installer.
Create a **From existing Python code** project pointing to `C:\ALPR\alpr_webui` and select `main.py`.
Select the interpreter `C:\ALPR\alpr_webui\.venv\Scripts\python.exe`.
Use the **View > Terminal** PowerShell terminal to run the Uvicorn command above; Uvicorn is the web server, and `main.py` is imported as an ASGI app.

## Notes / scope

- API supports **JPEG, PNG, WebP**, up to **10 MB** and **20 megapixels**.
- Uploaded image contents are processed in memory; this demo does not save them on the server.
- The frontend uses the API response to draw bounding boxes on the user's local preview.
- Supports multiple plates in one image. Characters are sorted left-to-right; double-row plates need more logic.
- `GET /` is the UI, `POST /api/recognize` is the API and `GET /docs` is OpenAPI docs.
- By default the server binds to localhost only. This is a **local development demo**: add authentication, TLS, request limits, security review and privacy safeguards before exposing it publicly. License plates can be personal data.
- The original UC3M-LP dataset is mostly Spanish; Danish plate recognition requires representative testing and possibly retraining.
- Sharing YOLO instances across threads is guarded by an inference lock, so CPU work is serialized in this demo.
- Ultralytics licensing (AGPL or commercial options) and dataset licensing require review before certain distributions or deployments.
