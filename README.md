# Automatic License Plate Recognition (ALPR)

I'm building a small machine learning project that can find a license plate in a photograph and read the characters on it. The idea is to make it easy to test different images through a simple website, while keeping the recognition itself available through an API for other applications.

The project started with Spanish plates using the [UC3M-LP dataset](https://github.com/ramajoballester/UC3M-LP). I've since worked on expanding the character-recognition training data with **synthetic Danish and German plate images**. It's still a work in progress: training on more plate styles doesn't automatically mean it will read real plates from every country reliably.

![Example of a recognized plate](test_car_result.jpg)

## How it works

There are two separate YOLO models:

1. **Plate detector:** Finds the license plate in a vehicle photo and returns its position.
2. **Character reader:** Takes the cropped plate image, detects letters and digits, and puts them into reading order.

I keep these models separate so I can improve character recognition without retraining the plate detector every time. The reader currently uses the same **36 classes (0–9 and A–Z)** across Spanish, Danish and German training examples.

The web app sends an uploaded image to a [FastAPI](https://fastapi.tiangolo.com/) endpoint and draws the detections over the image preview. The API also returns the results as JSON, so a different frontend could use it later.

## What's in this repository?

```text
ALPR/
├── ALPRApp/               # Visual Studio / Python development work
├── alpr_webui/            # FastAPI backend and web frontend
│   ├── main.py
│   ├── requirements.txt
│   ├── static/
│   └── models/
│       ├── plate_detector.pt
│       └── character_reader.pt
├── test_car.jpg           # Example input image
├── test_car_result.jpg    # Example recognition output
├── .gitignore
├── .gitattributes         # Git LFS configuration for model files
└── README.md
```

The two `.pt` files are the **trained weights used by the web app**. They're intended to be distributed using **Git LFS**, while the raw training datasets, virtual environments, and intermediate training runs stay outside Git. If the model files aren't available in a particular clone, the API can't run recognition until they're added.

## Run the web app locally

These instructions are for **Windows 11 with Python 3.11**. I use Visual Studio 2022 for development, but you can run the app directly from PowerShell.

### 1. Clone the repository

Install [Git](https://git-scm.com/) and [Git LFS](https://git-lfs.com/) first. Then run:

```powershell
git lfs install
git clone https://github.com/LH-Hansen/ALPR.git C:\ALPR
cd C:\ALPR
git lfs pull
```

### 2. Install Python dependencies

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r .\alpr_webui\requirements.txt
```

This uses CPU-based PyTorch. GPU support depends on the available hardware and PyTorch installation.

### 3. Check the model files

The API expects:

```text
C:\ALPR\alpr_webui\models\plate_detector.pt
C:\ALPR\alpr_webui\models\character_reader.pt
```

If the weights haven't been committed through Git LFS yet, copy your locally trained models into those locations. For example:

```powershell
New-Item -ItemType Directory -Force C:\ALPR\alpr_webui\models
Copy-Item C:\ALPR\runs\plate_detector\weights\best.pt C:\ALPR\alpr_webui\models\plate_detector.pt
Copy-Item C:\ALPR\runs\character_reader_dk_de\weights\best.pt C:\ALPR\alpr_webui\models\character_reader.pt
```

Your training output directory might have a suffix like `-2`, so use the actual directory printed at the end of training.

### 4. Start the web app

```powershell
cd C:\ALPR\alpr_webui
& C:\ALPR\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/**, upload an image and select **Recognize plates**.

Interactive API documentation is available at **http://127.0.0.1:8000/docs**.

## Using the API

The website isn't required. You can send an image to the same API from another frontend or program.

**Endpoint:** `POST /api/recognize`  
**Body:** `multipart/form-data` with a file field named `file`

PowerShell example:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/recognize -F "file=@C:\ALPR\test_car.jpg"
```

Example JSON (illustrative values):

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
        { "value": "A", "confidence": 0.92 },
        { "value": "B", "confidence": 0.88 }
      ]
    }
  ]
}
```

The character list is shortened above. The plate's `confidence` is the *detection confidence*, not a confidence score for the entire recognized plate number.

## Training and expanding the OCR model

The plate detector was trained using UC3M-LP images. The OCR model also started with UC3M-LP, and I've experimented with adding synthetic plate crops from Denmark and Germany.

| Training data | Purpose |
| --- | --- |
| UC3M-LP (Spain) | Original vehicle images, plate boxes, and character boxes |
| Danish synthetic OCR | Danish-style cropped plates with individual character labels |
| German synthetic OCR | German-style cropped plates with individual character labels |

The synthetic datasets use the same YOLO class numbering as the original OCR dataset. They are for **Model 2 only**; adding them does not require changing Model 1.

The training files themselves aren't stored in this repository because they can be large. To reproduce the training, start with the [UC3M-LP repository](https://github.com/ramajoballester/UC3M-LP) and its annotation conversion script, then prepare the combined OCR dataset locally.

An example fine-tuning command, assuming the combined dataset exists at `C:\ALPR\OCR_mixed_dk_de`:

```powershell
& C:\ALPR\.venv\Scripts\yolo.exe detect train model=C:/ALPR/runs/character_reader_dk-2/weights/best.pt data=C:/ALPR/OCR_mixed_dk_de/data.yaml epochs=25 imgsz=480 batch=4 device=cpu workers=0 fliplr=0 flipud=0 mosaic=0 project=C:/ALPR/runs name=character_reader_dk_de
```

Ultralytics can add numbered suffixes to run folders, so check the final output path before copying `best.pt` into the web app.

## Model files and Git LFS

I keep the production model files under `alpr_webui/models/`. Git LFS is configured to track `.pt` files in that folder, rather than putting large binary weights directly into normal Git history.

To add or update the weights:

```powershell
cd C:\ALPR
git lfs install
git lfs track "alpr_webui/models/*.pt"
git add .gitattributes alpr_webui/models/plate_detector.pt alpr_webui/models/character_reader.pt
git commit -m "Update trained ALPR models"
git push
```

Use `git lfs ls-files` to verify that both weights are tracked by LFS. Only the two models used for inference need to be published; training datasets and intermediate checkpoints can stay local.

## Current limitations

- **Country support is still being evaluated.** Adding synthetic Danish and German training images does not establish real-world recognition accuracy. More real photos and a separate test set are needed.
- **Single-row reading order:** Characters are currently sorted left to right. Double-row plates aren't handled correctly yet.
- **Image quality matters:** Angled, blurred, distant or partially obscured plates can cause mistakes.
- **Local demo:** The app runs inference on the CPU and isn't designed as a publicly exposed service yet.
- **Upload limits:** JPEG, PNG and WebP images are supported up to 10 MB and 20 megapixels. The API processes uploads in memory instead of storing them.

## What I want to improve next

- Measure **whole-plate accuracy** using real photos from each country.
- Add more representative Danish and German training data.
- Improve recognition of angled and double-row plates.
- Compare accuracy before and after OCR fine-tuning, so improvements for one country don't cause regressions for another.
- Explore ONNX and a possible C# integration.

## Credits and licensing

This project uses [UC3M-LP](https://github.com/ramajoballester/UC3M-LP), [Ultralytics YOLO](https://docs.ultralytics.com/) and [FastAPI](https://fastapi.tiangolo.com/).

The original dataset and ML tooling have their own licenses. Check those terms before redistributing training data or using the project commercially. If the app is made public, license plate information also needs appropriate privacy and security safeguards.
