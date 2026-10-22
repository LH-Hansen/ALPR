
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"C:\ALPR")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["plate", "ocr"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    if args.task == "plate":
        dataset = "LP"
        image_size = 640
        run_name = "plate_detector"
    else:
        dataset = "OCR"
        image_size = 480
        run_name = "character_reader"

    settings = {
        "data": str(
            ROOT / "UC3M-LP-yolo" / dataset / "data.yaml"
        ),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": image_size,
        "device": args.device,
        "workers": 0,
        "project": str(ROOT / "runs"),
        "name": args.name or run_name,
        "exist_ok": True,
        "plots": True
    }

    if args.task == "ocr":
        settings.update(
            fliplr=0.0,
            flipud=0.0,
            mosaic=0.0
        )

    model = YOLO("yolo11n.pt")
    model.train(**settings)

if __name__ == "__main__":
    main()
