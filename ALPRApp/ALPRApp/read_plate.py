

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(r"C:\ALPR")

detector = YOLO(
    str(ROOT / "runs/plate_detector/weights/best.pt")
)
reader = YOLO(
    str(ROOT / "runs/character_reader/weights/best.pt")
)


def recognize(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    height, width = image.shape[:2]

    result = detector.predict(
        image,
        imgsz=640,
        conf=0.25,
        device="cpu",
        verbose=False
    )[0]

    if len(result.boxes) == 0:
        print("No license plates detected.")
        return

    for index, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        # Add a small margin around the plate.
        x1 = max(0, x1 - 4)
        y1 = max(0, y1 - 4)
        x2 = min(width, x2 + 4)
        y2 = min(height, y2 + 4)

        plate = image[y1:y2, x1:x2]

        if plate.size == 0:
            continue

        characters = reader.predict(
            plate,
            imgsz=480,
            conf=0.25,
            device="cpu",
            verbose=False
        )[0]

        detected = []

        for char in characters.boxes:
            class_id = int(char.cls[0].item())
            character = reader.names[class_id]

            left = float(char.xyxy[0][0].item())
            detected.append((left, character))

        detected.sort(key=lambda item: item[0])
        plate_text = "".join(
            char for _, char in detected
        )

        print(f"Plate {index + 1}: {plate_text or 'Unreadable'}")

        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        cv2.putText(
            image,
            plate_text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    output = image_path.with_name(
        image_path.stem + "_result" + image_path.suffix
    )

    if not cv2.imwrite(str(output), image):
        raise RuntimeError("Could not save output image")

    print(f"Saved: {output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python read_plate.py <image_path>"
        )

    recognize(Path(sys.argv[1]))
