"""Run DeepJiandu YOLO glyph-box inference for one slip image.

This script is launched as a subprocess by review_store.py so the review
server does not need ultralytics installed in its own Python environment.
It prints one JSON object to stdout and exits with code 0 on success.
"""

import argparse
import json
import os
import sys


def _read_image_size(image_path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        image.load()
        return image.size


def _sort_reading_order(boxes: list[dict]) -> list[dict]:
    """Sort boxes top-to-bottom, then left-to-right."""
    ordered = sorted(boxes, key=lambda box: (box["y"], box["x"]))
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to the source slip image")
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "GLYPH_YOLO_MODEL", "models/deepjiandu-full-v1-best.pt"
        ),
        help="YOLO weights path",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    config_dir = os.environ.get("YOLO_CONFIG_DIR")
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = config_dir

    from ultralytics import YOLO

    import torch

    device = os.environ.get("GLYPH_YOLO_DEVICE") or (
        "0" if torch.cuda.is_available() else "cpu"
    )
    model = YOLO(args.model)
    results = model.predict(
        source=args.image,
        imgsz=640,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=False,
    )
    if not results:
        print(json.dumps({"ok": True, "boxes": []}))
        return 0

    result = results[0]
    if result.boxes is None:
        print(json.dumps({"ok": True, "boxes": []}))
        return 0

    width, height = _read_image_size(args.image)
    xyxy = result.boxes.xyxy.cpu().numpy().tolist()
    confs = result.boxes.conf.cpu().numpy().tolist()
    boxes: list[dict] = []
    for box, confidence in zip(xyxy, confs):
        x1 = max(0, min(width - 1, int(round(box[0]))))
        y1 = max(0, min(height - 1, int(round(box[1]))))
        x2 = max(0, min(width - 1, int(round(box[2]))))
        y2 = max(0, min(height - 1, int(round(box[3]))))
        # Expand every box a little so the cropped glyph keeps its strokes:
        # prefer 8% of the detected size on each side, never less than 2px.
        pad_x = max(2, int(round((x2 - x1) * 0.08)))
        pad_y = max(2, int(round((y2 - y1) * 0.08)))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width - 1, x2 + pad_x)
        y2 = min(height - 1, y2 + pad_y)
        box_width = max(2, x2 - x1)
        box_height = max(2, y2 - y1)
        boxes.append(
            {
                "x": x1,
                "y": y1,
                "w": box_width,
                "h": box_height,
                "confidence": round(float(confidence), 4),
            }
        )

    print(
        json.dumps(
            {"ok": True, "boxes": _sort_reading_order(boxes)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
