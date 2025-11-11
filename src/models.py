
import torch
from ultralytics import YOLO

from src.config import DEVICE


def load_yolo_model(weights: str = "yolo11n.pt") -> YOLO:

    print(f"[models] Loading YOLO model '{weights}' on {DEVICE} ...")
    model = YOLO(weights)
    model.to(DEVICE)
    return model


def load_faster_rcnn():

    print(f"[models] Loading Faster R-CNN (ResNet50-FPN, COCO-pretrained) on {DEVICE} ...")
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        try:
            # newer torchvision API
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn(weights=weights)
        except Exception:
            # fallback for older versions
            model = fasterrcnn_resnet50_fpn(pretrained=True)
    except ImportError as e:
        raise RuntimeError(
            "Faster R-CNN (fasterrcnn_resnet50_fpn) is not available in your torchvision installation.\n"
            "Please install torchvision with detection models "
            "(e.g. `pip install --upgrade torchvision`)."
        ) from e

    model.to(DEVICE)
    model.eval()
    return model


def load_detr():

    print(f"[models] Loading DETR (Hugging Face, facebook/detr-resnet-50) on {DEVICE} ...")

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as e:
        raise RuntimeError(
            "The 'transformers' package is not installed.\n"
            "Please install it with:\n"
            "  pip install transformers\n"
        ) from e

    if DEVICE.type == "cuda":
        device_index = DEVICE.index if DEVICE.index is not None else 0
        device = device_index
    else:
        device = -1

    det_pipe = hf_pipeline(
        task="object-detection",
        model="facebook/detr-resnet-50",
        device=device,
    )

    print(f"[models] DETR pipeline created on device {device}.")
    return det_pipe
