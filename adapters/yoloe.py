import logging
import threading
from typing import Any

import PIL.Image
import torch

from core import BaseDetectorAdapter, BBox, DetectionContext

logger = logging.getLogger(__name__)

def _compute_iou(a: BBox, b: BBox) -> float:
    inter_x1 = max(a.xtl, b.xtl)
    inter_y1 = max(a.ytl, b.ytl)
    inter_x2 = min(a.xbr, b.xbr)
    inter_y2 = min(a.ybr, b.ybr)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0

    area_a = (a.xbr - a.xtl) * (a.ybr - a.ytl)
    area_b = (b.xbr - b.xtl) * (b.ybr - b.ytl)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def _nms_per_label(boxes: list[BBox], iou_threshold: float = 0.6) -> list[BBox]:
    """Схлопывает перекрывающиеся рамки, если на одном объекте сработало несколько синонимов."""
    ordered = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: list[BBox] = []
    for box in ordered:
        is_duplicate = any(
            existing.label == box.label and _compute_iou(existing, box) >= iou_threshold
            for existing in kept
        )
        if not is_duplicate:
            kept.append(box)
    return kept

class YoloeAdapter(BaseDetectorAdapter):
    """
    Open-vocabulary детектор на базе Ultralytics YOLOE.
    Поддерживает ленивую загрузку весов, текстовые промпты и потокобезопасный инференс.
    """

    def __init__(
        self,
        weights: str = "yoloe-11s-seg.pt",
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self._weights = weights
        self._conf = conf
        self._imgsz = imgsz
        self._requested_device = device

        self._device: str | None = None
        self._model: Any = None
        self._cached_prompts: tuple[str, ...] | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "yoloe"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from ultralytics import YOLOE

        self._device = self._requested_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "Загрузка модели YOLOE (%s) на устройство '%s'...",
            self._weights,
            self._device,
        )
        model = YOLOE(self._weights)
        model.to(self._device)
        self._model = model
        logger.info("Модель YOLOE успешно загружена.")

    def _sync_prompts(self, prompts: list[str]) -> None:
        prompts_key = tuple(prompts)
        if prompts_key == self._cached_prompts:
            return

        logger.info("Обновление текстовых промптов YOLOE: %s", prompts)
        text_pe = self._model.get_text_pe(prompts)
        self._model.set_classes(prompts, text_pe)
        self._cached_prompts = prompts_key

    def predict(
        self, image: PIL.Image.Image, context: DetectionContext
    ) -> list[BBox]:
        prompts, prompt_labels = context.expand_prompts()
        if not prompts:
            return []

        with self._lock:
            self._ensure_loaded()
            self._sync_prompts(prompts)

            results = self._model.predict(
                source=image,
                conf=self._conf,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,
            )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes_tensor = results[0].boxes
        xyxy_list = boxes_tensor.xyxy.cpu().tolist()
        cls_list = boxes_tensor.cls.cpu().tolist()
        conf_list = boxes_tensor.conf.cpu().tolist()

        max_w = float(context.width)
        max_h = float(context.height)
        detections: list[BBox] = []

        for xyxy, cls_val, conf_val in zip(xyxy_list, cls_list, conf_list):
            cls_id = int(cls_val)
            if not (0 <= cls_id < len(prompt_labels)):
                continue

            xtl = max(0.0, min(float(xyxy[0]), max_w))
            ytl = max(0.0, min(float(xyxy[1]), max_h))
            xbr = max(0.0, min(float(xyxy[2]), max_w))
            ybr = max(0.0, min(float(xyxy[3]), max_h))

            if xbr <= xtl or ybr <= ytl:
                continue

            detections.append(
                BBox(
                    label=prompt_labels[cls_id],
                    xtl=round(xtl, 2),
                    ytl=round(ytl, 2),
                    xbr=round(xbr, 2),
                    ybr=round(ybr, 2),
                    confidence=round(float(conf_val), 4),
                )
            )

        return _nms_per_label(detections)
