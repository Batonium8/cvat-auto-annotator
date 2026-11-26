from collections.abc import Sequence
import logging
import threading
from typing import Any

import PIL.Image

from core.models import BBox, DetectionContext
from core.postprocess import nms_per_label

logger = logging.getLogger(__name__)


class YoloeAdapter:
    """
    Open-vocabulary детектор на базе Ultralytics YOLOE.
    Реализует протокол TextPromptDetector с ленивой загрузкой и кэшем промптов.
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
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "yoloe"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve_device(self, torch_module: Any) -> str:
        if not self._requested_device or self._requested_device == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        return self._requested_device

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return

            import torch
            from ultralytics import YOLOE

            self._device = self._resolve_device(torch)
            logger.info(
                "Загрузка модели YOLOE (%s) на устройство '%s'...",
                self._weights,
                self._device,
            )
            model = YOLOE(self._weights)
            model.to(self._device)
            self._model = model
            logger.info("Модель YOLOE успешно загружена.")

    def set_prompts(self, prompts: Sequence[str]) -> None:
        prompt_list = list(prompts)
        if not prompt_list:
            return

        prompts_key = tuple(prompt_list)
        with self._lock:
            self.load()
            if prompts_key == self._cached_prompts:
                return

            logger.info("Обновление текстовых промптов YOLOE: %s", prompt_list)
            text_pe = self._model.get_text_pe(prompt_list)
            self._model.set_classes(prompt_list, text_pe)
            self._cached_prompts = prompts_key

    def predict(
        self, image: PIL.Image.Image, context: DetectionContext
    ) -> list[BBox]:
        prompts, prompt_labels = context.expand_prompts()
        if not prompts:
            return []

        with self._lock:
            self.load()
            self.set_prompts(prompts)

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

        return nms_per_label(detections)
