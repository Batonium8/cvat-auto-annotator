from collections.abc import Sequence
import logging
import threading
from typing import Any

import PIL.Image

from core.models import BBox, DetectionContext
from core.postprocess import nms_per_label

logger = logging.getLogger(__name__)


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


class DartAdapter:
    """
    Open-vocabulary детектор на базе DART (SAM3 multi-class).
    Реализует протокол TextPromptDetector с ленивой загрузкой и кэшем промптов.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        conf: float = 0.3,
        nms: float = 0.5,
        imgsz: int = 1008,
        device: str = "cpu",
    ) -> None:
        self._checkpoint_path = checkpoint_path or None
        self._conf = conf
        self._nms = nms
        self._imgsz = imgsz
        self._device = device

        self._predictor: Any = None
        self._active_prompts: tuple[str, ...] | None = None
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "dart"

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def _resolve_device(self, torch_module: Any) -> str:
        if not self._device or self._device == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        return self._device

    def load(self) -> None:
        with self._lock:
            if self._predictor is not None:
                return

            import torch
            from sam3.model.sam3_multiclass import Sam3MultiClassPredictor
            from sam3.model_builder import build_sam3_image_model

            resolved_device = self._resolve_device(torch)
            logger.info(
                "Загрузка модели DART (checkpoint=%s, imgsz=%d) на устройство '%s'...",
                self._checkpoint_path or "HF:facebook/sam3",
                self._imgsz,
                resolved_device,
            )

            model = build_sam3_image_model(
                device=resolved_device,
                checkpoint_path=self._checkpoint_path,
                eval_mode=True,
            )

            if self._imgsz != 1008:
                pos_enc = model.backbone.vision_backbone.position_encoding
                pos_enc.precompute_for_resolution(self._imgsz)

            self._predictor = Sam3MultiClassPredictor(
                model,
                device=resolved_device,
                resolution=self._imgsz,
                detection_only=True,
            )
            logger.info("Модель DART успешно загружена.")

    def set_prompts(self, prompts: Sequence[str]) -> None:
        prompt_list = list(prompts)
        if not prompt_list:
            return

        key = tuple(prompt_list)
        import torch

        with self._lock, torch.inference_mode():
            self.load()
            if self._active_prompts == key:
                return

            logger.info("Обновление текстовых промптов DART: %s", prompt_list)
            self._predictor.set_classes(prompt_list)
            self._active_prompts = key

    def predict(
        self, image: PIL.Image.Image, context: DetectionContext
    ) -> list[BBox]:
        prompts, prompt_labels = context.expand_prompts()
        if not prompts:
            return []

        prompt_to_label = {
            prompt.lower(): label for prompt, label in zip(prompts, prompt_labels)
        }

        import torch

        with self._lock, torch.inference_mode():
            self.load()
            self.set_prompts(prompts)

            state = self._predictor.set_image(image)
            results = self._predictor.predict(
                state,
                confidence_threshold=self._conf,
                nms_threshold=self._nms,
            )

        if not results:
            return []

        scores_list = _to_list(results.get("scores", []))
        if not scores_list:
            return []

        boxes_list = _to_list(results.get("boxes", []))
        class_names = list(results.get("class_names", []))

        max_w = float(context.width)
        max_h = float(context.height)
        detections: list[BBox] = []

        for xyxy, cls_name, conf_val in zip(boxes_list, class_names, scores_list):
            cvat_label = prompt_to_label.get(str(cls_name).lower())
            if cvat_label is None:
                continue

            xtl = max(0.0, min(float(xyxy[0]), max_w))
            ytl = max(0.0, min(float(xyxy[1]), max_h))
            xbr = max(0.0, min(float(xyxy[2]), max_w))
            ybr = max(0.0, min(float(xyxy[3]), max_h))

            if xbr <= xtl or ybr <= ytl:
                continue

            detections.append(
                BBox(
                    label=cvat_label,
                    xtl=round(xtl, 2),
                    ytl=round(ytl, 2),
                    xbr=round(xbr, 2),
                    ybr=round(ybr, 2),
                    confidence=round(float(conf_val), 4),
                )
            )

        return nms_per_label(detections)
