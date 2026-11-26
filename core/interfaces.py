from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import PIL.Image

from core.models import BBox, DetectionContext, Polygon


@runtime_checkable
class ModelAdapter(Protocol):
    """Базовый контракт идентификации и жизненного цикла модели."""

    @property
    def name(self) -> str:
        """Уникальный идентификатор адаптера ('pseudo', 'yoloe', 'dart')."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Показывает, загружены ли веса модели в память."""
        ...

    def load(self) -> None:
        """Идемпотентная и потокобезопасная загрузка весов."""
        ...


@runtime_checkable
class Detector(ModelAdapter, Protocol):
    """Детектор прямоугольных рамок (bbox)."""

    def predict(
        self, image: PIL.Image.Image, context: DetectionContext
    ) -> list[BBox]:
        """Выполняет детекцию на одном кадре и возвращает список рамок BBox."""
        ...


@runtime_checkable
class TextPromptDetector(Detector, Protocol):
    """Open-vocabulary детектор, управляемый текстовыми промптами."""

    def set_prompts(self, prompts: Sequence[str]) -> None:
        """Кэширует и кодирует текстовые эмбеддинги для заданных промптов."""
        ...


@runtime_checkable
class Segmenter(ModelAdapter, Protocol):
    """Задел под модели сегментации (маски/полигоны)."""

    def predict_masks(
        self, image: PIL.Image.Image, context: DetectionContext
    ) -> list[Polygon]:
        """Выполняет сегментацию на одном кадре и возвращает список полигонов."""
        ...
