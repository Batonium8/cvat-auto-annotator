from abc import ABC, abstractmethod
import PIL.Image

from core.models import BBox, DetectionContext


class BaseDetectorAdapter(ABC):
    """Абстрактный базовый класс для всех адаптеров моделей авторазметки."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Уникальный идентификатор адаптера ('pseudo', 'yoloe', 'dart')."""
        pass

    @abstractmethod
    def predict(
        self, image: PIL.Image.Image, context: DetectionContext
    ) -> list[BBox]:
        """Выполняет детекцию на одном кадре и возвращает список рамок BBox."""
        pass
