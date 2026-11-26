from collections.abc import Sequence

from core.interfaces import Detector, ModelAdapter, Segmenter, TextPromptDetector
from core.models import ModelInfo


class ModelRegistry:
    """Независимый реестр адаптеров моделей с интроспекцией возможностей."""

    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        if not isinstance(adapter, ModelAdapter):
            raise TypeError(
                f"Адаптер {type(adapter).__name__} не реализует протокол ModelAdapter"
            )
        self._adapters[adapter.name] = adapter

    def get(self, model_name: str) -> ModelAdapter:
        adapter = self._adapters.get(model_name)
        if adapter is None:
            available = ", ".join(self._adapters.keys())
            raise KeyError(
                f"Модель '{model_name}' не найдена. Доступные модели: {available}"
            )
        return adapter

    def get_detector(self, model_name: str) -> Detector:
        adapter = self.get(model_name)
        if not isinstance(adapter, Detector):
            raise TypeError(
                f"Модель '{model_name}' не поддерживает детекцию (протокол Detector)"
            )
        return adapter

    def get_segmenter(self, model_name: str) -> Segmenter:
        adapter = self.get(model_name)
        if not isinstance(adapter, Segmenter):
            raise TypeError(
                f"Модель '{model_name}' не поддерживает сегментацию (протокол Segmenter)"
            )
        return adapter

    def list_models(self) -> list[str]:
        return list(self._adapters.keys())

    def describe(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                name=adapter.name,
                is_loaded=adapter.is_loaded,
                can_detect=isinstance(adapter, Detector),
                supports_prompts=isinstance(adapter, TextPromptDetector),
                can_segment=isinstance(adapter, Segmenter),
            )
            for adapter in self._adapters.values()
        ]

    def preload(self, model_names: Sequence[str] | None = None) -> None:
        targets = (
            list(self._adapters.keys())
            if model_names is None
            else list(model_names)
        )
        for name in targets:
            self.get(name).load()
