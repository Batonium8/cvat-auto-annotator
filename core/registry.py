from core.interfaces import BaseDetectorAdapter


class ModelRegistry:
    """Независимый реестр адаптеров моделей."""

    def __init__(self) -> None:
        self._adapters: dict[str, BaseDetectorAdapter] = {}

    def register(self, adapter: BaseDetectorAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def get(self, model_name: str) -> BaseDetectorAdapter:
        adapter = self._adapters.get(model_name)
        if adapter is None:
            available = ", ".join(self._adapters.keys())
            raise KeyError(
                f"Модель '{model_name}' не найдена. Доступные модели: {available}"
            )
        return adapter

    def list_models(self) -> list[str]:
        return list(self._adapters.keys())
