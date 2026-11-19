from adapters.matryoshka import MatryoshkaAdapter
from core import BaseDetectorAdapter

_REGISTRY: dict[str, BaseDetectorAdapter] = {
    "pseudo": MatryoshkaAdapter(),
}


def get_adapter(model_name: str) -> BaseDetectorAdapter:
    adapter = _REGISTRY.get(model_name)
    if adapter is None:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(
            f"Модель '{model_name}' не найдена. Доступные модели: {available}"
        )
    return adapter


def list_available_models() -> list[str]:
    return list(_REGISTRY.keys())
