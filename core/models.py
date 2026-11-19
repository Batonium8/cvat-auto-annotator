from dataclasses import dataclass, field


@dataclass(frozen=True)
class BBox:
    """Единый формат прямоугольной аннотации в пикселях (xtl, ytl, xbr, ybr)."""
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    confidence: float = 1.0


@dataclass
class DetectionContext:
    """Метаданные кадра и маппинг классов CVAT в текстовые промпты."""
    width: int
    height: int
    labels: list[str]
    prompts: dict[str, str] = field(default_factory=dict)

    def get_prompt(self, label: str) -> str:
        """Возвращает кастомный промпт (Этап 3) или имя класса по умолчанию (Этапы 1-2)."""
        custom = self.prompts.get(label, "").strip()
        return custom if custom else label
