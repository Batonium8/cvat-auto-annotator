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

    def expand_prompts(self) -> tuple[list[str], list[str]]:
        """
        Разбивает промпты через запятую в плоский список запросов для модели
        и параллельный список исходных классов CVAT.
        Пример: {'vehicle': 'car, truck, excavator'} ->
            prompts=['car', 'truck', 'excavator']
            prompt_labels=['vehicle', 'vehicle', 'vehicle']
        """
        expanded_prompts: list[str] = []
        prompt_labels: list[str] = []
        seen: set[str] = set()

        for label in self.labels:
            raw = self.get_prompt(label)
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            if not parts:
                parts = [label]

            for part in parts:
                key = part.lower()
                if key in seen:
                    continue
                seen.add(key)
                expanded_prompts.append(part)
                prompt_labels.append(label)

        return expanded_prompts, prompt_labels
