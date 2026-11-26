from core.models import BBox


def compute_iou(a: BBox, b: BBox) -> float:
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


def nms_per_label(boxes: list[BBox], iou_threshold: float = 0.6) -> list[BBox]:
    """Убирает перекрывающиеся дубли в рамках одного класса CVAT."""
    ordered = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: list[BBox] = []
    for box in ordered:
        is_duplicate = any(
            existing.label == box.label and compute_iou(existing, box) >= iou_threshold
            for existing in kept
        )
        if not is_duplicate:
            kept.append(box)
    return kept
