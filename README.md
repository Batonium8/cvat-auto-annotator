# CVAT Auto-Annotator

Расширяемый сервис авторазметки для **CVAT Community** с поддержкой open-vocabulary моделей **YOLOE** и **DART (SAM 3)**, псевдоразметчика **Matryoshka**, фоновой очереди с отслеживанием прогресса и встроенного UI-плагина.

## Архитектура

```text
cvat-auto-annotator/
├── core/                  # Независимое ядро (без зависимостей от ML и CVAT SDK)
│   ├── models.py          # BBox, Polygon, ModelInfo, DetectionContext
│   ├── interfaces.py      # Протоколы ModelAdapter, Detector, TextPromptDetector, Segmenter
│   ├── registry.py        # ModelRegistry с интроспекцией возможностей и предзагрузкой
│   └── postprocess.py     # Общий per-label NMS для синонимичных промптов
├── adapters/              # Подключаемые адаптеры моделей
│   ├── matryoshka.py      # pseudo — быстрая проверка пайплайна без весов
│   ├── yoloe.py           # yoloe — Ultralytics YOLOE (open-vocabulary)
│   └── dart.py            # dart — DART / SAM 3 multi-class (open-vocabulary)
├── service/               # FastAPI-сервис и интеграция с CVAT SDK
│   ├── app.py             # REST API, фоновый запуск задач, трекинг статусов
│   ├── config.py          # Конфигурация через .env (pydantic-settings)
│   └── cvat_annotator.py  # Чтение кадров/классов и запись аннотаций в CVAT
├── plugin/                # React/TypeScript плагин для CVAT UI
├── thirdparty/            # DART (SAM 3) + патч dart-cpu.patch для работы на CPU
├── Dockerfile
├── docker-compose.yml
└── install_plugin.sh
```

**Поток данных:** браузер отправляет в сервис только `task_id`, имя модели, опциональные текстовые промпты и флаг `clear_existing`. Кадры и список классов сервис забирает напрямую из CVAT через `cvat-sdk`, прогоняет инференс в фоновом потоке и записывает рамки обратно в задачу. По умолчанию `clear_existing=false`, поэтому результаты разных моделей (например, DART и YOLOE) можно последовательно накладывать друг на друга.

---

## REST API

| Метод и путь | Назначение |
| :--- | :--- |
| `GET /health` | Проверка доступности сервиса (`{"status": "ok"}`) |
| `GET /models` | Список имён зарегистрированных моделей (`["pseudo", "yoloe", "dart"]`) |
| `GET /models/info` | Метаданные возможностей: `is_loaded`, `can_detect`, `supports_prompts`, `can_segment` |
| `POST /annotate` | Постановка задачи в фоновую очередь (`202 Accepted`; `409 Conflict`, если задача уже идёт) |
| `GET /status/{task_id}` | Текущий статус (`running` / `completed` / `failed`), прогресс по кадрам и число рамок |

Пример запроса:

```bash
curl -X POST http://localhost:8000/annotate \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": 2,
    "model": "dart",
    "prompts": {"vehicle": "car, truck, excavator"},
    "clear_existing": false
  }'
```

---

## Быстрый старт

### 1. Конфигурация и DART (SAM 3)

```bash
cp .env.example .env
# Заполните CVAT_USER, CVAT_PASS и при необходимости HF_TOKEN
```

Если клонируете DART с нуля на машине без GPU, примените патч для CPU-инференса:

```bash
git clone <https://github.com/mkturkcan/DART.git> thirdparty/dart
git -C thirdparty/dart apply ../dart-cpu.patch
```

### 2. Вариант А — локальный запуск (`venv`)

```bash
python -m venv venv
source venv/bin/activate
pip install --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) torch torchvision
pip install -r requirements.txt -r requirements-ml.txt

PYTHONPATH=".:./thirdparty/dart" uvicorn service.app:app --host 0.0.0.0 --port 8000
```

### 3. Вариант Б — запуск в Docker рядом с CVAT

Убедитесь, что стек CVAT поднят, и проверьте имя его Docker-сети (`docker network ls | grep cvat`, обычно `cvat_cvat`):

```bash
mkdir -p weights
# Опционально: скопируйте уже скачанные локальные веса YOLOE в ./weights/
cp -n yoloe-11s-seg.pt weights/ 2>/dev/null || true

docker compose up -d --build
```

Проверка работы контейнера:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/models/info
```

Для запуска на NVIDIA GPU задайте в `.env`:

```env
TORCH_INDEX_URL=[https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
DART_DEVICE=cuda
YOLOE_DEVICE=cuda
```

Затем раскомментируйте блок `deploy.resources` в `docker-compose.yml` и пересоберите контейнер (`docker compose up -d --build`).

### 4. Установка UI-плагина в CVAT

```bash
./install_plugin.sh
```

> При повторной пересборке только фронтенда используйте `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build --no-deps cvat_ui`, чтобы не пересобирать `cvat_server`.

В меню `Actions` любой задачи появятся пункты **«Псевдоразметка»** и **«Авторазметка (AI)...»** с выбором модели, настройкой текстовых промптов, переключателем очистки старой разметки и живым прогрессом в уведомлениях.

---

## Как добавить новую модель за 3 шага

Ядро использует структурную типизацию (`typing.Protocol` с `@runtime_checkable`), поэтому наследоваться от базового класса не нужно — достаточно реализовать нужный контракт.

### Шаг 1. Создайте адаптер в `adapters/<my_model>.py`

Реализуйте протокол `Detector` (или `TextPromptDetector`, если модель принимает текстовые промпты — тогда плагин CVAT автоматически покажет таблицу промптов):

```python
from collections.abc import Sequence
import threading
import PIL.Image

from core.models import BBox, DetectionContext


class MyModelAdapter:
    def __init__(self, weights: str) -> None:
        self._weights = weights
        self._model = None
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "my_model"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            # Ленивая загрузка весов

    # Опционально: добавьте set_prompts(), чтобы модель определилась как TextPromptDetector
    def set_prompts(self, prompts: Sequence[str]) -> None:
        with self._lock:
            self.load()
            # Кодирование и кэширование текстовых промптов

    def predict(
        self, image: PIL.Image.Image, context: DetectionContext
    ) -> list[BBox]:
        with self._lock:
            self.load()
            # Инференс и возврат списка BBox в пикселях кадра
            return []
```

### Шаг 2. Экспортируйте класс в `adapters/__init__.py`

```python
from adapters.my_model import MyModelAdapter
```

### Шаг 3. Зарегистрируйте адаптер в `service/app.py`

Добавьте одну строку в `_build_registry()`:

```python
registry.register(MyModelAdapter(weights=settings.my_model_weights))
```

После перезапуска модель автоматически появится в `GET /models`, `GET /models/info` и выпадающем списке плагина CVAT. Для будущих моделей сегментации в `core/interfaces.py` уже подготовлен протокол `Segmenter` и структура `Polygon`.
