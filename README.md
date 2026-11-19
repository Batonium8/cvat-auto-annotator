# CVAT Auto-Annotator

Расширяемая система авторазметки для локального **CVAT Community**:
- **Бэкенд (`service/`, `adapters/`)** — сервис на FastAPI + `cvat-sdk` (порт `8000`) с модульными адаптерами моделей.
- **Фронтенд (`plugin/`)** — React-плагин для `cvat-ui` (порт `8080`), добавляющий в меню `Actions` задачи пункты **«Псевдоразметка»** и **«Авторазметка (AI)...»**.

## Архитектура

1. Пользователь запускает разметку из меню `Actions` задачи в CVAT (`http://localhost:8080`).
2. Плагин отправляет `POST http://localhost:8000/annotate` с `task_id`, именем модели и опциональной картой промптов `prompts` (сами изображения из браузера не передаются).
3. FastAPI-сервис через `cvat-sdk` забирает кадры и классы задачи из CVAT, прогоняет выбранный адаптер и записывает bounding boxes обратно в задачу.

## Структура репозитория

```text
cvat-auto-annotator/
├── adapters/             # Реестр и адаптеры моделей (pseudo и др.)
├── service/              # Конфигурация, клиент CVAT (cvat-sdk) и FastAPI API
│   ├── app.py
│   ├── config.py
│   └── cvat_annotator.py
├── plugin/               # Исходники React-плагина для cvat-ui
├── install_plugin.sh     # Скрипт установки плагина и пересборки cvat_ui
└── README.md
```

## Требования

- Docker, Docker Compose и плагин `docker-buildx` (для стабильной сборки `cvat_ui`).
- Python 3.10+.
- Клонированный рядом репозиторий CVAT (по умолчанию ожидается `../cvat`).

## Быстрый старт

### 1. Запуск CVAT

```bash
cd ../cvat
export CVAT_HOST=localhost
docker compose up -d
```

Веб-интерфейс доступен на `http://localhost:8080`.

### 2. Настройка и запуск FastAPI-бэкенда

Создайте файл `.env` в корне `cvat-auto-annotator` с учётными данными локального CVAT:

```env
CVAT_HOST=http://localhost:8080
CVAT_USER=admin
CVAT_PASS=your_password
```

Установите зависимости и запустите сервер:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn service.app:app --reload --port 8000
```

Swagger UI доступен на `http://localhost:8000/docs`.

### 3. Установка плагина в CVAT UI

Из корня `cvat-auto-annotator`:

```bash
./install_plugin.sh
# или с явным указанием пути к CVAT:
./install_plugin.sh /path/to/cvat
```

После пересборки `cvat_ui` откройте `http://localhost:8080/tasks` и выполните жёсткое обновление страницы (`Ctrl + Shift + R`).

## Эндпоинты API

| Метод | Путь | Описание |
| :--- | :--- | :--- |
| `GET` | `/health` | Проверка доступности сервиса (`{"status": "ok"}`) |
| `GET` | `/models` | Список зарегистрированных адаптеров (`{"models": ["pseudo"]}`) |
| `POST` | `/annotate` | Запуск авторазметки задачи по `task_id`, `model`, `prompts`, `clear_existing` |
