# Плагин авторазметки для CVAT UI (`cvat-auto-annotator/plugin`)

Клиентская часть расширяемой системы авторазметки данных для **CVAT Community** (`cvat-ui`).

## Что реализовано

1. **Этап 1 («Псевдоразметка»):**
   - Пункт меню **«Псевдоразметка»** в меню `Actions` задачи.
   - Отправляет `POST http://localhost:8000/annotate` с JSON `{"task_id": taskId, "model": "pseudo"}`.
   - Выводит неблокирующие уведомления через `antd` (`notification`) о запуске, успехе и ошибках.

2. **Этапы 2 и 3 («Авторазметка (AI)...»):**
   - Пункт меню **«Авторазметка (AI)...»** в меню `Actions` задачи.
   - Открывает модальное окно `AutoAnnotateModal`:
     - загрузка доступных моделей с `GET http://localhost:8000/models` (`YOLOE`, `DART`, `pseudo`);
     - таблица соответствия **«Класс CVAT» -> «Текстовый запрос для модели»** (по умолчанию заполнена именами классов из задачи CVAT, без переименования самих классов в CVAT);
     - отправка `POST http://localhost:8000/annotate` с `{ task_id, model, prompts }`.
   - Изображения и метаданные классов на бэкенд из браузера **не передаются**.

---

## Состав файлов плагина в репозитории

```text
plugin/
├── README.md                      # Документация плагина
├── auto-annotate-service.ts       # HTTP-клиент к FastAPI (порт 8000) и уведомления
├── AutoAnnotateModal.tsx          # Модальное окно выбора модели и настройки промптов
└── actions-menu/
    └── actions-menu.tsx           # Компонент меню действий задачи с добавленными пунктами
```

---

## Установка плагина в контейнер CVAT

В свежих версиях CVAT компонент меню действий задачи находится в `cvat-ui/src/components/tasks-page/`, а в версиях CVAT 2.15 — в `cvat-ui/src/components/actions-menu/`.

Скрипт `install_plugin.sh` (запускается из корня `cvat-auto-annotator`):

```bash
#!/usr/bin/env bash
set -e

CVAT_DIR="../cvat"
if [ -d "$CVAT_DIR/cvat-ui/src/components/tasks-page" ]; then
    TARGET_DIR="$CVAT_DIR/cvat-ui/src/components/tasks-page"
else
    TARGET_DIR="$CVAT_DIR/cvat-ui/src/components/actions-menu"
fi

echo "[1/2] Копирование файлов плагина в $TARGET_DIR..."
cp -v plugin/auto-annotate-service.ts "$TARGET_DIR/"
cp -v plugin/AutoAnnotateModal.tsx "$TARGET_DIR/"
cp -v plugin/actions-menu/actions-menu.tsx "$TARGET_DIR/"

echo "[2/2] Пересборка контейнера cvat_ui..."
cd "$CVAT_DIR"
export CVAT_HOST=localhost
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build cvat_ui
echo "Готово! Плагин обновлен на http://localhost:8080"
```
