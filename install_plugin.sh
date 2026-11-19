#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CVAT_DIR="${1:-$SCRIPT_DIR/../cvat}"
PLUGIN_DIR="$SCRIPT_DIR/plugin"

if [ ! -d "$CVAT_DIR" ]; then
  echo "Ошибка: директория CVAT не найдена по пути: $CVAT_DIR" >&2
  echo "Использование: $0 [путь_к_репозиторию_cvat]" >&2
  exit 1
fi

CVAT_DIR="$(cd "$CVAT_DIR" && pwd)"

if [ ! -f "$CVAT_DIR/docker-compose.yml" ] \vert{}\vert{} [ ! -f "$CVAT_DIR/docker-compose.dev.yml" ]; then
  echo "Ошибка: в $CVAT_DIR отсутствуют docker-compose.yml или docker-compose.dev.yml" >&2
  exit 1
fi

# Ищем именно каталог с существующим actions-menu.tsx, а не просто наличие папки
if [ -f "$CVAT_DIR/cvat-ui/src/components/tasks-page/actions-menu.tsx" ]; then
  TARGET_DIR="$CVAT_DIR/cvat-ui/src/components/tasks-page"
elif [ -f "$CVAT_DIR/cvat-ui/src/components/actions-menu/actions-menu.tsx" ]; then
  TARGET_DIR="$CVAT_DIR/cvat-ui/src/components/actions-menu"
else
  echo "Ошибка: не найден actions-menu.tsx в $CVAT_DIR/cvat-ui/src/components/" >&2
  exit 1
fi

echo "[1/2] Копирование файлов плагина в $TARGET_DIR..."
cp -v "$PLUGIN_DIR/auto-annotate-service.ts" "$TARGET_DIR/"
cp -v "$PLUGIN_DIR/AutoAnnotateModal.tsx" "$TARGET_DIR/"
cp -v "$PLUGIN_DIR/actions-menu/actions-menu.tsx" "$TARGET_DIR/"

echo "[2/2] Пересборка контейнера cvat_ui (без перезапуска БД и остальных сервисов)..."
cd "$CVAT_DIR"
export CVAT_HOST="${CVAT_HOST:-localhost}"
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build --no-deps cvat_ui

echo "Готово! Плагин обновлён на http://${CVAT_HOST}:8080 (нажмите Ctrl+Shift+R в браузере)."
