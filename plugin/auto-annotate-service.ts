import { notification } from "antd";

export type AutoAnnotateModel = "pseudo" | "yoloe" | "dart" | string;

export type PromptsMap = Record<string, string>;

export interface ModelInfoItem {
  name: string;
  is_loaded: boolean;
  can_detect: boolean;
  supports_prompts: boolean;
  can_segment: boolean;
}

export interface ModelsInfoResponse {
  models: ModelInfoItem[];
}

export interface ModelsListResponse {
  models: string[];
}

export interface FetchModelsResult {
  models: string[];
  modelsInfo: ModelInfoItem[];
  isFallback: boolean;
}

export interface AnnotateTaskRequest {
  task_id: number;
  model: AutoAnnotateModel;
  prompts?: PromptsMap | null;
  clear_existing?: boolean;
}

export interface AnnotateTaskResponse {
  status: string;
  task_id: number;
  model?: string;
  frames_processed?: number | null;
  shapes_created?: number | null;
  labels?: string[];
  message?: string;
  detail?: string;
}

export interface TaskStatusResponse {
  task_id: number;
  status: "running" | "completed" | "failed" | string;
  model: string;
  frames_processed: number;
  total_frames: number;
  shapes_created: number;
  labels: string[];
  message: string;
  error: string | null;
}

declare global {
  interface Window {
    __AUTO_ANNOTATE_API_URL__?: string;
  }
}

const POLL_INTERVAL_MS = 2000;
const MAX_CONSECUTIVE_POLL_ERRORS = 5;
const MAX_POLL_DURATION_MS = 30 * 60 * 1000; // 30 минут с запасом под CPU-инференс

const activePollTimers = new Map<number, number>();

function stopTaskPolling(taskId: number): void {
  const timerId = activePollTimers.get(taskId);
  if (timerId !== undefined) {
    window.clearTimeout(timerId);
    activePollTimers.delete(taskId);
  }
}

function buildDefaultModelInfo(name: string): ModelInfoItem {
  return {
    name,
    is_loaded: name === "pseudo",
    can_detect: true,
    supports_prompts: name !== "pseudo",
    can_segment: false,
  };
}

export function getAutoAnnotateBaseUrl(): string {
  if (typeof window !== "undefined" && window.__AUTO_ANNOTATE_API_URL__) {
    return window.__AUTO_ANNOTATE_API_URL__.replace(/\/+$/, "");
  }
  return "http://localhost:8000";
}

export function formatModelDisplayName(model: string): string {
  switch (model) {
    case "pseudo":
      return "Псевдоразметка (Матрёшка)";
    case "yoloe":
      return "YOLOE";
    case "dart":
      return "DART (SAM 3)";
    default:
      return model.toUpperCase();
  }
}

export async function checkServiceHealth(): Promise<boolean> {
  const baseUrl = getAutoAnnotateBaseUrl();
  try {
    const response = await fetch(`${baseUrl}/health`, { method: "GET" });
    if (!response.ok) {
      return false;
    }
    const data = (await response.json()) as { status?: string };
    return data.status === "ok";
  } catch {
    return false;
  }
}

export async function fetchAvailableModels(): Promise<FetchModelsResult> {
  const baseUrl = getAutoAnnotateBaseUrl();

  // 1. Пробуем новый эндпоинт /models/info с метаданными возможностей
  try {
    const infoResponse = await fetch(`${baseUrl}/models/info`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });

    if (infoResponse.ok) {
      const data = (await infoResponse.json()) as ModelsInfoResponse;
      if (Array.isArray(data.models) && data.models.length > 0) {
        const detectors = data.models.filter((item) => item.can_detect);
        if (detectors.length > 0) {
          return {
            models: detectors.map((item) => item.name),
            modelsInfo: detectors,
            isFallback: false,
          };
        }
      }
    }
  } catch {
    // Продолжаем с фолбэком на /models
  }

  // 2. Фолбэк на базовый эндпоинт /models
  try {
    const response = await fetch(`${baseUrl}/models`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });

    if (response.ok) {
      const data = (await response.json()) as ModelsListResponse;
      if (Array.isArray(data.models) && data.models.length > 0) {
        return {
          models: data.models,
          modelsInfo: data.models.map(buildDefaultModelInfo),
          isFallback: false,
        };
      }
    }
  } catch {
    // Переходим к локальному списку по умолчанию
  }

  const fallbackModels = ["pseudo", "yoloe", "dart"];
  return {
    models: fallbackModels,
    modelsInfo: fallbackModels.map(buildDefaultModelInfo),
    isFallback: true,
  };
}

export async function fetchTaskStatus(
  taskId: number,
): Promise<TaskStatusResponse> {
  const baseUrl = getAutoAnnotateBaseUrl();
  const response = await fetch(`${baseUrl}/status/${taskId}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}`);
  }

  return (await response.json()) as TaskStatusResponse;
}

export async function runAutoAnnotation(
  taskId: number,
  model: AutoAnnotateModel = "pseudo",
  prompts: PromptsMap | null = null,
  clearExisting = false,
): Promise<AnnotateTaskResponse> {
  const baseUrl = getAutoAnnotateBaseUrl();

  const payload: AnnotateTaskRequest = {
    task_id: taskId,
    model,
    clear_existing: clearExisting,
  };

  if (prompts !== null) {
    payload.prompts = prompts;
  }

  const response = await fetch(`${baseUrl}/annotate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(payload),
  });

  let data: AnnotateTaskResponse | null = null;
  try {
    data = (await response.json()) as AnnotateTaskResponse;
  } catch {
    data = null;
  }

  if (!response.ok) {
    const errorDetail =
      data?.detail ||
      data?.message ||
      `HTTP ${response.status} ${response.statusText}`;
    throw new Error(errorDetail);
  }

  return data || { status: "success", task_id: taskId };
}

function startTaskStatusPolling(
  taskId: number,
  modelLabel: string,
  notificationKey: string,
): void {
  stopTaskPolling(taskId);

  const startedAt = Date.now();
  let consecutiveErrors = 0;

  const scheduleNext = (): void => {
    const timerId = window.setTimeout(() => {
      void pollOnce();
    }, POLL_INTERVAL_MS);
    activePollTimers.set(taskId, timerId);
  };

  const pollOnce = async (): Promise<void> => {
    if (Date.now() - startedAt > MAX_POLL_DURATION_MS) {
      stopTaskPolling(taskId);
      notification.warning({
        key: notificationKey,
        message: `Задача #${taskId} (${modelLabel}) всё ещё выполняется`,
        description:
          "Опрос статуса остановлен по таймауту. Проверьте задачу позже и обновите страницу (F5).",
        duration: 10,
      });
      return;
    }

    try {
      const statusData = await fetchTaskStatus(taskId);
      consecutiveErrors = 0;

      const normalized = (statusData.status || "").toLowerCase();

      if (normalized === "running") {
        const progressLine =
          statusData.total_frames > 0
            ? `Кадры: ${statusData.frames_processed}/${statusData.total_frames} • Рамок: ${statusData.shapes_created}`
            : statusData.message || "Подготовка модели и загрузка весов...";

        const loadingHint =
          statusData.frames_processed === 0
            ? " (при первом запуске модель загружает веса)"
            : "";

        notification.info({
          key: notificationKey,
          message: `Авторазметка задачи #${taskId} (${modelLabel})`,
          description: `${progressLine}${loadingHint}`,
          duration: 0,
        });

        scheduleNext();
        return;
      }

      stopTaskPolling(taskId);

      if (normalized === "completed") {
        notification.success({
          key: notificationKey,
          message: `Разметка завершена (${modelLabel})`,
          description: `Задача #${taskId}: обработано кадров ${statusData.frames_processed}, создано рамок ${statusData.shapes_created}. Обновите страницу (F5), чтобы увидеть аннотации.`,
          duration: 10,
        });
        return;
      }

      if (normalized === "failed") {
        notification.error({
          key: notificationKey,
          message: `Ошибка авторазметки (${modelLabel})`,
          description: `Задача #${taskId}: ${statusData.error || statusData.message || "неизвестная ошибка"}`,
          duration: 10,
        });
        return;
      }

      // Неизвестный промежуточный статус — продолжаем опрос
      scheduleNext();
    } catch (error) {
      consecutiveErrors += 1;
      if (consecutiveErrors >= MAX_CONSECUTIVE_POLL_ERRORS) {
        stopTaskPolling(taskId);
        const reason = error instanceof Error ? error.message : String(error);
        notification.warning({
          key: notificationKey,
          message: `Потеряна связь со статусом задачи #${taskId}`,
          description: `Не удалось получить статус (${reason}). Разметка могла завершиться на сервере — обновите задачу чуть позже.`,
          duration: 10,
        });
        return;
      }
      scheduleNext();
    }
  };

  scheduleNext();
}

export async function triggerAutoAnnotationWithNotifications(
  taskId: number,
  model: AutoAnnotateModel = "pseudo",
  prompts: PromptsMap | null = null,
  clearExisting = false,
): Promise<AnnotateTaskResponse | null> {
  const notificationKey = `auto-annotate-${taskId}`;
  const modelLabel = formatModelDisplayName(model);

  notification.info({
    key: notificationKey,
    message: `Запуск авторазметки (${modelLabel})`,
    description: `Отправка задачи #${taskId} в сервис...`,
    duration: 0,
  });

  try {
    const result = await runAutoAnnotation(
      taskId,
      model,
      prompts,
      clearExisting,
    );
    const normalizedStatus = (result.status || "").toLowerCase();
    const isBackgroundTask = [
      "started",
      "queued",
      "accepted",
      "running",
      "processing",
    ].includes(normalizedStatus);

    if (isBackgroundTask) {
      notification.info({
        key: notificationKey,
        message: `Авторазметка задачи #${taskId} (${modelLabel})`,
        description:
          result.message || "Задача поставлена в очередь, ожидаем прогресс...",
        duration: 0,
      });
      startTaskStatusPolling(taskId, modelLabel, notificationKey);
    } else {
      const labelsInfo =
        result.labels && result.labels.length > 0
          ? ` Классы: ${result.labels.join(", ")}.`
          : "";
      notification.success({
        key: notificationKey,
        message: `Разметка завершена (${modelLabel})`,
        description: `Задача #${taskId} успешно размечена.${labelsInfo} Обновите страницу (F5), чтобы увидеть рамки.`,
        duration: 8,
      });
    }

    return result;
  } catch (error) {
    const baseUrl = getAutoAnnotateBaseUrl();
    const reason = error instanceof Error ? error.message : String(error);
    const isNetworkError =
      reason.includes("Failed to fetch") || reason.includes("NetworkError");

    notification.error({
      key: notificationKey,
      message: "Ошибка авторазметки",
      description: isNetworkError
        ? `Не удалось подключиться к сервису авторазметки (${baseUrl}). Проверьте, запущен ли FastAPI-сервис.`
        : `Задача #${taskId}: ${reason}`,
      duration: 8,
    });

    return null;
  }
}
