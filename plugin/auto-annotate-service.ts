import { notification } from "antd";

export type AutoAnnotateModel = "pseudo" | "yoloe" | "dart" | string;

export type PromptsMap = Record<string, string>;

export interface AnnotateTaskRequest {
  task_id: number;
  model: AutoAnnotateModel;
  prompts?: PromptsMap | null;
}

export interface AnnotateTaskResponse {
  status: string;
  task_id: number;
  labels?: string[];
  message?: string;
  detail?: string;
}

export interface ModelsListResponse {
  models: string[];
}

export interface FetchModelsResult {
  models: string[];
  isFallback: boolean;
}

declare global {
  interface Window {
    __AUTO_ANNOTATE_API_URL__?: string;
  }
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
      return "DART";
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
  try {
    const response = await fetch(`${baseUrl}/models`, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });

    if (response.status === 404) {
      return {
        models: ["pseudo"],
        isFallback: true,
      };
    }

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = (await response.json()) as ModelsListResponse;
    if (Array.isArray(data.models) && data.models.length > 0) {
      return {
        models: data.models,
        isFallback: false,
      };
    }

    return {
      models: ["pseudo"],
      isFallback: true,
    };
  } catch {
    return {
      models: ["pseudo", "yoloe", "dart"],
      isFallback: true,
    };
  }
}

export async function runAutoAnnotation(
  taskId: number,
  model: AutoAnnotateModel = "pseudo",
  prompts: PromptsMap | null = null,
): Promise<AnnotateTaskResponse> {
  const baseUrl = getAutoAnnotateBaseUrl();

  const payload: AnnotateTaskRequest = {
    task_id: taskId,
    model,
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

export async function triggerAutoAnnotationWithNotifications(
  taskId: number,
  model: AutoAnnotateModel = "pseudo",
  prompts: PromptsMap | null = null,
): Promise<AnnotateTaskResponse | null> {
  const notificationKey = `auto-annotate-${taskId}-${Date.now()}`;
  const modelLabel = formatModelDisplayName(model);

  notification.info({
    key: notificationKey,
    message: "Авторазметка запущена",
    description: `Задача #${taskId} • Модель: ${modelLabel}. Пожалуйста, подождите...`,
    duration: 4,
  });

  try {
    const result = await runAutoAnnotation(taskId, model, prompts);
    const normalizedStatus = (result.status || "").toLowerCase();
    const isBackgroundTask = [
      "started",
      "queued",
      "accepted",
      "running",
      "processing",
    ].includes(normalizedStatus);

    const labelsInfo =
      result.labels && result.labels.length > 0
        ? ` Классы: ${result.labels.join(", ")}.`
        : "";

    if (isBackgroundTask) {
      notification.success({
        key: notificationKey,
        message: "Задача поставлена в очередь",
        description:
          result.message ||
          `Авторазметка задачи #${taskId} (${modelLabel}) выполняется в фоне. Обновите страницу или откройте задачу чуть позже.`,
        duration: 6,
      });
    } else {
      notification.success({
        key: notificationKey,
        message: "Разметка завершена",
        description: `Задача #${taskId} (${modelLabel}) успешно размечена.${labelsInfo} Обновите экран задания, чтобы увидеть рамки.`,
        duration: 6,
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
