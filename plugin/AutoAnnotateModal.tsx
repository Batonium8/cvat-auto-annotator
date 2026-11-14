import React, { useEffect, useMemo, useState } from "react";
import ReactDOM from "react-dom";
import {
  Alert,
  Button,
  Input,
  Modal,
  Select,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
} from "antd";
import type { TableProps } from "antd";

import {
  AnnotateTaskResponse,
  AutoAnnotateModel,
  fetchAvailableModels,
  formatModelDisplayName,
  getAutoAnnotateBaseUrl,
  PromptsMap,
  triggerAutoAnnotationWithNotifications,
} from "./auto-annotate-service";

const { Text } = Typography;

export interface CvatTaskLabel {
  id?: number;
  name: string;
}

export interface AutoAnnotateModalProps {
  visible: boolean;
  taskId: number;
  taskName?: string;
  labels: CvatTaskLabel[];
  onClose: () => void;
  onSuccess?: (response: AnnotateTaskResponse) => void;
}

interface PromptRow {
  key: string;
  labelName: string;
}

export function AutoAnnotateModal(
  props: AutoAnnotateModalProps,
): React.ReactElement {
  const { visible, taskId, taskName, labels, onClose, onSuccess } = props;

  const [models, setModels] = useState<string[]>(["pseudo", "yoloe", "dart"]);
  const [selectedModel, setSelectedModel] =
    useState<AutoAnnotateModel>("yoloe");
  const [loadingModels, setLoadingModels] = useState<boolean>(false);
  const [modelsFallback, setModelsFallback] = useState<boolean>(false);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [promptsMap, setPromptsMap] = useState<PromptsMap>({});

  const labelNames = useMemo(
    () => labels.map((label) => label.name).filter((name) => Boolean(name)),
    [labels],
  );

  const buildDefaultPrompts = (names: string[]): PromptsMap => {
    const defaults: PromptsMap = {};
    names.forEach((name) => {
      defaults[name] = name;
    });
    return defaults;
  };

  useEffect(() => {
    if (!visible) {
      return;
    }

    setPromptsMap(buildDefaultPrompts(labelNames));

    let cancelled = false;
    setLoadingModels(true);

    fetchAvailableModels()
      .then((result) => {
        if (cancelled) {
          return;
        }
        setModels(result.models);
        setModelsFallback(result.isFallback);

        if (result.models.includes("yoloe")) {
          setSelectedModel("yoloe");
        } else if (result.models.includes("dart")) {
          setSelectedModel("dart");
        } else if (result.models.length > 0) {
          setSelectedModel(result.models[0]);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingModels(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [visible, labelNames]);

  const handlePromptChange = (labelName: string, value: string): void => {
    setPromptsMap((prev) => ({
      ...prev,
      [labelName]: value,
    }));
  };

  const handleResetPrompts = (): void => {
    setPromptsMap(buildDefaultPrompts(labelNames));
  };

  const handleSubmit = async (): Promise<void> => {
    setSubmitting(true);
    try {
      let finalPrompts: PromptsMap | null = null;

      if (selectedModel !== "pseudo" && labelNames.length > 0) {
        const normalized: PromptsMap = {};
        labelNames.forEach((name) => {
          const customValue = (promptsMap[name] || "").trim();
          normalized[name] = customValue || name;
        });
        finalPrompts = normalized;
      }

      const response = await triggerAutoAnnotationWithNotifications(
        taskId,
        selectedModel,
        finalPrompts,
      );

      if (response) {
        if (onSuccess) {
          onSuccess(response);
        }
        onClose();
      }
    } finally {
      setSubmitting(false);
    }
  };

  const tableData: PromptRow[] = useMemo(
    () =>
      labelNames.map((name) => ({
        key: name,
        labelName: name,
      })),
    [labelNames],
  );

  const isPseudoModel = selectedModel === "pseudo";

  const columns: TableProps<PromptRow>["columns"] = [
    {
      title: "Класс CVAT",
      dataIndex: "labelName",
      key: "labelName",
      width: "40%",
      render: (name: string) => <Tag color="blue">{name}</Tag>,
    },
    {
      title: "Текстовый запрос для модели",
      key: "prompt",
      width: "60%",
      render: (_: unknown, record: PromptRow) => (
        <Input
          value={promptsMap[record.labelName] ?? record.labelName}
          placeholder={record.labelName}
          disabled={isPseudoModel || submitting}
          onChange={(event) =>
            handlePromptChange(record.labelName, event.target.value)
          }
        />
      ),
    },
  ];

  const modalTitle = taskName
    ? `Авторазметка задачи #${taskId} (${taskName})`
    : `Авторазметка задачи #${taskId}`;

  return (
    <Modal
      title={modalTitle}
      visible={visible}
      width={680}
      destroyOnClose
      maskClosable={!submitting}
      onCancel={onClose}
      footer={[
        <Button key="cancel" disabled={submitting} onClick={onClose}>
          Отмена
        </Button>,
        <Button
          key="submit"
          type="primary"
          loading={submitting}
          disabled={labelNames.length === 0 || loadingModels}
          onClick={() => {
            void handleSubmit();
          }}
        >
          Запустить
        </Button>,
      ]}
    >
      <Space direction="vertical" size="middle" style={{ width: "100%" }}>
        <div>
          <Text strong style={{ display: "block", marginBottom: 8 }}>
            Модель
          </Text>
          <Spin spinning={loadingModels} size="small">
            <Select
              value={selectedModel}
              disabled={submitting}
              style={{ width: "100%" }}
              onChange={(value: AutoAnnotateModel) => setSelectedModel(value)}
            >
              {models.map((modelId) => (
                <Select.Option key={modelId} value={modelId}>
                  {formatModelDisplayName(modelId)}
                </Select.Option>
              ))}
            </Select>
          </Spin>
        </div>

        {modelsFallback && (
          <Alert
            type="info"
            showIcon
            message="Список моделей загружен по умолчанию"
            description={`Эндпоинт ${getAutoAnnotateBaseUrl()}/models не ответил — показаны доступные профили по умолчанию.`}
          />
        )}

        {labelNames.length === 0 ? (
          <Alert
            type="warning"
            showIcon
            message="У задачи нет классов (labels)"
            description="Добавьте хотя бы один класс в настройках задачи CVAT перед запуском авторазметки."
          />
        ) : (
          <div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 8,
              }}
            >
              <Text strong>Соответствие классов и текстовых запросов</Text>
              {!isPseudoModel && (
                <Button
                  size="small"
                  type="link"
                  disabled={submitting}
                  onClick={handleResetPrompts}
                >
                  Сбросить по умолчанию
                </Button>
              )}
            </div>

            {isPseudoModel && (
              <Alert
                type="info"
                showIcon
                style={{ marginBottom: 8 }}
                message="Режим псевдоразметки («Матрёшка»)"
                description="Промпты не используются: бэкенд сгенерирует вложенные рамки по центру кадра для каждого класса задачи."
              />
            )}

            <Table<PromptRow>
              size="small"
              pagination={false}
              scroll={{ y: 260 }}
              columns={columns}
              dataSource={tableData}
            />

            <Text
              type="secondary"
              style={{ display: "block", marginTop: 8, fontSize: 12 }}
            >
              Классы внутри задачи CVAT не переименовываются. В модель
              передаются только указанные текстовые запросы.
            </Text>
          </div>
        )}
      </Space>
    </Modal>
  );
}

export interface OpenAutoAnnotateModalParams {
  taskId: number;
  taskName?: string;
  labels: CvatTaskLabel[];
  onSuccess?: (response: AnnotateTaskResponse) => void;
}

export function openAutoAnnotateModal(
  params: OpenAutoAnnotateModalParams,
): void {
  const container = document.createElement("div");
  document.body.appendChild(container);

  const destroy = (): void => {
    ReactDOM.unmountComponentAtNode(container);
    if (container.parentNode) {
      container.parentNode.removeChild(container);
    }
  };

  ReactDOM.render(
    <AutoAnnotateModal
      visible
      taskId={params.taskId}
      taskName={params.taskName}
      labels={params.labels}
      onClose={destroy}
      onSuccess={params.onSuccess}
    />,
    container,
  );
}
