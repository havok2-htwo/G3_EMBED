export type AdminOption = {
  label: string;
  value: string;
};

export type AdminKeyMetadata = {
  id: string;
  label: string;
  created_at: string | null;
  last_used_at: string | null;
};

export type AdminSettings = {
  default_model: string;
  execution_target: string;
  backend_override: string;
  model_cache_path: string;
  batching_enabled: boolean;
  batch_wait_time_ms: number;
  batch_max_texts: number;
  batch_max_chars: number;
  precision: string;
  compile_mode: string;
  warmup_on_load: boolean;
  huggingface_token: string;
};

export type ManagedModel = {
  id: string;
  label: string;
  family: string;
  tier: string;
  status: string;
  local_path: string | null;
  cache_path: string | null;
  storage_root: string;
  approx_size_gb: number | null;
  size_on_disk_gb: number | null;
  dimensions: number | null;
  recommended_backend: string;
  notes: string;
  error: string | null;
  updated_at: string | null;
};

export type RuntimeInfo = {
  loaded_models: Array<Record<string, unknown>>;
  last_runtime_error: string | null;
  resolved_runtime: {
    backend: string;
    device: string;
    device_label: string;
    source: string;
    detail: string;
  };
  system: {
    platform: string;
    machine: string;
    processor: string;
    nvidia_gpus: string[];
    windows_video_devices: string[];
    windows_npu_devices: string[];
    openvino_devices: string[];
    detected_accelerator_hardware: string[];
    torch: Record<string, unknown>;
  };
};

export type SettingsResponse = {
  settings: AdminSettings;
  options: {
    models: AdminOption[];
    execution_targets: AdminOption[];
    backends: AdminOption[];
    precisions: AdminOption[];
    compile_modes: AdminOption[];
  };
  models: ManagedModel[];
  runtime: RuntimeInfo;
};

export type StatsResponse = {
  summary: {
    total_requests: number;
    successful_requests: number;
    avg_total_duration_ms: number | null;
    avg_encode_duration_ms: number | null;
    total_texts: number;
  };
  history: Array<Record<string, unknown>>;
  recent_batches: Array<Record<string, unknown>>;
};

export type QueueResponse = {
  worker_running: boolean;
  queue_size?: number;
  active_batch_id?: string | null;
  active_batch_size?: number;
  active_batch_chars?: number;
  active_batch_started_at?: string | null;
  last_batch_completed_at?: string | null;
  last_batch_duration_ms?: number | null;
  last_batch_texts_per_second?: number | null;
  last_error?: string | null;
  total_batches_processed?: number;
  total_texts_processed?: number;
  recent_batches: Array<Record<string, unknown>>;
};

export type CompareResponse = {
  ok: boolean;
  model_id: string;
  dimension: number;
  backend: string;
  device: string;
  device_label: string;
  batched: boolean;
  load_duration_ms: number;
  encode_duration_ms: number;
  total_duration_ms: number;
  cosine_similarity: number | null;
  dot_product: number | null;
  euclidean_distance: number | null;
  vectors: number[][];
};

export type BenchmarkResponse = {
  ok: boolean;
  model_id: string;
  dimension: number;
  backend: string;
  device: string;
  device_label: string;
  repeat_count: number;
  texts_per_run: number;
  total_texts: number;
  total_chars: number;
  total_wall_time_ms: number;
  avg_wall_time_per_run_ms: number;
  texts_per_second: number | null;
};

type RequestOptions = RequestInit & {
  adminKey?: string;
};

async function requestJson<T>(input: string, init?: RequestOptions): Promise<T> {
  const { adminKey, ...requestInit } = init ?? {};
  const nextHeaders = new Headers(init?.headers ?? {});
  if (!(requestInit.body instanceof FormData) && !nextHeaders.has("Content-Type")) {
    nextHeaders.set("Content-Type", "application/json");
  }
  if (adminKey) {
    nextHeaders.set("X-Admin-Key", adminKey);
  }

  const response = await fetch(input, {
    headers: nextHeaders,
    ...requestInit,
  });

  if (response.status === 401) {
    throw new Error("unauthorized");
  }

  if (!response.ok) {
    const payload = await response
      .json()
      .catch(() => ({ detail: `HTTP ${response.status}` })) as { detail?: string };
    throw new Error(payload.detail ?? `HTTP ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function getKeys(adminKey: string) {
  return requestJson<{ admin_key: AdminKeyMetadata }>("/api/admin/keys", {
    method: "GET",
    adminKey,
  });
}

export async function rotateAdminKey(adminKey: string) {
  return requestJson<{ key: AdminKeyMetadata & { token: string }; keys: { admin_key: AdminKeyMetadata } }>("/api/admin/keys", {
    method: "POST",
    adminKey,
  });
}

export async function getSettings(adminKey: string) {
  return requestJson<SettingsResponse>("/api/admin/settings", {
    method: "GET",
    adminKey,
  });
}

export async function saveSettings(adminKey: string, settings: AdminSettings) {
  return requestJson<SettingsResponse & { ok: boolean }>("/api/admin/settings", {
    method: "PUT",
    adminKey,
    body: JSON.stringify(settings),
  });
}

export async function getSystem(adminKey: string) {
  return requestJson<{ system: RuntimeInfo["system"]; runtime: RuntimeInfo }>("/api/admin/system", {
    method: "GET",
    adminKey,
  });
}

export async function getModels(adminKey: string, storagePath?: string) {
  const query = storagePath !== undefined ? `?storage_path=${encodeURIComponent(storagePath)}` : "";
  return requestJson<{ models: ManagedModel[] }>(`/api/admin/models${query}`, {
    method: "GET",
    adminKey,
  });
}

export async function downloadModel(adminKey: string, modelId: string, storagePath: string, huggingfaceToken?: string) {
  return requestJson<{ job: Record<string, unknown>; models: ManagedModel[] }>("/api/admin/models/download", {
    method: "POST",
    adminKey,
    body: JSON.stringify({
      model_id: modelId,
      storage_path: storagePath,
      huggingface_token: huggingfaceToken,
    }),
  });
}

export async function deleteModel(adminKey: string, modelId: string, storagePath: string) {
  return requestJson<{ ok: boolean; removed: boolean; removed_path: string | null; storage_root: string; models: ManagedModel[] }>(
    "/api/admin/models/delete",
    {
      method: "POST",
      adminKey,
      body: JSON.stringify({
        model_id: modelId,
        storage_path: storagePath,
      }),
    },
  );
}

export async function getStats(adminKey: string) {
  return requestJson<StatsResponse>("/api/admin/stats", {
    method: "GET",
    adminKey,
  });
}

export async function getQueue(adminKey: string) {
  return requestJson<QueueResponse>("/api/admin/queue", {
    method: "GET",
    adminKey,
  });
}

export async function compareTexts(adminKey: string, model: string, textA: string, textB: string) {
  return requestJson<CompareResponse>("/api/admin/compare", {
    method: "POST",
    adminKey,
    body: JSON.stringify({
      model,
      text_a: textA,
      text_b: textB,
    }),
  });
}

export async function runBenchmark(adminKey: string, model: string, inputs: string[], repeatCount: number) {
  return requestJson<BenchmarkResponse>("/api/admin/benchmark", {
    method: "POST",
    adminKey,
    body: JSON.stringify({
      model,
      inputs,
      repeat_count: repeatCount,
    }),
  });
}
