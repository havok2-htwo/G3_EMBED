import { FormEvent, startTransition, useEffect, useRef, useState } from "react";

import {
  AdminOption,
  AdminSettings,
  ApiKeyInfo,
  BenchmarkResponse,
  CompareResponse,
  CreatedApiKey,
  ManagedModel,
  MediaCounts,
  QueueResponse,
  RuntimeInfo,
  SettingsPreset,
  SettingsResponse,
  StatsResponse,
  changePassword,
  compareTexts,
  createApiKey,
  deleteApiKey,
  deleteModel,
  deletePreset,
  downloadModel,
  getModels,
  getQueue,
  getSettings,
  getStats,
  listApiKeys,
  listPresets,
  login,
  logout,
  runBenchmark,
  savePreset,
  saveSettings,
  whoami,
} from "./api";

type HistoryEntry = {
  timestamp?: string;
  source_ip?: string;
  route?: string;
  model_id?: string;
  dimension?: number;
  backend?: string;
  device?: string;
  input_count?: number;
  input_chars?: number;
  input_tokens?: number | null;
  media?: MediaCounts | null;
  total_duration_ms?: number;
  encode_duration_ms?: number;
  texts_per_second?: number;
  batched?: boolean;
  success?: boolean;
  error?: string;
};

type QueuePoint = {
  queue: number;
  textsPerSecond: number;
};

type AuthState = "loading" | "login" | "change" | "ready";

const NUMBER_LOCALE = "de-DE";
const POLL_MS = 5000;
const MODEL_STATUS_POLL_MS = 2000;
const HISTORY_POINTS = 72;

const emptySettings: AdminSettings = {
  default_model: "google/embeddinggemma-2",
  execution_target: "auto",
  gpu_device: "auto",
  backend_override: "auto",
  model_cache_path: ".\\models",
  batching_enabled: true,
  batch_wait_time_ms: 25,
  batch_max_texts: 32,
  batch_max_chars: 64000,
  encode_batch_size: 0,
  encode_token_batching: true,
  max_seq_length: 0,
  too_long_policy: "reject",
  precision: "auto",
  quantization: "none",
  compile_mode: "off",
  cuda_graphs: true,
  warmup_on_load: false,
  idle_trim_seconds: 5,
  huggingface_token: "",
  image_inputs_enabled: true,
  audio_inputs_enabled: true,
  video_inputs_enabled: true,
  max_images_per_input: 4,
  max_images_per_request: 16,
  max_audio_clips_per_input: 2,
  max_audio_seconds_per_clip: 30,
  max_audio_seconds_per_request: 120,
  max_videos_per_request: 1,
  max_video_frames: 8,
  max_media_item_mb: 8,
  max_image_megapixels: 25,
  max_request_mb: 32,
  media_pass_budget_mb: 1200,
  max_concurrent_media_requests: 2,
};

type MediaLimitKey =
  | "max_images_per_input"
  | "max_images_per_request"
  | "max_image_megapixels"
  | "max_audio_clips_per_input"
  | "max_audio_seconds_per_clip"
  | "max_audio_seconds_per_request"
  | "max_videos_per_request"
  | "max_video_frames"
  | "max_media_item_mb"
  | "max_request_mb"
  | "media_pass_budget_mb"
  | "max_concurrent_media_requests";

// min/max mirror MEDIA_LIMIT_SETTINGS in the backend (normalize_settings clamps to the same range).
const mediaLimitFields: Array<{ key: MediaLimitKey; label: string; min: number; max: number; step?: number }> = [
  { key: "max_images_per_input", label: "Images / Input", min: 1, max: 8 },
  { key: "max_images_per_request", label: "Images / Request", min: 1, max: 32 },
  { key: "max_image_megapixels", label: "Image Size (MP)", min: 1, max: 50 },
  { key: "max_audio_clips_per_input", label: "Audio Clips / Input", min: 1, max: 4 },
  { key: "max_audio_seconds_per_clip", label: "Audio / Clip (s)", min: 1, max: 30 },
  { key: "max_audio_seconds_per_request", label: "Audio / Request (s)", min: 1, max: 300 },
  { key: "max_videos_per_request", label: "Videos / Request", min: 1, max: 2 },
  { key: "max_video_frames", label: "Frames / Video", min: 1, max: 16 },
  { key: "max_media_item_mb", label: "Media Item (MB)", min: 1, max: 25 },
  { key: "max_request_mb", label: "Request Body (MB)", min: 1, max: 256 },
  { key: "media_pass_budget_mb", label: "VRAM Budget / Pass (MB)", min: 200, max: 4096, step: 100 },
  { key: "max_concurrent_media_requests", label: "Concurrent Media Requests", min: 1, max: 8 },
];

const mediaToggleFields: Array<{ key: "image_inputs_enabled" | "audio_inputs_enabled" | "video_inputs_enabled"; label: string }> = [
  { key: "image_inputs_enabled", label: "Allow images" },
  { key: "audio_inputs_enabled", label: "Allow audio" },
  { key: "video_inputs_enabled", label: "Allow video (frames)" },
];

const emptyOptions = {
  models: [] as AdminOption[],
  execution_targets: [] as AdminOption[],
  gpu_devices: [] as AdminOption[],
  backends: [] as AdminOption[],
  precisions: [] as AdminOption[],
  quantizations: [] as AdminOption[],
  too_long_policies: [] as AdminOption[],
  compile_modes: [] as AdminOption[],
};

function formatValue(value: number | null | undefined, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  const maximumFractionDigits = Number.isInteger(value) ? 0 : 2;
  return `${new Intl.NumberFormat(NUMBER_LOCALE, { maximumFractionDigits }).format(value)}${suffix}`;
}

// Per loaded model with compile_mode != off: whether dynamo really compiled (it falls back to eager silently).
function compileSummary(models: Array<Record<string, unknown>>) {
  const parts = models.flatMap((model) => {
    const status = model.compile as { effective?: boolean; graphs?: number; reason?: string | null } | null | undefined;
    if (!status) {
      return [];
    }
    const name = String(model.model_id ?? "model");
    if (!status.effective) {
      return [`${name}: NOT effective (eager)${status.reason ? `, ${status.reason}` : ""}`];
    }
    return [`${name}: ${String(model.compile_mode)} (${formatValue(status.graphs ?? 0)} graphs)`];
  });
  return parts.length ? parts.join("; ") : "off";
}

function idleTrimSummary(idle: RuntimeInfo["idle_trim"]) {
  if (!idle) {
    return "n/a";
  }
  if (!idle.idle_trim_seconds) {
    return "off";
  }
  const next = idle.next_trim_in_seconds !== null ? `, next in ${formatValue(idle.next_trim_in_seconds)} s` : "";
  const freed = idle.last_freed_mb !== null ? `, last freed ${formatValue(idle.last_freed_mb)} MB` : "";
  return `after ${idle.idle_trim_seconds} s, ${formatValue(idle.trims)} trims${freed}${next}`;
}

function formatFixed(value: number | null | undefined, digits = 2, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return `${new Intl.NumberFormat(NUMBER_LOCALE, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value)}${suffix}`;
}

function formatDateTime(value?: string | null) {
  if (!value) {
    return "n/a";
  }
  return new Intl.DateTimeFormat(NUMBER_LOCALE, {
    year: "2-digit",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function formatModelSize(model: ManagedModel) {
  if (model.size_on_disk_gb !== null && model.size_on_disk_gb !== undefined) {
    return `${formatFixed(model.size_on_disk_gb, model.size_on_disk_gb >= 10 ? 1 : 2)} GB`;
  }
  if (model.approx_size_gb !== null && model.approx_size_gb !== undefined) {
    return `~${formatFixed(model.approx_size_gb, model.approx_size_gb >= 10 ? 1 : 2)} GB`;
  }
  return "n/a";
}

function formatModelStatus(status: string) {
  if (status === "ready") return "Ready";
  if (status === "downloading") return "Downloading";
  if (status === "partial") return "Partial";
  if (status === "error") return "Error";
  return "Missing";
}

// History never holds payloads, only counts; audio seconds are unknown when decoding failed.
function formatMediaCounts(media: MediaCounts) {
  const parts: string[] = [];
  if (media.images) parts.push(`${formatValue(media.images)} img`);
  if (media.audio_seconds) parts.push(`${formatValue(media.audio_seconds, " s")} audio`);
  if (media.video_frames) parts.push(`${formatValue(media.video_frames)} frames`);
  return parts.length > 0 ? parts.join(" · ") : "media";
}

function resolveOptionLabel(options: AdminOption[], value: string | null | undefined, fallback = "n/a") {
  if (!value) {
    return fallback;
  }
  return options.find((option) => option.value === value)?.label || value;
}

function numberFromRecord(record: Record<string, unknown>, key: string) {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function QueueSparkline({ points }: { points: QueuePoint[] }) {
  const visible = points.slice(-HISTORY_POINTS);
  const queuePoints = visible.map((entry) => entry.queue);
  const tpsPoints = visible.map((entry) => entry.textsPerSecond);
  const width = 180;
  const height = 180;
  const pad = 18;
  const chartWidth = width - pad * 2;
  const chartHeight = height - pad * 2;
  const queueMax = Math.max(1, ...queuePoints);
  const tpsMax = Math.max(1, ...tpsPoints);
  const pathFor = (values: number[], maxValue: number) =>
    values
      .map((value, index) => {
        const x = pad + (values.length <= 1 ? 0 : (index / (values.length - 1)) * chartWidth);
        const y = pad + chartHeight - (Math.max(0, value) / maxValue) * chartHeight;
        return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(" ");

  return (
    <div className="sparkline-shell">
      <svg className="graph" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-label="Embedding queue and throughput history">
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} className="graph-grid-line" />
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} className="graph-grid-line" />
        {pathFor(queuePoints, queueMax) ? <path d={pathFor(queuePoints, queueMax)} className="graph-path graph-path-queue" /> : null}
        {pathFor(tpsPoints, tpsMax) ? <path d={pathFor(tpsPoints, tpsMax)} className="graph-path graph-path-throughput" /> : null}
      </svg>
      <div className="graph-caption">
        <span>Queue</span>
        <strong>
          {formatValue(queuePoints.length ? queuePoints[queuePoints.length - 1] : 0)}
          {" | "}
          {formatValue(tpsPoints.length ? tpsPoints[tpsPoints.length - 1] : 0, "/s")}
        </strong>
      </div>
    </div>
  );
}

async function copyTextToClipboard(value: string) {
  if (!value) {
    return;
  }
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const textArea = document.createElement("textarea");
  textArea.value = value;
  textArea.setAttribute("readonly", "readonly");
  textArea.style.position = "fixed";
  textArea.style.opacity = "0";
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand("copy");
  document.body.removeChild(textArea);
}

function runtimeLabel(runtime: RuntimeInfo | null) {
  if (!runtime) {
    return "n/a";
  }
  const resolved = runtime.resolved_runtime;
  return `${resolved.backend}/${resolved.device} (${resolved.device_label})`;
}

function hardwareSummary(runtime: RuntimeInfo | null) {
  const items = runtime?.system.detected_accelerator_hardware ?? [];
  return items.length > 0 ? items.join(", ") : "CPU fallback";
}

export default function App() {
  const [authState, setAuthState] = useState<AuthState>("loading");
  const [currentUser, setCurrentUser] = useState("");
  const [loginUsername, setLoginUsername] = useState("admin");
  const [loginPassword, setLoginPassword] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");

  const [pwCurrent, setPwCurrent] = useState("");
  const [pwNew, setPwNew] = useState("");
  const [pwConfirm, setPwConfirm] = useState("");
  const [pwBusy, setPwBusy] = useState(false);
  const [pwError, setPwError] = useState("");
  const [pwMessage, setPwMessage] = useState("");

  const [apiKeys, setApiKeys] = useState<ApiKeyInfo[]>([]);
  const [newKeyAlias, setNewKeyAlias] = useState("");
  const [createdKey, setCreatedKey] = useState<CreatedApiKey | null>(null);
  const [apiKeyBusy, setApiKeyBusy] = useState(false);

  const [globalError, setGlobalError] = useState("");
  const [actionMessage, setActionMessage] = useState("");
  const [settingsForm, setSettingsForm] = useState<AdminSettings>(emptySettings);
  const [settingsOptions, setSettingsOptions] = useState(emptyOptions);
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const [managedModels, setManagedModels] = useState<ManagedModel[]>([]);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [queue, setQueue] = useState<QueueResponse | null>(null);
  const [queueHistory, setQueueHistory] = useState<QueuePoint[]>([]);
  const [saveBusy, setSaveBusy] = useState(false);
  // True while the settings form holds edits (or a loaded preset) that are not saved yet: the 5 s dashboard poll
  // then leaves the form alone instead of overwriting it with the stored settings.
  const settingsDirty = useRef(false);
  const [presets, setPresets] = useState<SettingsPreset[]>([]);
  const [presetName, setPresetName] = useState("");
  const [presetBusy, setPresetBusy] = useState(false);
  const [modelBusyId, setModelBusyId] = useState("");
  const [modelActionKind, setModelActionKind] = useState<"refresh" | "download" | "delete" | null>(null);
  const [compareTextA, setCompareTextA] = useState("Genesis stores memories as vectors.");
  const [compareTextB, setCompareTextB] = useState("A vector database can compare semantic meaning.");
  const [compareModel, setCompareModel] = useState(emptySettings.default_model);
  const [compareBusy, setCompareBusy] = useState(false);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [benchmarkInputs, setBenchmarkInputs] = useState("Text one\nText two\nText three");
  const [benchmarkRepeatCount, setBenchmarkRepeatCount] = useState(3);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(null);

  const isReady = authState === "ready";
  const hasDownloadingModels = managedModels.some((model) => model.status === "downloading");
  const history = (stats?.history ?? []) as HistoryEntry[];

  function applySettings(payload: SettingsResponse) {
    // Merge with defaults so an older backend without newer fields cannot crash the render.
    if (!settingsDirty.current) {
      setSettingsForm({ ...emptySettings, ...payload.settings });
    }
    setSettingsOptions({ ...emptyOptions, ...payload.options });
    setManagedModels(payload.models ?? []);
    setRuntime(payload.runtime);
    setCompareModel((current) => current || payload.settings.default_model);
  }

  function recordQueuePoint(nextQueue: QueueResponse) {
    const latestBatch = nextQueue.recent_batches?.[0] as Record<string, unknown> | undefined;
    const point = {
      queue: Number(nextQueue.queue_size ?? 0),
      textsPerSecond: latestBatch ? numberFromRecord(latestBatch, "texts_per_second") : Number(nextQueue.last_batch_texts_per_second ?? 0),
    };
    setQueueHistory((current) => [...current.slice(-(HISTORY_POINTS - 1)), point]);
  }

  function resetDashboardState() {
    startTransition(() => {
      setStats(null);
      setQueue(null);
      setRuntime(null);
      setManagedModels([]);
      settingsDirty.current = false;
      setSettingsForm(emptySettings);
      setPresets([]);
      setPresetName("");
      setSettingsOptions(emptyOptions);
      setApiKeys([]);
      setCreatedKey(null);
      setActionMessage("");
      setGlobalError("");
      setCompareResult(null);
      setBenchmarkResult(null);
    });
  }

  function handleApiError(error: unknown, fallback: string): boolean {
    const message = error instanceof Error ? error.message : "";
    if (message === "unauthorized") {
      resetDashboardState();
      setAuthState("login");
      setAuthError("Your session has expired. Please sign in again.");
      return true;
    }
    if (message === "password_change_required") {
      setAuthState("change");
      return true;
    }
    setGlobalError(error instanceof Error ? error.message : fallback);
    return false;
  }

  async function refreshDashboard() {
    try {
      const [settingsResponse, statsResponse, queueResponse, keysResponse] = await Promise.all([
        getSettings(),
        getStats(),
        getQueue(),
        listApiKeys(),
      ]);
      startTransition(() => {
        applySettings(settingsResponse);
        setStats(statsResponse);
        setQueue(queueResponse);
        setApiKeys(keysResponse.keys);
        recordQueuePoint(queueResponse);
        setGlobalError("");
      });
    } catch (error) {
      handleApiError(error, "Dashboard refresh failed.");
    }
  }

  // Bootstrap: check the admin session on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const me = await whoami();
        if (cancelled) return;
        setCurrentUser(me.username);
        setAuthState(me.must_change_password ? "change" : "ready");
      } catch {
        if (!cancelled) setAuthState("login");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!isReady) return;
    void refreshDashboard();
    const interval = window.setInterval(() => {
      void refreshDashboard();
    }, POLL_MS);
    return () => window.clearInterval(interval);
  }, [authState]);

  useEffect(() => {
    if (!isReady || !hasDownloadingModels) {
      return;
    }
    const interval = window.setInterval(() => {
      void refreshModels();
    }, MODEL_STATUS_POLL_MS);
    return () => window.clearInterval(interval);
  }, [authState, hasDownloadingModels, settingsForm.model_cache_path]);

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthBusy(true);
    setAuthError("");
    try {
      const me = await login(loginUsername.trim(), loginPassword);
      setCurrentUser(me.username);
      setLoginPassword("");
      setAuthState(me.must_change_password ? "change" : "ready");
    } catch (error) {
      setAuthError(
        error instanceof Error && error.message === "unauthorized"
          ? "Invalid username or password."
          : error instanceof Error
            ? error.message
            : "Login failed.",
      );
    } finally {
      setAuthBusy(false);
    }
  }

  async function handleChangePassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPwError("");
    setPwMessage("");
    if (pwNew.length < 4) {
      setPwError("The new password must be at least 4 characters.");
      return;
    }
    if (pwNew !== pwConfirm) {
      setPwError("The new password and its confirmation do not match.");
      return;
    }
    setPwBusy(true);
    try {
      await changePassword(pwCurrent, pwNew);
      setPwCurrent("");
      setPwNew("");
      setPwConfirm("");
      setPwMessage("Password updated.");
      setAuthState("ready");
    } catch (error) {
      setPwError(error instanceof Error ? error.message : "Password change failed.");
    } finally {
      setPwBusy(false);
    }
  }

  async function handleLogout() {
    try {
      await logout();
    } catch {
      // clear locally regardless
    }
    resetDashboardState();
    setLoginPassword("");
    setAuthState("login");
  }

  async function handleCreateApiKey(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setApiKeyBusy(true);
    setActionMessage("");
    setGlobalError("");
    try {
      const created = await createApiKey(newKeyAlias.trim());
      setCreatedKey(created);
      setNewKeyAlias("");
      setActionMessage(`API key "${created.alias}" created. Copy it now — it is shown only once.`);
      const keysResponse = await listApiKeys();
      setApiKeys(keysResponse.keys);
    } catch (error) {
      handleApiError(error, "The API key could not be created.");
    } finally {
      setApiKeyBusy(false);
    }
  }

  async function handleDeleteApiKey(keyId: string, alias: string) {
    setActionMessage("");
    setGlobalError("");
    try {
      await deleteApiKey(keyId);
      setCreatedKey((current) => (current?.id === keyId ? null : current));
      setActionMessage(`API key "${alias}" deleted.`);
      const keysResponse = await listApiKeys();
      setApiKeys(keysResponse.keys);
    } catch (error) {
      handleApiError(error, "The API key could not be deleted.");
    }
  }

  async function handleSaveSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaveBusy(true);
    setActionMessage("");
    setGlobalError("");
    try {
      const response = await saveSettings(settingsForm);
      settingsDirty.current = false;
      applySettings(response);
      setActionMessage("Settings saved.");
    } catch (error) {
      handleApiError(error, "Settings could not be saved.");
    } finally {
      setSaveBusy(false);
    }
  }

  async function refreshModels() {
    if (!isReady) {
      return;
    }
    setModelBusyId("__refresh__");
    setModelActionKind("refresh");
    try {
      const response = await getModels(settingsForm.model_cache_path);
      setManagedModels(response.models ?? []);
    } catch (error) {
      handleApiError(error, "Model refresh failed.");
    } finally {
      setModelBusyId("");
      setModelActionKind(null);
    }
  }

  async function handleDownloadModel(model: ManagedModel) {
    setModelBusyId(model.id);
    setModelActionKind("download");
    setGlobalError("");
    try {
      const response = await downloadModel(model.id, settingsForm.model_cache_path, settingsForm.huggingface_token);
      setManagedModels(response.models ?? []);
    } catch (error) {
      handleApiError(error, "Model download failed.");
    } finally {
      setModelBusyId("");
      setModelActionKind(null);
    }
  }

  async function handleDeleteModel(model: ManagedModel) {
    setModelBusyId(model.id);
    setModelActionKind("delete");
    setGlobalError("");
    try {
      const response = await deleteModel(model.id, settingsForm.model_cache_path);
      setManagedModels(response.models ?? []);
    } catch (error) {
      handleApiError(error, "Model delete failed.");
    } finally {
      setModelBusyId("");
      setModelActionKind(null);
    }
  }

  async function handleCompare(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setCompareBusy(true);
    setCompareResult(null);
    setGlobalError("");
    try {
      const result = await compareTexts(compareModel, compareTextA, compareTextB);
      setCompareResult(result);
      setActionMessage(`Compared ${result.dimension} dimensions in ${formatValue(result.total_duration_ms, " ms")}.`);
      await refreshDashboard();
    } catch (error) {
      handleApiError(error, "Compare failed.");
    } finally {
      setCompareBusy(false);
    }
  }

  async function handleBenchmark(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const inputs = benchmarkInputs.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (inputs.length === 0) {
      setGlobalError("Benchmark inputs are empty.");
      return;
    }
    setBenchmarkBusy(true);
    setBenchmarkResult(null);
    setGlobalError("");
    try {
      const result = await runBenchmark(compareModel, inputs, benchmarkRepeatCount);
      setBenchmarkResult(result);
      setActionMessage(`Benchmark finished: ${formatValue(result.total_texts)} texts in ${formatValue(result.total_wall_time_ms, " ms")}.`);
    } catch (error) {
      handleApiError(error, "Benchmark failed.");
    } finally {
      setBenchmarkBusy(false);
    }
  }

  function updateSetting<K extends keyof AdminSettings>(key: K, value: AdminSettings[K]) {
    settingsDirty.current = true;
    setSettingsForm((current) => ({ ...current, [key]: value }));
  }

  function handleDiscardSettings() {
    settingsDirty.current = false;
    setActionMessage("");
    void refreshDashboard();
  }

  const selectedPreset = presets.find((preset) => preset.name === presetName);

  async function reloadPresets() {
    try {
      setPresets((await listPresets()).presets);
    } catch (error) {
      handleApiError(error, "Presets could not be loaded.");
    }
  }

  useEffect(() => {
    // Loaded once per sign-in; saving or deleting reloads explicitly.
    if (isReady) void reloadPresets();
  }, [authState]);

  function handleLoadPreset() {
    if (!selectedPreset) return;
    setActionMessage("");
    setGlobalError("");
    // Into the form only: Save Settings applies it through the normal validated PUT.
    settingsDirty.current = true;
    setSettingsForm((current) => ({ ...current, ...selectedPreset.values }));
    setActionMessage(
      `Preset "${selectedPreset.name}" loaded into the form; review it and click Save Settings to apply.` +
        (selectedPreset.note ? ` ${selectedPreset.note}` : ""),
    );
  }

  async function handleSavePreset() {
    const name = presetName.trim();
    if (!name) return;
    setPresetBusy(true);
    setActionMessage("");
    setGlobalError("");
    try {
      const response = await savePreset(name, settingsForm);
      await reloadPresets();
      setPresetName(response.preset.name);
      setActionMessage(`Preset "${response.preset.name}" saved from the current form.`);
    } catch (error) {
      handleApiError(error, "Preset could not be saved.");
    } finally {
      setPresetBusy(false);
    }
  }

  async function handleDeletePreset() {
    if (!selectedPreset || selectedPreset.built_in) return;
    const name = selectedPreset.name;
    setPresetBusy(true);
    setActionMessage("");
    setGlobalError("");
    try {
      await deletePreset(name);
      setPresetName("");
      await reloadPresets();
      setActionMessage(`Preset "${name}" deleted.`);
    } catch (error) {
      handleApiError(error, "Preset could not be deleted.");
    } finally {
      setPresetBusy(false);
    }
  }

  if (authState === "loading") {
    return (
      <main className="shell centered">
        <section className="panel login-panel">
          <p className="message">Loading...</p>
        </section>
      </main>
    );
  }

  if (authState === "login") {
    return (
      <main className="shell centered">
        <section className="panel login-panel">
          <div className="hero-copy">
            <span className="eyebrow">Private Access</span>
            <h1>G3_EMBED Admin</h1>
            <p>
              The public embedding API stays open until you create an API key. The private dashboard for models,
              hardware routing, batching, and local test runs is protected by a username and password login.
            </p>
          </div>
          <form className="login-form" onSubmit={handleLogin}>
            <label>
              <span>Username</span>
              <input value={loginUsername} autoComplete="username" onChange={(event) => setLoginUsername(event.target.value)} placeholder="admin" />
            </label>
            <label>
              <span>Password</span>
              <input type="password" value={loginPassword} autoComplete="current-password" onChange={(event) => setLoginPassword(event.target.value)} placeholder="admin" />
            </label>
            <button type="submit" disabled={authBusy || !loginUsername.trim() || !loginPassword}>
              {authBusy ? "Signing in..." : "Sign In"}
            </button>
            <p className="message">Default credentials: admin / admin. You must change the password on first login.</p>
            {authError ? <p className="message error">{authError}</p> : null}
          </form>
        </section>
      </main>
    );
  }

  if (authState === "change") {
    return (
      <main className="shell centered">
        <section className="panel login-panel">
          <div className="hero-copy">
            <span className="eyebrow">Security</span>
            <h1>Set a New Password</h1>
            <p>You are signed in as <strong>{currentUser || "admin"}</strong>. Choose a new password before continuing.</p>
          </div>
          <form className="login-form" onSubmit={handleChangePassword}>
            <label>
              <span>Current Password</span>
              <input type="password" value={pwCurrent} autoComplete="current-password" onChange={(event) => setPwCurrent(event.target.value)} />
            </label>
            <label>
              <span>New Password</span>
              <input type="password" value={pwNew} autoComplete="new-password" onChange={(event) => setPwNew(event.target.value)} />
            </label>
            <label>
              <span>Confirm New Password</span>
              <input type="password" value={pwConfirm} autoComplete="new-password" onChange={(event) => setPwConfirm(event.target.value)} />
            </label>
            <button type="submit" disabled={pwBusy || !pwCurrent || !pwNew || !pwConfirm}>
              {pwBusy ? "Saving..." : "Save New Password"}
            </button>
            <button type="button" className="ghost-button" onClick={() => void handleLogout()}>Cancel & Sign Out</button>
            {pwMessage ? <p className="message">{pwMessage}</p> : null}
            {pwError ? <p className="message error">{pwError}</p> : null}
          </form>
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      <header className="hero">
        <div className="hero-copy-block">
          <span className="eyebrow">G3_EMBED</span>
          <h1>Local embedding server</h1>
          <p className="hero-copy">Genesis-compatible vectors, OpenAI-compatible embeddings, and a small lab for checking semantic distance.</p>
        </div>
        <div className="hero-actions">
          <a className="secondary-link" href="/docs">OpenAPI</a>
          <button type="button" className="secondary-button" onClick={() => void refreshDashboard()}>Refresh</button>
          <button type="button" className="ghost-button" onClick={() => void handleLogout()}>Logout</button>
        </div>
      </header>

      {globalError ? <p className="message error">{globalError}</p> : null}
      {actionMessage ? <p className="message">{actionMessage}</p> : null}

      <section className="stats-grid">
        <div className="stat-card">
          <span>Active Model</span>
          <strong>{resolveOptionLabel(settingsOptions.models, settingsForm.default_model)}</strong>
          <small className="mono">{settingsForm.default_model}</small>
        </div>
        <div className="stat-card">
          <span>Runtime</span>
          <strong>{runtimeLabel(runtime)}</strong>
          <small>{runtime?.resolved_runtime.detail ?? "n/a"}</small>
        </div>
        <div className="stat-card">
          <span>Requests</span>
          <strong>{formatValue(stats?.summary.total_requests ?? 0)}</strong>
          <small>{formatValue(stats?.summary.avg_total_duration_ms, " ms")} avg total</small>
        </div>
        <div className="stat-card">
          <span>Signed in as</span>
          <strong>{currentUser || "admin"}</strong>
          <small>{formatValue(queue?.queue_size ?? 0)} in queue</small>
        </div>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Account & API Keys</span>
            <h2>Access</h2>
          </div>
        </div>
        <div className="panel-grid">
          <form className="settings-form" onSubmit={handleChangePassword}>
            <span className="eyebrow full-width">Change password ({currentUser || "admin"})</span>
            <label className="full-width"><span>Current Password</span><input type="password" value={pwCurrent} onChange={(event) => setPwCurrent(event.target.value)} /></label>
            <label className="full-width"><span>New Password</span><input type="password" value={pwNew} onChange={(event) => setPwNew(event.target.value)} /></label>
            <label className="full-width"><span>Confirm</span><input type="password" value={pwConfirm} onChange={(event) => setPwConfirm(event.target.value)} /></label>
            <div className="form-actions full-width">
              <button type="submit" disabled={pwBusy || !pwCurrent || !pwNew || !pwConfirm}>{pwBusy ? "Saving..." : "Change Password"}</button>
              {pwMessage ? <p className="message">{pwMessage}</p> : null}
              {pwError ? <p className="message error">{pwError}</p> : null}
            </div>
          </form>
          <div className="side-panel">
            <span className="eyebrow full-width">API Keys</span>
            <p className="section-copy">
              While no key exists, <code>POST /embed</code> and <code>POST /v1/embeddings</code> are open. Once one key
              exists, a valid <code>X-API-Key</code> header is required. Usage counts processed texts per key.
            </p>
            {createdKey ? (
              <div className="key-token-card">
                <strong>{createdKey.alias}</strong>
                <p>Copy this key now — it is shown only once.</p>
                <div className="key-token-value mono">{createdKey.token}</div>
                <button type="button" className="secondary-button" onClick={() => void copyTextToClipboard(createdKey.token)}>Copy</button>
              </div>
            ) : null}
            <form className="benchmark-form" onSubmit={handleCreateApiKey}>
              <label className="full-width"><span>Alias</span><input value={newKeyAlias} onChange={(event) => setNewKeyAlias(event.target.value)} placeholder="e.g. Key fuer Projekt X" /></label>
              <div className="form-actions"><button type="submit" disabled={apiKeyBusy || !newKeyAlias.trim()}>{apiKeyBusy ? "Creating..." : "Create API Key"}</button></div>
            </form>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Alias</th><th>Created</th><th>Texts</th><th>Requests</th><th>Last Used</th><th></th></tr></thead>
                <tbody>
                  {apiKeys.length === 0 ? <tr><td colSpan={6}>No API keys — the public API is currently open.</td></tr> : null}
                  {apiKeys.map((key) => (
                    <tr key={key.id}>
                      <td>{key.alias}</td>
                      <td>{formatDateTime(key.created_at)}</td>
                      <td>{formatValue(key.usage.total_items_processed)}</td>
                      <td>{key.usage.request_count}</td>
                      <td>{formatDateTime(key.usage.last_used_at)}</td>
                      <td><button type="button" className="ghost-button danger-button" onClick={() => void handleDeleteApiKey(key.id, key.alias)}>Delete</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <div className="panel-grid">
        <section className="panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Embedding Lab</span>
              <h2>Compare two texts</h2>
            </div>
            <div className="loaded-model">
              <span>Hardware</span>
              <strong>{hardwareSummary(runtime)}</strong>
            </div>
          </div>
          <form className="lab-form" onSubmit={handleCompare}>
            <label>
              <span>Model</span>
              <select value={compareModel} onChange={(event) => setCompareModel(event.target.value)}>
                {settingsOptions.models.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label className="full-width">
              <span>Text A</span>
              <textarea value={compareTextA} onChange={(event) => setCompareTextA(event.target.value)} rows={6} />
            </label>
            <label className="full-width">
              <span>Text B</span>
              <textarea value={compareTextB} onChange={(event) => setCompareTextB(event.target.value)} rows={6} />
            </label>
            <div className="form-actions full-width">
              <button type="submit" disabled={compareBusy}>{compareBusy ? "Embedding..." : "Embed & Compare"}</button>
              {compareResult ? (
                <button type="button" className="secondary-button" onClick={() => void copyTextToClipboard(JSON.stringify(compareResult.vectors, null, 2))}>
                  Copy Vectors
                </button>
              ) : null}
            </div>
          </form>

          {compareResult ? (
            <div className="result-grid">
              <div><span>Cosine</span><strong>{formatFixed(compareResult.cosine_similarity, 6)}</strong></div>
              <div><span>Dot</span><strong>{formatFixed(compareResult.dot_product, 6)}</strong></div>
              <div><span>Distance</span><strong>{formatFixed(compareResult.euclidean_distance, 6)}</strong></div>
              <div><span>Dimension</span><strong>{formatValue(compareResult.dimension)}</strong></div>
              <div><span>Total</span><strong>{formatValue(compareResult.total_duration_ms, " ms")}</strong></div>
              <div><span>Encode</span><strong>{formatValue(compareResult.encode_duration_ms, " ms")}</strong></div>
              <div><span>Backend</span><strong>{compareResult.backend}/{compareResult.device}</strong></div>
              <div><span>Batched</span><strong>{compareResult.batched ? "yes" : "no"}</strong></div>
            </div>
          ) : null}
        </section>

        <section className="panel side-panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Queue</span>
              <h2>Live throughput</h2>
            </div>
          </div>
          <QueueSparkline points={queueHistory} />
          <div className="queue-metrics">
            <div><span>Active Batch</span><strong>{queue?.active_batch_id ?? "none"}</strong></div>
            <div><span>Active Texts</span><strong>{formatValue(queue?.active_batch_size ?? 0)}</strong></div>
            <div><span>Active Chars</span><strong>{formatValue(queue?.active_batch_chars ?? 0)}</strong></div>
            <div><span>Processed</span><strong>{formatValue(queue?.total_texts_processed ?? 0)}</strong></div>
          </div>
        </section>
      </div>

      <div className="content-grid">
        <section className="panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Settings</span>
              <h2>Model, backend, batching</h2>
            </div>
          </div>
          <span className="eyebrow">Presets</span>
          <p className="section-copy">
            Built-in GPU tiers: 32GB VRAM (RTX 5090) and 24GB VRAM (RTX 4090) with every G3 service on one card; 24+8 VRAM runs
            EMBED on the 24 GB card of a 3090 + 2070S box (the GPU is picked by card size). Loading fills the form below;
            Save Settings applies it.
          </p>
          <div className="settings-form">
            <label>
              <span>Preset</span>
              <select value={selectedPreset ? presetName : ""} onChange={(event) => setPresetName(event.target.value)}>
                <option value="">(choose)</option>
                {presets.map((preset) => (
                  <option key={preset.name} value={preset.name}>{preset.built_in ? `${preset.name} (built-in)` : preset.name}</option>
                ))}
              </select>
              {selectedPreset?.description ? <small className="muted">{selectedPreset.description}</small> : null}
              {selectedPreset?.note ? <small className="muted">{selectedPreset.note}</small> : null}
            </label>
            <label>
              <span>Save the current form as</span>
              <input value={presetName} maxLength={64} placeholder="preset name" onChange={(event) => setPresetName(event.target.value)} />
              <small className="muted">Built-in names cannot be overwritten. The cache path and the Hugging Face token are never stored in a preset.</small>
            </label>
            <div className="button-row full-width">
              <button type="button" className="secondary-button" onClick={handleLoadPreset} disabled={!selectedPreset || presetBusy || saveBusy}>Load into form</button>
              <button type="button" className="secondary-button" onClick={() => void handleSavePreset()} disabled={!presetName.trim() || selectedPreset?.built_in === true || presetBusy}>Save as preset</button>
              <button type="button" className="ghost-button danger-button" onClick={() => void handleDeletePreset()} disabled={!selectedPreset || selectedPreset.built_in || presetBusy}>Delete preset</button>
            </div>
          </div>
          <div className="panel-divider" />
          <form className="settings-form" onSubmit={handleSaveSettings}>
            <label>
              <span>Default Model</span>
              <select value={settingsForm.default_model} onChange={(event) => updateSetting("default_model", event.target.value)}>
                {settingsOptions.models.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Target</span>
              <select value={settingsForm.execution_target} onChange={(event) => updateSetting("execution_target", event.target.value)}>
                {settingsOptions.execution_targets.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label title="Numbered like nvidia-smi. Switching reloads the model on the new card; the old card keeps the CUDA context (~0.5 GB) until restart.">
              <span>GPU</span>
              <select value={settingsForm.gpu_device} onChange={(event) => updateSetting("gpu_device", event.target.value)}>
                {!(settingsOptions.gpu_devices ?? []).some((option) => option.value === settingsForm.gpu_device) ? (
                  <option value={settingsForm.gpu_device}>{settingsForm.gpu_device}</option>
                ) : null}
                {(settingsOptions.gpu_devices ?? []).map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Backend</span>
              <select value={settingsForm.backend_override} onChange={(event) => updateSetting("backend_override", event.target.value)}>
                {settingsOptions.backends.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Precision</span>
              <select value={settingsForm.precision} onChange={(event) => updateSetting("precision", event.target.value)}>
                {settingsOptions.precisions.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Quantization (CUDA)</span>
              <select value={settingsForm.quantization} onChange={(event) => updateSetting("quantization", event.target.value)}>
                {settingsOptions.quantizations.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Compile</span>
              <select value={settingsForm.compile_mode} onChange={(event) => updateSetting("compile_mode", event.target.value)}>
                {settingsOptions.compile_modes.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Batch Wait (ms)</span>
              <input type="number" min={0} value={settingsForm.batch_wait_time_ms} onChange={(event) => updateSetting("batch_wait_time_ms", Number(event.target.value))} />
            </label>
            <label>
              <span>Max Texts</span>
              <input type="number" min={1} value={settingsForm.batch_max_texts} onChange={(event) => updateSetting("batch_max_texts", Number(event.target.value))} />
            </label>
            <label>
              <span>Max Chars</span>
              <input type="number" min={256} value={settingsForm.batch_max_chars} onChange={(event) => updateSetting("batch_max_chars", Number(event.target.value))} />
            </label>
            <label>
              <span>GPU Batch Size (0 = auto)</span>
              <input type="number" min={0} value={settingsForm.encode_batch_size} onChange={(event) => updateSetting("encode_batch_size", Number(event.target.value))} />
            </label>
            <label>
              <span>Max Tokens / Text (0 = model default)</span>
              <input type="number" min={0} step={256} value={settingsForm.max_seq_length} onChange={(event) => updateSetting("max_seq_length", Number(event.target.value))} />
            </label>
            <label title="Seconds after the last request before the pass working set (cached CUDA memory) goes back to the driver; CUDA graphs stay.">
              <span>Idle Trim (s, 0 = off)</span>
              <input type="number" min={0} max={3600} value={settingsForm.idle_trim_seconds} onChange={(event) => updateSetting("idle_trim_seconds", Number(event.target.value))} />
            </label>
            <label>
              <span>Texts Over Token Limit</span>
              <select value={settingsForm.too_long_policy} onChange={(event) => updateSetting("too_long_policy", event.target.value)}>
                {settingsOptions.too_long_policies.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label className="full-width">
              <span>Cache Path</span>
              <input value={settingsForm.model_cache_path} onChange={(event) => updateSetting("model_cache_path", event.target.value)} />
            </label>
            <label className="full-width">
              <span>Hugging Face Token</span>
              <input type="password" placeholder="hf_... (optional)" value={settingsForm.huggingface_token} onChange={(event) => updateSetting("huggingface_token", event.target.value)} />
            </label>
            <label className="switch-row">
              <input type="checkbox" checked={settingsForm.batching_enabled} onChange={(event) => updateSetting("batching_enabled", event.target.checked)} />
              <span>Enable batching</span>
            </label>
            <label className="switch-row">
              <input type="checkbox" checked={settingsForm.warmup_on_load} onChange={(event) => updateSetting("warmup_on_load", event.target.checked)} />
              <span>Warm up on load</span>
            </label>
            <label className="switch-row">
              <input type="checkbox" checked={settingsForm.encode_token_batching} onChange={(event) => updateSetting("encode_token_batching", event.target.checked)} />
              <span>Pack texts by tokens</span>
            </label>
            <label className="switch-row">
              <input type="checkbox" checked={settingsForm.cuda_graphs} onChange={(event) => updateSetting("cuda_graphs", event.target.checked)} />
              <span>CUDA graphs (EmbeddingGemma 2)</span>
            </label>
            <span className="eyebrow full-width">Media limits (image / audio / video)</span>
            <div className="button-row full-width">
              {mediaToggleFields.map((field) => (
                <label key={field.key} className="switch-row">
                  <input type="checkbox" checked={settingsForm[field.key]} onChange={(event) => updateSetting(field.key, event.target.checked)} />
                  <span>{field.label}</span>
                </label>
              ))}
            </div>
            {mediaLimitFields.map((field) => (
              <label key={field.key}>
                <span>{field.label}</span>
                <input type="number" min={field.min} max={field.max} step={field.step ?? 1} value={settingsForm[field.key]} onChange={(event) => updateSetting(field.key, Number(event.target.value))} />
              </label>
            ))}
            <div className="form-actions full-width">
              <button type="submit" disabled={saveBusy}>{saveBusy ? "Saving..." : "Save Settings"}</button>
              <button type="button" className="ghost-button" disabled={saveBusy} onClick={handleDiscardSettings}>Discard changes</button>
            </div>
          </form>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Benchmark</span>
              <h2>Throughput test</h2>
            </div>
          </div>
          <form className="benchmark-form" onSubmit={handleBenchmark}>
            <label className="full-width">
              <span>Inputs</span>
              <textarea value={benchmarkInputs} onChange={(event) => setBenchmarkInputs(event.target.value)} rows={6} />
            </label>
            <label>
              <span>Repeats</span>
              <input type="number" min={1} max={128} value={benchmarkRepeatCount} onChange={(event) => setBenchmarkRepeatCount(Math.max(1, Math.min(128, Number(event.target.value) || 1)))} />
            </label>
            <div className="form-actions">
              <button type="submit" disabled={benchmarkBusy}>{benchmarkBusy ? "Running..." : "Run Benchmark"}</button>
            </div>
          </form>
          {benchmarkResult ? (
            <div className="result-grid compact">
              <div><span>Total</span><strong>{formatValue(benchmarkResult.total_wall_time_ms, " ms")}</strong></div>
              <div><span>Texts/s</span><strong>{formatValue(benchmarkResult.texts_per_second)}</strong></div>
              <div><span>Texts</span><strong>{formatValue(benchmarkResult.total_texts)}</strong></div>
              <div><span>Backend</span><strong>{benchmarkResult.backend}/{benchmarkResult.device}</strong></div>
            </div>
          ) : null}
        </section>
      </div>

      <section className="panel">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Models</span>
            <h2>Cache Manager</h2>
            <p className="section-copy">Download or remove supported embedding models in the configured cache path.</p>
          </div>
          <button type="button" className="secondary-button" onClick={() => void refreshModels()} disabled={modelBusyId === "__refresh__"}>
            {modelBusyId === "__refresh__" ? "Refreshing..." : "Refresh Models"}
          </button>
        </div>
        <div className="model-path-card">
          <span>Selected Cache Path</span>
          <strong className="mono">{settingsForm.model_cache_path || "Default Hugging Face cache"}</strong>
        </div>
        <div className="table-wrap">
          <table className="model-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Tier</th>
                <th>Size</th>
                <th>Status</th>
                <th>Path</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {managedModels.map((model) => {
                const rowBusy = modelBusyId === model.id;
                const isDownloading = model.status === "downloading";
                return (
                  <tr key={model.id}>
                    <td>
                      <strong>{model.label}</strong>
                      <div className="muted mono model-id">{model.id}</div>
                      <div className="muted">{model.notes}</div>
                    </td>
                    <td>{model.tier}</td>
                    <td>{formatModelSize(model)}</td>
                    <td className="model-status-cell">
                      <strong>{formatModelStatus(model.status)}</strong>
                      {model.error ? <span className="model-status-error">{model.error}</span> : null}
                    </td>
                    <td className="mono model-path-cell">{model.local_path || model.cache_path || model.storage_root}</td>
                    <td className="model-actions-cell">
                      <div className="table-actions">
                        <button type="button" className="secondary-button" onClick={() => void handleDownloadModel(model)} disabled={isDownloading || rowBusy}>
                          {isDownloading ? "Downloading..." : rowBusy && modelActionKind === "download" ? "Starting..." : model.status === "ready" ? "Download Again" : "Download"}
                        </button>
                        <button type="button" className="ghost-button danger-button" onClick={() => void handleDeleteModel(model)} disabled={!model.cache_path || isDownloading || rowBusy}>
                          {rowBusy && modelActionKind === "delete" ? "Deleting..." : "Delete"}
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <div className="content-grid">
        <section className="panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">History</span>
              <h2>Latest embeddings</h2>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Route</th>
                  <th>Model</th>
                  <th>Inputs</th>
                  <th>Dim</th>
                  <th>Total</th>
                  <th>Encode</th>
                  <th>Batch</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {history.length === 0 ? (
                  <tr><td colSpan={9}>No embedding history recorded yet.</td></tr>
                ) : history.map((entry, index) => (
                  <tr key={`${entry.timestamp ?? "row"}-${index}`}>
                    <td>{formatDateTime(entry.timestamp)}</td>
                    <td>{entry.route ?? "n/a"}</td>
                    <td className="mono">{entry.model_id ?? "n/a"}</td>
                    <td>
                      {formatValue(entry.input_count)} / {formatValue(entry.input_chars)} chars
                      {entry.input_tokens !== null && entry.input_tokens !== undefined ? ` / ${formatValue(entry.input_tokens)} tokens` : null}
                      {entry.media ? <div className="muted">{formatMediaCounts(entry.media)}</div> : null}
                    </td>
                    <td>{formatValue(entry.dimension)}</td>
                    <td>{formatValue(entry.total_duration_ms, " ms")}</td>
                    <td>{formatValue(entry.encode_duration_ms, " ms")}</td>
                    <td>{entry.batched ? "yes" : "no"}</td>
                    <td>{entry.success ? "ok" : entry.error ?? "error"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">System</span>
              <h2>Detected devices</h2>
            </div>
          </div>
          <div className="system-list">
            <div><span>Platform</span><strong>{runtime?.system.platform ?? "n/a"}</strong></div>
            <div><span>OpenVINO</span><strong>{runtime?.system.openvino_devices?.join(", ") || "n/a"}</strong></div>
            <div><span>NVIDIA</span><strong>{runtime?.system.nvidia_gpus?.join(", ") || "n/a"}</strong></div>
            <div><span>NPU</span><strong>{runtime?.system.windows_npu_devices?.join(", ") || "n/a"}</strong></div>
            <div><span>Loaded Models</span><strong>{formatValue(runtime?.loaded_models.length ?? 0)}</strong></div>
            <div><span>Compile</span><strong>{compileSummary(runtime?.loaded_models ?? [])}</strong></div>
            <div><span>Idle Trim</span><strong>{idleTrimSummary(runtime?.idle_trim)}</strong></div>
            <div><span>Last Runtime Error</span><strong>{runtime?.last_runtime_error ?? "none"}</strong></div>
          </div>
        </section>
      </div>
    </main>
  );
}
