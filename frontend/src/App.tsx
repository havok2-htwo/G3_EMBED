import { FormEvent, KeyboardEvent, ReactNode, useCallback, useEffect, useRef, useState } from "react";

import {
  AdminOption,
  AdminSettings,
  ApiKeyInfo,
  BenchmarkResponse,
  CompareResponse,
  ManagedModel,
  MediaCounts,
  QueueResponse,
  RuntimeInfo,
  SettingsPreset,
  SettingsResponse,
  SettingsUpdate,
  SetupStatus,
  StatsResponse,
  changePassword,
  compareTexts,
  createApiKey,
  deleteApiKey,
  deleteModel,
  deletePreset,
  downloadModel,
  getModels,
  getPerf,
  getQueue,
  getSettings,
  getSetupStatus,
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
import { ApiKeysPanel } from "./ApiKeysPanel";
import { fmtDateTime, fmtDuration, fmtFixed, fmtGb, fmtMs, fmtNum, fmtRate } from "./format";
import { PerfPanel, clearPerfHistory, gpuTitle, usePerfFeed, usePerfLatest } from "./PerfPanel";
import { CopyButton, ToastHost, toast } from "./Toasts";
import { useVisiblePolling } from "./usePolling";

// ---------- constants ----------
type AuthState = "loading" | "error" | "login" | "change" | "ready";
type TabId = "dashboard" | "playground" | "models" | "settings" | "benchmark" | "history" | "api-keys" | "account";

const TABS: Array<{ id: TabId; label: string }> = [
  { id: "dashboard", label: "Dashboard" },
  { id: "playground", label: "Playground" },
  { id: "models", label: "Models" },
  { id: "settings", label: "Settings" },
  { id: "benchmark", label: "Benchmark" },
  { id: "history", label: "History" },
  { id: "api-keys", label: "API keys" },
  { id: "account", label: "Account" },
];

const POLL_MS = 5000;
const MODEL_STATUS_POLL_MS = 2000;
const KEYS_BACKGROUND_POLL_MS = 60_000;
const MIN_PASSWORD_LENGTH = 8;
const SERVICE_NAME = "G3 Embed";
const PUBLIC_ROUTES = "POST /embed, POST /v1/embeddings and GET /v1/models";

function tabFromHash(): TabId {
  const value = window.location.hash.replace(/^#\/?/, "");
  return TABS.some((tab) => tab.id === value) ? (value as TabId) : "dashboard";
}

const errorText = (error: unknown, fallback: string) => (error instanceof Error && error.message ? error.message : fallback);

// ---------- settings form model ----------
type NumberKey = { [K in keyof AdminSettings]: AdminSettings[K] extends number ? K : never }[keyof AdminSettings];
// Number fields may be "" while the user edits them: an empty field is "not set" and is never sent as 0.
type SettingsForm = { [K in keyof AdminSettings]: AdminSettings[K] extends number ? number | "" : AdminSettings[K] };

const emptySettings: AdminSettings = {
  default_model: "",
  execution_target: "auto",
  gpu_device: "auto",
  backend_override: "auto",
  model_cache_path: "",
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

// Ranges mirror normalize_settings / MEDIA_LIMIT_SETTINGS in the backend (it clamps to the same values).
const NUMBER_FIELDS: Record<NumberKey, { label: string; min: number; max: number; step?: number; hint?: string }> = {
  batch_wait_time_ms: { label: "Batch wait (ms)", min: 0, max: 5000, hint: "How long the queue collects requests into one batch." },
  batch_max_texts: { label: "Max texts per batch", min: 1, max: 2048 },
  batch_max_chars: { label: "Max characters per batch", min: 256, max: 4_000_000 },
  encode_batch_size: { label: "GPU batch size", min: 0, max: 2048, hint: "0 = automatic." },
  max_seq_length: { label: "Max tokens per text", min: 0, max: 32768, step: 256, hint: "0 = the model's own limit." },
  idle_trim_seconds: { label: "Idle trim (s)", min: 0, max: 3600, hint: "Seconds after the last request before cached CUDA memory goes back to the driver. 0 = off." },
  max_images_per_input: { label: "Images per input", min: 1, max: 8 },
  max_images_per_request: { label: "Images per request", min: 1, max: 32 },
  max_image_megapixels: { label: "Image size (MP)", min: 1, max: 50 },
  max_audio_clips_per_input: { label: "Audio clips per input", min: 1, max: 4 },
  max_audio_seconds_per_clip: { label: "Audio per clip (s)", min: 1, max: 30 },
  max_audio_seconds_per_request: { label: "Audio per request (s)", min: 1, max: 300 },
  max_videos_per_request: { label: "Videos per request", min: 1, max: 2 },
  max_video_frames: { label: "Frames per video", min: 1, max: 16 },
  max_media_item_mb: { label: "Media item (MB)", min: 1, max: 25 },
  max_request_mb: { label: "Request body (MB)", min: 1, max: 256 },
  media_pass_budget_mb: { label: "VRAM budget per pass (MB)", min: 200, max: 4096, step: 100 },
  max_concurrent_media_requests: { label: "Concurrent media requests", min: 1, max: 8 },
};

const MEDIA_LIMIT_KEYS: NumberKey[] = [
  "max_images_per_input",
  "max_images_per_request",
  "max_image_megapixels",
  "max_audio_clips_per_input",
  "max_audio_seconds_per_clip",
  "max_audio_seconds_per_request",
  "max_videos_per_request",
  "max_video_frames",
  "max_media_item_mb",
  "max_request_mb",
  "media_pass_budget_mb",
  "max_concurrent_media_requests",
];

const MEDIA_TOGGLES: Array<{ key: "image_inputs_enabled" | "audio_inputs_enabled" | "video_inputs_enabled"; label: string }> = [
  { key: "image_inputs_enabled", label: "Allow images" },
  { key: "audio_inputs_enabled", label: "Allow audio" },
  { key: "video_inputs_enabled", label: "Allow video (frames)" },
];

const SETTINGS_KEYS = (Object.keys(emptySettings) as Array<keyof AdminSettings>).filter((key) => key !== "huggingface_token");

function fieldError(key: NumberKey, value: number | ""): string | null {
  const spec = NUMBER_FIELDS[key];
  if (value === "") return "Required.";
  if (!Number.isInteger(value)) return "Whole numbers only.";
  if (value < spec.min || value > spec.max) return `Must be between ${fmtNum(spec.min)} and ${fmtNum(spec.max)}.`;
  return null;
}

function dirtySettingKeys(form: SettingsForm, saved: AdminSettings) {
  return SETTINGS_KEYS.filter((key) => form[key] !== saved[key]);
}

const emptyOptions: SettingsResponse["options"] = {
  models: [],
  execution_targets: [],
  gpu_devices: [],
  backends: [],
  precisions: [],
  quantizations: [],
  too_long_policies: [],
  compile_modes: [],
};

// ---------- small view helpers ----------
type HistoryEntry = {
  timestamp?: string;
  route?: string;
  model_id?: string;
  dimension?: number;
  input_count?: number;
  input_chars?: number;
  input_tokens?: number | null;
  media?: MediaCounts | null;
  total_duration_ms?: number;
  encode_duration_ms?: number;
  batched?: boolean;
  success?: boolean;
  error?: string;
};

type LoadedModel = {
  model_id?: string;
  backend?: string;
  device?: string;
  precision?: string;
  compile_mode?: string;
  quantization?: string;
  compile?: { effective?: boolean; graphs?: number; reason?: string | null } | null;
  max_seq_length?: number | null;
  loaded_at?: number | null;
  load_duration_ms?: number | null;
};

function optionLabel(options: AdminOption[], value: string | null | undefined) {
  if (!value) return "—";
  return options.find((option) => option.value === value)?.label || value;
}

function modelSize(model: ManagedModel) {
  if (model.size_on_disk_gb != null) return fmtGb(model.size_on_disk_gb);
  if (model.approx_size_gb != null) return `~${fmtGb(model.approx_size_gb)}`;
  return "—";
}

function modelStatusBadge(status: string) {
  if (status === "ready") return <span className="badge ok">Cached</span>;
  if (status === "downloading") return <span className="badge warn">Downloading</span>;
  if (status === "partial") return <span className="badge warn">Partial</span>;
  if (status === "error") return <span className="badge danger">Failed</span>;
  return <span className="badge plain">Not downloaded</span>;
}

function mediaCounts(media: MediaCounts) {
  const parts: string[] = [];
  if (media.images) parts.push(`${fmtNum(media.images)} img`);
  if (media.audio_seconds) parts.push(`${fmtNum(media.audio_seconds, 1)} s audio`);
  if (media.video_frames) parts.push(`${fmtNum(media.video_frames)} frames`);
  return parts.length > 0 ? parts.join(" · ") : "media";
}

function compileText(model: LoadedModel) {
  const status = model.compile;
  if (!status || model.compile_mode === "off") return "Off";
  if (!status.effective) return `Not effective (eager)${status.reason ? `: ${status.reason}` : ""}`;
  return `${model.compile_mode} · ${fmtNum(status.graphs ?? 0)} graphs`;
}

function idleTrimText(idle: RuntimeInfo["idle_trim"]) {
  if (!idle) return "—";
  if (!idle.idle_trim_seconds) return "Off";
  const parts = [`After ${fmtDuration(idle.idle_trim_seconds)}`, `${fmtNum(idle.trims)} trims`];
  if (idle.last_freed_mb != null) parts.push(`last freed ${fmtNum(idle.last_freed_mb)} MB`);
  if (idle.next_trim_in_seconds != null) parts.push(`next in ${fmtDuration(idle.next_trim_in_seconds)}`);
  return parts.join(" · ");
}

function Field({ label, hint, error, dirty, children, wide }: { label: string; hint?: ReactNode; error?: string | null; dirty?: boolean; children: ReactNode; wide?: boolean }) {
  return (
    <label className={`field${dirty ? " field-dirty" : ""}${wide ? " field-wide" : ""}`}>
      <span className="field-label">{label}{dirty ? <span className="dirty-mark" aria-label="changed"> •</span> : null}</span>
      {children}
      {hint ? <span className="hint">{hint}</span> : null}
      {error ? <span className="field-error">{error}</span> : null}
    </label>
  );
}

function Card({ label, value, sub, title }: { label: string; value: ReactNode; sub?: ReactNode; title?: string }) {
  return (
    <div className="card" title={title}>
      <span className="card-label">{label}</span>
      <span className="card-value">{value}</span>
      {sub ? <span className="card-sub">{sub}</span> : null}
    </div>
  );
}

function PanelError({ message, onRetry }: { message: string | null; onRetry: () => void }) {
  if (!message) return null;
  return (
    <div className="banner danger" role="alert">
      <span>{message}</span>
      <button type="button" className="secondary-button small-button" onClick={onRetry}>Retry</button>
    </div>
  );
}

// ---------- app ----------
export default function App() {
  const [authState, setAuthState] = useState<AuthState>("loading");
  const [bootError, setBootError] = useState("");
  const [currentUser, setCurrentUser] = useState("");
  const [sessionExpiresAt, setSessionExpiresAt] = useState<string | null>(null);
  const [setupStatus, setSetupStatus] = useState<SetupStatus | null>(null);
  const [loginUsername, setLoginUsername] = useState("admin");
  const [loginPassword, setLoginPassword] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");

  const [pwCurrent, setPwCurrent] = useState("");
  const [pwNew, setPwNew] = useState("");
  const [pwConfirm, setPwConfirm] = useState("");
  const [pwBusy, setPwBusy] = useState(false);
  const [pwError, setPwError] = useState("");

  const [activeTab, setActiveTab] = useState<TabId>(tabFromHash);
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  // Server state (all tabs read from here, so a tab switch loses nothing).
  const [savedSettings, setSavedSettings] = useState<AdminSettings | null>(null);
  const [settingsForm, setSettingsForm] = useState<SettingsForm>(emptySettings);
  const [settingsOptions, setSettingsOptions] = useState(emptyOptions);
  const [tokenSet, setTokenSet] = useState(false);
  const [tokenHint, setTokenHint] = useState<string | null>(null);
  const [tokenInput, setTokenInput] = useState("");
  const [tokenClear, setTokenClear] = useState(false);
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const [managedModels, setManagedModels] = useState<ManagedModel[] | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [queue, setQueue] = useState<QueueResponse | null>(null);
  const [apiKeys, setApiKeys] = useState<ApiKeyInfo[] | null>(null);
  const [panelErrors, setPanelErrors] = useState<{ settings: string | null; stats: string | null; queue: string | null; keys: string | null }>({
    settings: null,
    stats: null,
    queue: null,
    keys: null,
  });

  const [saveBusy, setSaveBusy] = useState(false);
  const [presets, setPresets] = useState<SettingsPreset[]>([]);
  const [presetChoice, setPresetChoice] = useState("");
  const [presetSaveName, setPresetSaveName] = useState("");
  const [presetBusy, setPresetBusy] = useState(false);

  const [modelBusy, setModelBusy] = useState<{ id: string; kind: "download" | "delete" } | null>(null);
  const [modelsRefreshing, setModelsRefreshing] = useState(false);

  // "" = follow the saved default model.
  const [compareModel, setCompareModel] = useState("");
  const [compareTextA, setCompareTextA] = useState("Genesis stores memories as vectors.");
  const [compareTextB, setCompareTextB] = useState("A vector database can compare semantic meaning.");
  const [compareBusy, setCompareBusy] = useState(false);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [benchmarkModel, setBenchmarkModel] = useState("");
  const [benchmarkInputs, setBenchmarkInputs] = useState("Text one\nText two\nText three");
  const [benchmarkRepeatCount, setBenchmarkRepeatCount] = useState<number | "">(3);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(null);

  const isReady = authState === "ready";
  const saved = savedSettings ?? emptySettings;
  const savedRef = useRef(saved);
  const formRef = useRef(settingsForm);
  savedRef.current = saved;
  formRef.current = settingsForm;

  const dirtyKeys = savedSettings ? dirtySettingKeys(settingsForm, savedSettings) : [];
  const tokenDirty = tokenInput.trim() !== "" || tokenClear;
  const dirtyCount = dirtyKeys.length + (tokenDirty ? 1 : 0);
  const invalidKeys = (Object.keys(NUMBER_FIELDS) as NumberKey[]).filter((key) => fieldError(key, settingsForm[key]) !== null);
  const defaultModel = saved.default_model;
  const hasDownloadingModels = (managedModels ?? []).some((model) => model.status === "downloading");

  // ---------- tabs ----------
  useEffect(() => {
    const handler = () => setActiveTab(tabFromHash());
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);

  function selectTab(tab: TabId, focus = false) {
    if (window.location.hash !== `#${tab}`) window.location.hash = tab;
    setActiveTab(tab);
    if (focus) tabRefs.current[tab]?.focus();
  }

  function onTabKey(event: KeyboardEvent<HTMLElement>) {
    const index = TABS.findIndex((tab) => tab.id === activeTab);
    let next = -1;
    if (event.key === "ArrowRight") next = (index + 1) % TABS.length;
    else if (event.key === "ArrowLeft") next = (index - 1 + TABS.length) % TABS.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = TABS.length - 1;
    if (next >= 0) {
      event.preventDefault();
      selectTab(TABS[next].id, true);
    }
  }

  // ---------- session ----------
  const resetState = useCallback(() => {
    setSavedSettings(null);
    setSettingsForm(emptySettings);
    setSettingsOptions(emptyOptions);
    setTokenSet(false);
    setTokenHint(null);
    setTokenInput("");
    setTokenClear(false);
    setRuntime(null);
    setManagedModels(null);
    setStats(null);
    setQueue(null);
    setApiKeys(null);
    setPanelErrors({ settings: null, stats: null, queue: null, keys: null });
    setPresets([]);
    setPresetChoice("");
    setPresetSaveName("");
    setCompareResult(null);
    setBenchmarkResult(null);
    setPwCurrent("");
    setPwNew("");
    setPwConfirm("");
    setPwError("");
    clearPerfHistory();
    toast.clear();
  }, []);

  /** true when the error was a lost session (handled: back to login) or a forced password change. */
  const handleAuthError = useCallback(
    (error: unknown): boolean => {
      const message = error instanceof Error ? error.message : "";
      if (message === "unauthorized") {
        resetState();
        setAuthState("login");
        toast.warn("Session expired. Please log in again.");
        return true;
      }
      if (message === "password_change_required") {
        setAuthState("change");
        return true;
      }
      return false;
    },
    [resetState],
  );

  const bootstrap = useCallback(async () => {
    setAuthState("loading");
    setBootError("");
    try {
      const me = await whoami();
      setCurrentUser(me.username);
      setSessionExpiresAt(me.session_expires_at ?? null);
      setAuthState(me.must_change_password ? "change" : "ready");
    } catch (error) {
      const message = errorText(error, "");
      if (message === "unauthorized") {
        setAuthState("login");
      } else if (message === "password_change_required") {
        setAuthState("change");
      } else {
        setBootError(message || "The server did not answer.");
        setAuthState("error");
      }
    }
  }, []);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  // The login page shows the default-credentials hint only while first-login setup is pending.
  useEffect(() => {
    if (authState !== "login") return;
    let cancelled = false;
    getSetupStatus()
      .then((status) => !cancelled && setSetupStatus(status))
      .catch(() => !cancelled && setSetupStatus(null));
    return () => {
      cancelled = true;
    };
  }, [authState]);

  // ---------- data ----------
  const applySettingsResponse = useCallback((payload: SettingsResponse, replaceForm: boolean) => {
    const next = { ...emptySettings, ...payload.settings };
    const form = formRef.current;
    const edits = replaceForm ? {} : Object.fromEntries(dirtySettingKeys(form, savedRef.current).map((key) => [key, form[key]]));
    setSavedSettings(next);
    // Polling never overwrites unsaved edits: they are laid over the fresh values.
    setSettingsForm({ ...next, ...edits });
    setSettingsOptions({ ...emptyOptions, ...payload.options });
    setTokenSet(Boolean(payload.huggingface_token_set ?? payload.settings.huggingface_token));
    setTokenHint(payload.huggingface_token_hint ?? (payload.settings.huggingface_token || null));
    setManagedModels(payload.models ?? []);
    setRuntime(payload.runtime);
  }, []);

  const refreshCore = useCallback(
    async (signal?: AbortSignal) => {
      const [settingsResult, statsResult, queueResult] = await Promise.allSettled([getSettings(signal), getStats(signal), getQueue(signal)]);
      if (signal?.aborted) return;
      for (const result of [settingsResult, statsResult, queueResult]) {
        if (result.status === "rejected" && handleAuthError(result.reason)) return;
      }
      if (settingsResult.status === "fulfilled") applySettingsResponse(settingsResult.value, false);
      if (statsResult.status === "fulfilled") setStats(statsResult.value);
      if (queueResult.status === "fulfilled") setQueue(queueResult.value);
      const reason = (result: PromiseSettledResult<unknown>, what: string) =>
        result.status === "rejected" ? `Could not load ${what}: ${errorText(result.reason, "request failed")}` : null;
      setPanelErrors((current) => ({
        ...current,
        settings: reason(settingsResult, "settings and runtime status"),
        stats: reason(statsResult, "request statistics"),
        queue: reason(queueResult, "the batch queue"),
      }));
    },
    [applySettingsResponse, handleAuthError],
  );

  const refreshKeys = useCallback(
    async (signal?: AbortSignal) => {
      try {
        const response = await listApiKeys(signal);
        if (signal?.aborted) return;
        setApiKeys(response.keys);
        setPanelErrors((current) => ({ ...current, keys: null }));
      } catch (error) {
        if (signal?.aborted || handleAuthError(error)) return;
        setPanelErrors((current) => ({ ...current, keys: errorText(error, "request failed") }));
      }
    },
    [handleAuthError],
  );

  const refreshModelsQuiet = useCallback(
    async (signal?: AbortSignal) => {
      try {
        const response = await getModels(signal);
        if (!signal?.aborted) setManagedModels(response.models ?? []);
      } catch (error) {
        if (!signal?.aborted && !handleAuthError(error)) toast.error(`Could not refresh the model list: ${errorText(error, "request failed")}`);
      }
    },
    [handleAuthError],
  );

  const reloadPresets = useCallback(async () => {
    try {
      setPresets((await listPresets()).presets);
    } catch (error) {
      if (!handleAuthError(error)) toast.error(`Could not load presets: ${errorText(error, "request failed")}`);
    }
  }, [handleAuthError]);

  useVisiblePolling(refreshCore, POLL_MS, isReady);
  useVisiblePolling(refreshKeys, activeTab === "api-keys" ? POLL_MS : KEYS_BACKGROUND_POLL_MS, isReady);
  useVisiblePolling(refreshModelsQuiet, MODEL_STATUS_POLL_MS, isReady && hasDownloadingModels);
  usePerfFeed((signal) => getPerf(signal), isReady);

  useEffect(() => {
    if (isReady) void reloadPresets();
  }, [isReady, reloadPresets]);

  // ---------- auth actions ----------
  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthBusy(true);
    setAuthError("");
    try {
      const me = await login(loginUsername.trim(), loginPassword);
      setCurrentUser(me.username);
      setLoginPassword("");
      toast.clear();
      if (me.must_change_password) {
        setPwCurrent(loginPassword);
        setAuthState("change");
      } else {
        const info = await whoami().catch(() => null);
        setSessionExpiresAt(info?.session_expires_at ?? null);
        setAuthState("ready");
      }
    } catch (error) {
      const message = errorText(error, "Login failed.");
      setAuthError(message === "unauthorized" ? "Invalid username or password." : message);
    } finally {
      setAuthBusy(false);
    }
  }

  async function handleChangePassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPwError("");
    if (pwNew.length < MIN_PASSWORD_LENGTH) {
      setPwError(`The new password must be at least ${MIN_PASSWORD_LENGTH} characters.`);
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
      const info = await whoami().catch(() => null);
      setSessionExpiresAt(info?.session_expires_at ?? null);
      setAuthState("ready");
      toast.ok("Password changed.");
    } catch (error) {
      if (errorText(error, "") === "unauthorized") {
        handleAuthError(error);
        return;
      }
      setPwError(`Could not change the password: ${errorText(error, "request failed")}`);
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
    resetState();
    setLoginPassword("");
    setAuthState("login");
  }

  // ---------- settings actions ----------
  function updateSetting<K extends keyof SettingsForm>(key: K, value: SettingsForm[K]) {
    setSettingsForm((current) => ({ ...current, [key]: value }));
  }

  function discardSettings() {
    setSettingsForm(saved);
    setTokenInput("");
    setTokenClear(false);
  }

  async function reloadSettings() {
    try {
      const response = await getSettings();
      applySettingsResponse(response, true);
      setTokenInput("");
      setTokenClear(false);
      toast.info("Settings reloaded from the server.");
    } catch (error) {
      if (!handleAuthError(error)) toast.error(`Could not reload settings: ${errorText(error, "request failed")}`);
    }
  }

  async function handleSaveSettings(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    if (invalidKeys.length > 0 || dirtyCount === 0) return;
    const payload: SettingsUpdate = {};
    for (const key of SETTINGS_KEYS) {
      const value = settingsForm[key];
      if (value !== "") (payload as Record<string, unknown>)[key] = value;
    }
    if (tokenClear) payload.clear_huggingface_token = true;
    else if (tokenInput.trim()) payload.huggingface_token = tokenInput.trim();
    setSaveBusy(true);
    try {
      const response = await saveSettings(payload);
      const adjusted = SETTINGS_KEYS.filter((key) => key in payload && (payload as Record<string, unknown>)[key] !== response.settings[key]);
      applySettingsResponse(response, true);
      setTokenInput("");
      setTokenClear(false);
      if (adjusted.length > 0) {
        toast.warn(`Settings saved. The server adjusted: ${adjusted.map((key) => `${key} → ${String(response.settings[key])}`).join(", ")}.`);
      } else {
        toast.ok("Settings saved.");
      }
    } catch (error) {
      if (!handleAuthError(error)) toast.error(`Could not save settings: ${errorText(error, "request failed")}`);
    } finally {
      setSaveBusy(false);
    }
  }

  const selectedPreset = presets.find((preset) => preset.name === presetChoice);

  function handleLoadPreset() {
    if (!selectedPreset) return;
    setSettingsForm((current) => ({ ...current, ...selectedPreset.values }));
    toast.info(`Preset "${selectedPreset.name}" loaded into the form. Review it, then save the changes.${selectedPreset.note ? ` ${selectedPreset.note}` : ""}`);
  }

  async function handleSavePreset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const name = presetSaveName.trim();
    if (!name) return;
    const builtIn = presets.find((preset) => preset.name === name && preset.built_in);
    if (builtIn) {
      toast.error(`"${name}" is a built-in preset and cannot be overwritten. Choose another name.`);
      return;
    }
    if (presets.some((preset) => preset.name === name) && !window.confirm(`Overwrite preset "${name}"?\n\nIts saved values are replaced by the current form.`)) {
      return;
    }
    const values = Object.fromEntries(SETTINGS_KEYS.filter((key) => settingsForm[key] !== "").map((key) => [key, settingsForm[key]])) as Partial<AdminSettings>;
    setPresetBusy(true);
    try {
      const response = await savePreset(name, values);
      await reloadPresets();
      setPresetChoice(response.preset.name);
      setPresetSaveName("");
      toast.ok(`Preset "${response.preset.name}" saved from the current form.`);
    } catch (error) {
      if (!handleAuthError(error)) toast.error(`Could not save the preset: ${errorText(error, "request failed")}`);
    } finally {
      setPresetBusy(false);
    }
  }

  async function handleDeletePreset() {
    if (!selectedPreset || selectedPreset.built_in) return;
    const name = selectedPreset.name;
    if (!window.confirm(`Delete preset "${name}"?\n\nThe saved values are removed. The current settings do not change.`)) return;
    setPresetBusy(true);
    try {
      await deletePreset(name);
      setPresetChoice("");
      await reloadPresets();
      toast.ok(`Preset "${name}" deleted.`);
    } catch (error) {
      if (!handleAuthError(error)) toast.error(`Could not delete the preset: ${errorText(error, "request failed")}`);
    } finally {
      setPresetBusy(false);
    }
  }

  // ---------- model actions ----------
  async function handleRefreshModels() {
    setModelsRefreshing(true);
    await refreshModelsQuiet();
    setModelsRefreshing(false);
  }

  async function handleDownloadModel(model: ManagedModel) {
    setModelBusy({ id: model.id, kind: "download" });
    try {
      const response = await downloadModel(model.id);
      setManagedModels(response.models ?? []);
      toast.info(`Download of "${model.label}" started.`);
    } catch (error) {
      if (!handleAuthError(error)) toast.error(`Could not download "${model.label}": ${errorText(error, "request failed")}`);
    } finally {
      setModelBusy(null);
    }
  }

  async function handleDeleteModel(model: ManagedModel) {
    const loaded = model.id === defaultModel ? "\n\nIt is the default model: the next request downloads it again." : "";
    if (!window.confirm(`Delete cached model "${model.id}"?\n\nIt must be downloaded again before it can be loaded.${loaded}`)) return;
    setModelBusy({ id: model.id, kind: "delete" });
    try {
      const response = await deleteModel(model.id);
      setManagedModels(response.models ?? []);
      toast.ok(response.removed ? `Cached model "${model.label}" deleted.` : `"${model.label}" was not in the cache.`);
    } catch (error) {
      if (!handleAuthError(error)) toast.error(`Could not delete "${model.label}": ${errorText(error, "request failed")}`);
    } finally {
      setModelBusy(null);
    }
  }

  // ---------- playground + benchmark ----------
  async function handleCompare(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setCompareBusy(true);
    try {
      const result = await compareTexts(compareModel || defaultModel, compareTextA, compareTextB);
      setCompareResult(result);
    } catch (error) {
      if (!handleAuthError(error)) toast.error(`Compare failed: ${errorText(error, "request failed")}`);
    } finally {
      setCompareBusy(false);
    }
  }

  async function handleBenchmark(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const inputs = benchmarkInputs.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (inputs.length === 0) {
      toast.error("Add at least one input line.");
      return;
    }
    if (benchmarkRepeatCount === "" || benchmarkRepeatCount < 1 || benchmarkRepeatCount > 128) {
      toast.error("Repeats must be between 1 and 128.");
      return;
    }
    setBenchmarkBusy(true);
    try {
      const result = await runBenchmark(benchmarkModel || defaultModel, inputs, benchmarkRepeatCount);
      setBenchmarkResult(result);
    } catch (error) {
      if (!handleAuthError(error)) toast.error(`Benchmark failed: ${errorText(error, "request failed")}`);
    } finally {
      setBenchmarkBusy(false);
    }
  }

  // ---------- derived view data ----------
  const perf = usePerfLatest();
  const serviceGpus = (perf?.gpus ?? []).filter((gpu) => gpu.used_by_service);
  const loadedModels = (runtime?.loaded_models ?? []) as LoadedModel[];
  const lastError = runtime?.last_runtime_error ?? null;
  const defaultLoaded = loadedModels.some((model) => model.model_id === defaultModel);
  const history = (stats?.history ?? []) as HistoryEntry[];
  const keyCount = apiKeys?.length ?? 0;
  const modelLabel = optionLabel(settingsOptions.models, defaultModel);

  let stateBadge: ReactNode = <span className="badge warn">Loading</span>;
  if (runtime) {
    if (lastError && loadedModels.length === 0) stateBadge = <span className="badge danger" title={lastError}>Error</span>;
    else if (defaultLoaded) stateBadge = <span className="badge ok">Ready</span>;
    else if (loadedModels.length > 0) stateBadge = <span className="badge ok" title="Another model is loaded; the default loads on first use.">Ready</span>;
    else stateBadge = <span className="badge plain">Idle · model unloaded</span>;
  }

  // ---------- render: pre-login states ----------
  if (authState === "loading") {
    return (
      <>
        <ToastHost />
        <main className="login-shell">
          <div className="state" role="status"><span className="spinner" />Loading…</div>
        </main>
      </>
    );
  }

  if (authState === "error") {
    return (
      <>
        <ToastHost />
        <main className="login-shell">
          <section className="panel login-panel">
            <p className="eyebrow">G3 GPU service · Admin</p>
            <h1>{SERVICE_NAME}</h1>
            <div className="banner danger" role="alert">
              <span>Could not reach the server: {bootError}</span>
              <button type="button" className="secondary-button small-button" onClick={() => void bootstrap()}>Retry</button>
            </div>
          </section>
        </main>
      </>
    );
  }

  if (authState === "login") {
    return (
      <>
        <ToastHost />
        <main className="login-shell">
          <section className="panel login-panel" aria-labelledby="login-title">
            <p className="eyebrow">G3 GPU service · Admin</p>
            <h1 id="login-title">{SERVICE_NAME}</h1>
            <p className="dim">Log in to manage models, settings and API keys.</p>
            <form className="form-stack" onSubmit={handleLogin}>
              <Field label="Username">
                <input value={loginUsername} autoComplete="username" onChange={(event) => setLoginUsername(event.target.value)} />
              </Field>
              <Field label="Password">
                <input type="password" value={loginPassword} autoComplete="current-password" onChange={(event) => setLoginPassword(event.target.value)} />
              </Field>
              {authError ? <div className="banner danger" role="alert">{authError}</div> : null}
              {setupStatus?.setup_required ? (
                <p className="hint">First start: log in as "{setupStatus.default_username}" with the password "admin". You set a new password right after.</p>
              ) : null}
              <div className="form-actions">
                <button type="submit" disabled={authBusy || !loginUsername.trim() || !loginPassword} aria-busy={authBusy}>
                  {authBusy ? "Logging in…" : "Log in"}
                </button>
              </div>
            </form>
          </section>
        </main>
      </>
    );
  }

  const passwordForm = (submitLabel: string) => (
    <form className="form-stack" onSubmit={handleChangePassword}>
      <input type="text" className="sr-only" autoComplete="username" value={currentUser} readOnly tabIndex={-1} aria-hidden="true" />
      <Field label="Current password">
        <input type="password" value={pwCurrent} autoComplete="current-password" onChange={(event) => setPwCurrent(event.target.value)} />
      </Field>
      <Field label="New password" hint={`At least ${MIN_PASSWORD_LENGTH} characters.`}>
        <input type="password" value={pwNew} autoComplete="new-password" minLength={MIN_PASSWORD_LENGTH} onChange={(event) => setPwNew(event.target.value)} />
      </Field>
      <Field label="Confirm new password" error={pwConfirm && pwNew !== pwConfirm ? "Does not match the new password." : null}>
        <input type="password" value={pwConfirm} autoComplete="new-password" aria-invalid={Boolean(pwConfirm) && pwNew !== pwConfirm} onChange={(event) => setPwConfirm(event.target.value)} />
      </Field>
      {pwError ? <div className="banner danger" role="alert">{pwError}</div> : null}
      <div className="form-actions">
        <button type="submit" disabled={pwBusy || !pwCurrent || !pwNew || !pwConfirm} aria-busy={pwBusy}>{pwBusy ? "Saving…" : submitLabel}</button>
      </div>
    </form>
  );

  if (authState === "change") {
    return (
      <>
        <ToastHost />
        <main className="login-shell">
          <section className="panel login-panel" aria-labelledby="change-title">
            <p className="eyebrow">G3 GPU service · Admin</p>
            <h1 id="change-title">Set a new password</h1>
            <p className="dim">Signed in as <strong>{currentUser || "admin"}</strong>. Choose your own password before you continue.</p>
            {passwordForm("Change password")}
            <button type="button" className="ghost-button" onClick={() => void handleLogout()}>Log out</button>
          </section>
        </main>
      </>
    );
  }

  // ---------- render: tabs ----------
  const dashboard = (
    <>
      <div className="grid grid-auto card-row">
        <Card label="Model" value={<span className="card-value-text" title={defaultModel}>{modelLabel}</span>} sub={<span className="mono">{defaultModel || "—"}</span>} />
        <Card label="State" value={stateBadge} sub={queue ? `${fmtNum(queue.queue_size ?? 0)} waiting in the queue` : "—"} />
        <Card
          label="Requests"
          value={fmtNum(stats?.summary.requests_since_start ?? null)}
          sub={stats?.summary.started_at ? `since ${fmtDateTime(stats.summary.started_at)}` : "since start"}
        />
        <Card
          label="Errors"
          value={fmtNum(stats?.summary.errors_since_start ?? null)}
          sub={stats?.summary.avg_total_duration_ms != null ? `avg. request ${fmtMs(stats.summary.avg_total_duration_ms)}` : "since start"}
        />
      </div>
      <PanelError message={panelErrors.stats} onRetry={() => void refreshCore()} />

      <PerfPanel />

      <section className="panel" aria-labelledby="queue-title">
        <div className="panel-head">
          <h2 id="queue-title">Batch queue</h2>
          {queue ? <span className={`badge ${queue.worker_running ? "ok" : "warn"}`}>{queue.worker_running ? "Worker running" : "Worker stopped"}</span> : null}
          <p className="hint">Requests are collected into batches when batching is on. The batch values describe the latest batch only.</p>
        </div>
        <PanelError message={panelErrors.queue} onRetry={() => void refreshCore()} />
        <div className="grid grid-auto card-row">
          <Card label="Waiting" value={fmtNum(queue?.queue_size ?? null)} sub="requests in the queue" />
          <Card label="Active batch" value={queue?.active_batch_id ? fmtNum(queue.active_batch_size ?? null) : "None"} sub={queue?.active_batch_id ? `${fmtNum(queue.active_batch_chars ?? null)} characters` : "no batch running"} />
          <Card label="Latest batch" value={fmtMs(queue?.last_batch_duration_ms)} sub={queue?.last_batch_texts_per_second != null ? `${fmtRate(queue.last_batch_texts_per_second, "texts/s")} · ${fmtDateTime(queue.last_batch_completed_at, true)}` : "—"} />
          <Card label="Processed" value={fmtNum(queue?.total_texts_processed ?? null)} sub={`texts in ${fmtNum(queue?.total_batches_processed ?? null)} batches since start`} />
        </div>
        {queue?.last_error ? <div className="banner danger">Last batch failed: {queue.last_error}</div> : null}
      </section>

      <section className="panel" aria-labelledby="system-title">
        <div className="panel-head">
          <h2 id="system-title">System</h2>
          <p className="hint">Runtime routing, loaded models and detected hardware.</p>
        </div>
        <PanelError message={panelErrors.settings} onRetry={() => void refreshCore()} />
        <dl className="kv">
          <dt>Runtime</dt>
          <dd>{runtime ? `${runtime.resolved_runtime.backend} · ${runtime.resolved_runtime.device_label}` : "—"}<span className="hint"> {runtime?.resolved_runtime.detail ?? ""}</span></dd>
          <dt>GPU in use</dt>
          <dd>{perf ? (serviceGpus.length ? serviceGpus.map(gpuTitle).join(", ") : "CPU") : "—"}</dd>
          <dt>NVIDIA cards on host</dt>
          <dd>{perf && perf.gpus.length ? perf.gpus.map(gpuTitle).join(", ") : runtime?.system.nvidia_gpus?.join(", ") || "—"}</dd>
          <dt>OpenVINO</dt>
          <dd>{runtime?.system.openvino_devices?.join(", ") || "—"}</dd>
          <dt>Platform</dt>
          <dd>{runtime?.system.platform ?? "—"}</dd>
          <dt>Idle trim</dt>
          <dd>{idleTrimText(runtime?.idle_trim)}</dd>
          <dt>Last runtime error</dt>
          <dd className={lastError ? "text-danger" : undefined}>{lastError ?? "None"}</dd>
        </dl>
        <h3>Loaded models</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Model</th><th>Device</th><th>Precision</th><th>Compile</th><th className="num">Max tokens</th><th className="num">Load time</th></tr>
            </thead>
            <tbody>
              {loadedModels.length === 0 ? (
                <tr><td colSpan={6} className="empty">No model loaded. The default model loads on the first request.</td></tr>
              ) : (
                loadedModels.map((model, index) => (
                  <tr key={`${model.model_id ?? "model"}-${model.device ?? index}-${model.precision ?? ""}`}>
                    <td className="wrap"><span className="mono">{model.model_id ?? "—"}</span>{model.model_id === defaultModel ? <span className="badge accent inline-badge">Default</span> : null}</td>
                    <td>{[model.backend, model.device].filter(Boolean).join(" · ") || "—"}</td>
                    <td>{[model.precision, model.quantization && model.quantization !== "none" ? model.quantization : null].filter(Boolean).join(" · ") || "—"}</td>
                    <td className="wrap">{compileText(model)}</td>
                    <td className="num">{fmtNum(model.max_seq_length ?? null)}</td>
                    <td className="num">{fmtMs(model.load_duration_ms)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );

  const modelSelect = (value: string, onChange: (value: string) => void, id: string) => (
    <select id={id} value={value || defaultModel} onChange={(event) => onChange(event.target.value === defaultModel ? "" : event.target.value)}>
      {settingsOptions.models.map((option) => (
        <option key={option.value} value={option.value}>{option.label}{option.value === defaultModel ? " (default)" : ""}</option>
      ))}
    </select>
  );

  const playground = (
    <section className="panel" aria-labelledby="compare-title">
      <div className="panel-head">
        <h2 id="compare-title">Compare two texts</h2>
        <p className="hint">Embeds both texts with the chosen model and compares the vectors. Runs through the same queue as API requests and counts in the history.</p>
      </div>
      <form className="form-grid" onSubmit={handleCompare}>
        <Field label="Model" hint={runtime ? `Runs on ${runtime.resolved_runtime.device_label}.` : undefined}>
          {modelSelect(compareModel, setCompareModel, "compare-model")}
        </Field>
        <Field label="Text A" wide>
          <textarea value={compareTextA} onChange={(event) => setCompareTextA(event.target.value)} rows={5} />
        </Field>
        <Field label="Text B" wide>
          <textarea value={compareTextB} onChange={(event) => setCompareTextB(event.target.value)} rows={5} />
        </Field>
        <div className="form-actions field-wide">
          <button type="submit" disabled={compareBusy || !compareTextA.trim() || !compareTextB.trim()} aria-busy={compareBusy}>
            {compareBusy ? "Embedding…" : "Run"}
          </button>
          {compareResult ? <CopyButton text={JSON.stringify(compareResult.vectors)} label="Copy vectors" className="secondary-button" /> : null}
        </div>
      </form>
      {compareResult ? (
        <div className="grid grid-auto card-row" aria-live="polite">
          <Card label="Cosine similarity" value={fmtFixed(compareResult.cosine_similarity, 4)} />
          <Card label="Dot product" value={fmtFixed(compareResult.dot_product, 4)} />
          <Card label="Euclidean distance" value={fmtFixed(compareResult.euclidean_distance, 4)} />
          <Card label="Dimensions" value={fmtNum(compareResult.dimension)} sub={<span className="mono">{compareResult.model_id}</span>} />
          <Card label="Total time" value={fmtMs(compareResult.total_duration_ms)} sub={`encode ${fmtMs(compareResult.encode_duration_ms)} · load ${fmtMs(compareResult.load_duration_ms)}`} />
          <Card label="Runtime" value={compareResult.device_label || compareResult.device} sub={`${compareResult.backend} · ${compareResult.batched ? "batched" : "direct"}`} />
        </div>
      ) : null}
    </section>
  );

  const cachePathDirty = settingsForm.model_cache_path !== saved.model_cache_path;
  const models = (
    <section className="panel" aria-labelledby="models-title">
      <div className="panel-head">
        <h2 id="models-title">Model cache</h2>
        <button type="button" className="secondary-button" onClick={() => void handleRefreshModels()} disabled={modelsRefreshing} aria-busy={modelsRefreshing}>
          {modelsRefreshing ? "Refreshing…" : "Refresh"}
        </button>
        <p className="hint">
          Download or remove supported models in the saved cache path <span className="mono">{saved.model_cache_path || "(Hugging Face default)"}</span>.
          Downloads use the saved Hugging Face token.
        </p>
      </div>
      {cachePathDirty ? <div className="banner warn">The cache path has unsaved changes in Settings. This list shows the saved path until you save.</div> : null}
      <PanelError message={panelErrors.settings} onRetry={() => void refreshCore()} />
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Model</th><th>Tier</th><th className="num">Size</th><th>Status</th><th>Path</th><th className="actions"><span className="sr-only">Actions</span></th></tr>
          </thead>
          <tbody>
            {managedModels === null ? <tr><td colSpan={6} className="empty"><span className="spinner" /> Loading…</td></tr> : null}
            {managedModels !== null && managedModels.length === 0 ? <tr><td colSpan={6} className="empty">No supported models found.</td></tr> : null}
            {(managedModels ?? []).map((model) => {
              const busy = modelBusy?.id === model.id;
              const downloading = model.status === "downloading";
              return (
                <tr key={model.id}>
                  <td className="wrap model-cell">
                    <strong>{model.label}</strong>{model.id === defaultModel ? <span className="badge accent inline-badge">Default</span> : null}
                    <div className="mono dim">{model.id}</div>
                    {model.notes ? <div className="hint">{model.notes}</div> : null}
                  </td>
                  <td>{model.tier || "—"}</td>
                  <td className="num">{modelSize(model)}</td>
                  <td>
                    {modelStatusBadge(model.status)}
                    {model.error ? <div className="field-error">{model.error}</div> : null}
                  </td>
                  <td className="wrap"><span className="mono dim">{model.local_path || model.cache_path || model.storage_root || "—"}</span></td>
                  <td className="actions">
                    <button type="button" className="secondary-button" onClick={() => void handleDownloadModel(model)} disabled={downloading || busy} aria-busy={busy && modelBusy?.kind === "download"}>
                      {downloading ? "Downloading…" : busy && modelBusy?.kind === "download" ? "Starting…" : model.status === "ready" ? "Download again" : "Download"}
                    </button>
                    <button type="button" className="danger-button" onClick={() => void handleDeleteModel(model)} disabled={!model.cache_path || downloading || busy} aria-busy={busy && modelBusy?.kind === "delete"}>
                      {busy && modelBusy?.kind === "delete" ? "Deleting…" : "Delete"}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );

  const isDirty = (key: keyof AdminSettings) => (dirtyKeys as Array<keyof AdminSettings>).includes(key);
  const selectField = (key: "default_model" | "execution_target" | "gpu_device" | "backend_override" | "precision" | "quantization" | "compile_mode" | "too_long_policy", label: string, options: AdminOption[], hint?: string) => (
    <Field label={label} hint={hint} dirty={isDirty(key)}>
      <select value={settingsForm[key]} onChange={(event) => updateSetting(key, event.target.value)}>
        {!options.some((option) => option.value === settingsForm[key]) ? <option value={settingsForm[key]}>{settingsForm[key] || "—"}</option> : null}
        {options.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
    </Field>
  );
  const numberField = (key: NumberKey) => {
    const spec = NUMBER_FIELDS[key];
    const error = fieldError(key, settingsForm[key]);
    return (
      <Field key={key} label={spec.label} hint={`${spec.hint ? `${spec.hint} ` : ""}${fmtNum(spec.min)}–${fmtNum(spec.max)}.`} error={error} dirty={isDirty(key)}>
        <input
          type="number"
          inputMode="numeric"
          min={spec.min}
          max={spec.max}
          step={spec.step ?? 1}
          value={settingsForm[key]}
          aria-invalid={error ? true : undefined}
          onChange={(event) => updateSetting(key, event.target.value === "" ? "" : Number(event.target.value))}
        />
      </Field>
    );
  };
  const toggleField = (key: "batching_enabled" | "warmup_on_load" | "encode_token_batching" | "cuda_graphs" | "image_inputs_enabled" | "audio_inputs_enabled" | "video_inputs_enabled", label: string, hint?: string) => (
    <label className={`switch-row${isDirty(key) ? " field-dirty" : ""}`} key={key}>
      <input type="checkbox" checked={settingsForm[key]} onChange={(event) => updateSetting(key, event.target.checked)} />
      <span>
        <span className="field-label">{label}</span>
        {hint ? <span className="hint"> {hint}</span> : null}
      </span>
    </label>
  );

  const settingsTab = (
    <>
      <section className="panel" aria-labelledby="presets-title">
        <div className="panel-head">
          <h2 id="presets-title">Presets</h2>
          <p className="hint">Built-in GPU tiers plus your saved presets. Loading fills the form below; nothing changes until you save.</p>
        </div>
        <div className="form-grid">
          <Field label="Preset" hint={selectedPreset ? [selectedPreset.description, selectedPreset.note].filter(Boolean).join(" ") || undefined : undefined}>
            <select value={selectedPreset ? presetChoice : ""} onChange={(event) => setPresetChoice(event.target.value)}>
              <option value="">Choose a preset…</option>
              {presets.map((preset) => (
                <option key={preset.name} value={preset.name}>{preset.built_in ? `${preset.name} (built-in)` : preset.name}</option>
              ))}
            </select>
          </Field>
          <div className="button-row align-end">
            <button type="button" className="secondary-button" onClick={handleLoadPreset} disabled={!selectedPreset || presetBusy || saveBusy}>Load into form</button>
            <button type="button" className="danger-button" onClick={() => void handleDeletePreset()} disabled={!selectedPreset || selectedPreset.built_in || presetBusy}>Delete</button>
          </div>
        </div>
        <form className="form-grid" onSubmit={handleSavePreset}>
          <Field label="Save the current form as" hint="The cache path and the Hugging Face token are never stored in a preset.">
            <input value={presetSaveName} maxLength={64} placeholder="Preset name" onChange={(event) => setPresetSaveName(event.target.value)} />
          </Field>
          <div className="button-row align-end">
            <button type="submit" className="secondary-button" disabled={!presetSaveName.trim() || presetBusy || invalidKeys.length > 0} aria-busy={presetBusy}>
              {presetBusy ? "Saving…" : "Save as preset"}
            </button>
          </div>
        </form>
      </section>

      <form className="settings-form" onSubmit={(event) => void handleSaveSettings(event)}>
        <PanelError message={panelErrors.settings} onRetry={() => void refreshCore()} />
        <section className="panel" aria-labelledby="settings-runtime">
          <div className="panel-head"><h2 id="settings-runtime">Model and hardware</h2></div>
          <div className="form-grid">
            {selectField("default_model", "Default model", settingsOptions.models, "Used when a request names no model.")}
            {selectField("execution_target", "Target", settingsOptions.execution_targets)}
            {selectField("gpu_device", "GPU", settingsOptions.gpu_devices, "Numbered like nvidia-smi. Switching reloads the model on the new card; the old card keeps the CUDA context (about 0.5 GB) until restart.")}
            {selectField("backend_override", "Backend", settingsOptions.backends)}
            {selectField("precision", "Precision", settingsOptions.precisions)}
            {selectField("quantization", "Quantization (CUDA)", settingsOptions.quantizations)}
            {selectField("compile_mode", "Compile", settingsOptions.compile_modes, "torch.compile; the first requests after loading are slower.")}
          </div>
          <div className="toggle-grid">
            {toggleField("cuda_graphs", "CUDA graphs", "EmbeddingGemma 2 only.")}
            {toggleField("warmup_on_load", "Warm up on load")}
          </div>
        </section>

        <section className="panel" aria-labelledby="settings-batching">
          <div className="panel-head"><h2 id="settings-batching">Batching and limits</h2></div>
          <div className="toggle-grid">
            {toggleField("batching_enabled", "Enable batching", "Collect concurrent requests into one GPU pass.")}
            {toggleField("encode_token_batching", "Pack texts by tokens")}
          </div>
          <div className="form-grid">
            {(["batch_wait_time_ms", "batch_max_texts", "batch_max_chars", "encode_batch_size", "max_seq_length", "idle_trim_seconds"] as NumberKey[]).map(numberField)}
            {selectField("too_long_policy", "Texts over the token limit", settingsOptions.too_long_policies)}
          </div>
        </section>

        <section className="panel" aria-labelledby="settings-storage">
          <div className="panel-head"><h2 id="settings-storage">Storage and downloads</h2></div>
          <div className="form-grid">
            <Field label="Cache path" hint="Folder for downloaded models. Empty = the Hugging Face default cache." dirty={isDirty("model_cache_path")} wide>
              <input value={settingsForm.model_cache_path} onChange={(event) => updateSetting("model_cache_path", event.target.value)} />
            </Field>
            <Field
              label="Hugging Face token"
              hint={tokenClear ? "The stored token is removed when you save." : tokenSet ? "A token is stored. Type a new one to replace it." : "Optional. Needed only for gated models."}
              dirty={tokenDirty}
              wide
            >
              <div className="input-row">
                <input
                  type="password"
                  autoComplete="new-password"
                  value={tokenInput}
                  disabled={tokenClear}
                  placeholder={tokenSet ? `Saved: ${tokenHint ?? "hf_…"} · leave empty to keep` : "hf_…"}
                  onChange={(event) => setTokenInput(event.target.value)}
                />
                {tokenSet || tokenClear ? (
                  <button type="button" className="ghost-button" onClick={() => { setTokenClear(!tokenClear); setTokenInput(""); }}>
                    {tokenClear ? "Keep token" : "Clear"}
                  </button>
                ) : null}
              </div>
            </Field>
          </div>
        </section>

        <section className="panel" aria-labelledby="settings-media">
          <div className="panel-head">
            <h2 id="settings-media">Media inputs</h2>
            <p className="hint">Images, audio and video frames for multimodal models. Media requests bypass the batch queue.</p>
          </div>
          <div className="toggle-grid">
            {MEDIA_TOGGLES.map((field) => toggleField(field.key, field.label))}
          </div>
          <div className="form-grid">{MEDIA_LIMIT_KEYS.map(numberField)}</div>
        </section>

        <div className={`save-bar${dirtyCount > 0 ? " is-dirty" : ""}`} role="region" aria-label="Save settings">
          <span className="save-bar-status">
            {dirtyCount === 0 ? "No unsaved changes" : `${dirtyCount} unsaved change${dirtyCount === 1 ? "" : "s"}`}
            {invalidKeys.length > 0 ? <span className="text-danger"> · {invalidKeys.length} invalid</span> : null}
          </span>
          <div className="button-row">
            <button type="button" className="ghost-button" onClick={discardSettings} disabled={dirtyCount === 0 || saveBusy}>Discard</button>
            <button type="button" className="secondary-button" onClick={() => void reloadSettings()} disabled={saveBusy}>Reload</button>
            <button type="submit" disabled={dirtyCount === 0 || invalidKeys.length > 0 || saveBusy} aria-busy={saveBusy}>{saveBusy ? "Saving…" : "Save changes"}</button>
          </div>
        </div>
      </form>
    </>
  );

  const benchmark = (
    <section className="panel" aria-labelledby="bench-title">
      <div className="panel-head">
        <h2 id="bench-title">Throughput benchmark</h2>
        <p className="hint">Encodes every input line the given number of times, directly on the model (no queue). The load shows up in Performance on the Dashboard.</p>
      </div>
      <form className="form-grid" onSubmit={handleBenchmark}>
        <Field label="Model">{modelSelect(benchmarkModel, setBenchmarkModel, "bench-model")}</Field>
        <Field label="Repeats" hint="1–128." error={benchmarkRepeatCount === "" ? "Required." : null}>
          <input type="number" min={1} max={128} value={benchmarkRepeatCount} onChange={(event) => setBenchmarkRepeatCount(event.target.value === "" ? "" : Number(event.target.value))} />
        </Field>
        <Field label="Inputs" hint="One text per line." wide>
          <textarea value={benchmarkInputs} onChange={(event) => setBenchmarkInputs(event.target.value)} rows={6} />
        </Field>
        <div className="form-actions field-wide">
          <button type="submit" disabled={benchmarkBusy} aria-busy={benchmarkBusy}>{benchmarkBusy ? "Running…" : "Run benchmark"}</button>
        </div>
      </form>
      {benchmarkResult ? (
        <div className="grid grid-auto card-row" aria-live="polite">
          <Card label="Throughput" value={fmtRate(benchmarkResult.texts_per_second, "texts/s")} />
          <Card label="Total time" value={fmtMs(benchmarkResult.total_wall_time_ms)} sub={`${fmtMs(benchmarkResult.avg_wall_time_per_run_ms)} per run`} />
          <Card label="Texts" value={fmtNum(benchmarkResult.total_texts)} sub={`${fmtNum(benchmarkResult.texts_per_run)} × ${fmtNum(benchmarkResult.repeat_count)} · ${fmtNum(benchmarkResult.total_chars)} characters`} />
          <Card label="Runtime" value={benchmarkResult.device_label || benchmarkResult.device} sub={<span className="mono">{benchmarkResult.model_id}</span>} />
        </div>
      ) : null}
    </section>
  );

  const historyTab = (
    <section className="panel" aria-labelledby="history-title">
      <div className="panel-head">
        <h2 id="history-title">Latest requests</h2>
        <p className="hint">The last {fmtNum(history.length)} embedding requests (API, playground). Payloads are never stored, only counts.</p>
      </div>
      <PanelError message={panelErrors.stats} onRetry={() => void refreshCore()} />
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Time</th><th>Route</th><th>Model</th><th className="num">Inputs</th><th className="num">Dim.</th><th className="num">Total</th><th className="num">Encode</th><th>Mode</th><th>Status</th></tr>
          </thead>
          <tbody>
            {stats === null ? <tr><td colSpan={9} className="empty"><span className="spinner" /> Loading…</td></tr> : null}
            {stats !== null && history.length === 0 ? <tr><td colSpan={9} className="empty">No requests yet. Run a comparison in Playground or call the API.</td></tr> : null}
            {history.slice(0, 50).map((entry, index) => (
              <tr key={`${entry.timestamp ?? "row"}-${index}`}>
                <td className="nowrap">{fmtDateTime(entry.timestamp, true)}</td>
                <td className="nowrap mono">{entry.route ?? "—"}</td>
                <td className="wrap"><span className="mono">{entry.model_id ?? "—"}</span></td>
                <td className="num">
                  {fmtNum(entry.input_count ?? null)}
                  <div className="hint">
                    {fmtNum(entry.input_chars ?? null)} chars{entry.input_tokens != null ? ` · ${fmtNum(entry.input_tokens)} tokens` : ""}
                    {entry.media ? ` · ${mediaCounts(entry.media)}` : ""}
                  </div>
                </td>
                <td className="num">{fmtNum(entry.dimension ?? null)}</td>
                <td className="num">{fmtMs(entry.total_duration_ms)}</td>
                <td className="num">{fmtMs(entry.encode_duration_ms)}</td>
                <td>{entry.batched ? "Batched" : "Direct"}</td>
                <td className="status-cell">
                  {entry.success ? <span className="badge ok">OK</span> : <span className="badge danger" title={entry.error}>Failed</span>}
                  {!entry.success && entry.error ? <div className="field-error">{entry.error}</div> : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );

  const origin = window.location.origin;
  const apiKeysTab = (
    <ApiKeysPanel
      keys={apiKeys?.map((key) => ({
        id: key.id,
        alias: key.alias,
        hint: key.hint ?? null,
        created_at: key.created_at,
        last_used_at: key.usage.last_used_at,
        requests: key.usage.request_count,
        usage: key.usage.total_items_processed,
      })) ?? null}
      loadError={panelErrors.keys}
      usageLabel="Texts"
      baseUrl={origin}
      routesText={PUBLIC_ROUTES}
      exampleCurl={(key) =>
        `curl -s ${origin}/v1/embeddings \\\n  -H "Authorization: Bearer ${key}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model": "${defaultModel || "google/embeddinggemma-2"}", "input": ["Hello world"]}'`
      }
      onCreate={async (alias) => {
        try {
          return await createApiKey(alias);
        } catch (error) {
          handleAuthError(error);
          throw error;
        }
      }}
      onRevoke={async (id) => {
        try {
          await deleteApiKey(id);
        } catch (error) {
          handleAuthError(error);
          throw error;
        }
      }}
      onChanged={() => void refreshKeys()}
    />
  );

  const account = (
    <div className="grid grid-2">
      <section className="panel" aria-labelledby="account-title">
        <div className="panel-head"><h2 id="account-title">Account</h2></div>
        <dl className="kv">
          <dt>Signed in as</dt>
          <dd>{currentUser || "—"}</dd>
          <dt>Session expires</dt>
          <dd>{fmtDateTime(sessionExpiresAt)}</dd>
        </dl>
        <div className="button-row">
          <button type="button" className="ghost-button" onClick={() => void handleLogout()}>Log out</button>
        </div>
      </section>
      <section className="panel" aria-labelledby="password-title">
        <div className="panel-head">
          <h2 id="password-title">Change password</h2>
          <p className="hint">Changing the password signs out every other session of this user.</p>
        </div>
        {passwordForm("Change password")}
      </section>
    </div>
  );

  const content: Record<TabId, ReactNode> = {
    dashboard,
    playground,
    models,
    settings: settingsTab,
    benchmark,
    history: historyTab,
    "api-keys": null, // always mounted below, so a new key survives tab switches (never logout)
    account,
  };

  return (
    <>
      <ToastHost />
      <div className="shell">
        <header className="app-header">
          <div className="titles">
            <p className="eyebrow">G3 GPU service · Admin</p>
            <h1>{SERVICE_NAME}</h1>
            <div className="status-strip">
              {stateBadge}
              <span className="chip" title={defaultModel}><span className="card-label">Model</span><span className="value">{defaultModel || "None"}</span></span>
              <span className="chip" title={serviceGpus.map(gpuTitle).join(", ")}>
                <span className="card-label">GPU</span>
                <span className="value">{perf ? (serviceGpus.length ? serviceGpus.map(gpuTitle).join(", ") : "CPU") : "—"}</span>
              </span>
              <a className="chip chip-link" href="#api-keys" onClick={(event) => { event.preventDefault(); selectTab("api-keys"); }}>
                <span className="card-label">API</span>
                {apiKeys === null ? <span className="value">—</span> : <span className={`badge ${keyCount > 0 ? "ok" : "warn"}`}>{keyCount > 0 ? "Protected" : "Open"}</span>}
              </a>
            </div>
          </div>
          <div className="actions">
            <a className="button-link ghost" href="/docs" target="_blank" rel="noreferrer">API docs</a>
            <button type="button" className="ghost-button" onClick={() => void handleLogout()}>Log out</button>
          </div>
        </header>

        <nav className="tab-bar" role="tablist" aria-label="Sections" onKeyDown={onTabKey}>
          {TABS.map((tab) => (
            <button
              key={tab.id}
              ref={(element) => { tabRefs.current[tab.id] = element; }}
              type="button"
              role="tab"
              id={`tab-${tab.id}`}
              className="tab-button"
              aria-selected={activeTab === tab.id}
              aria-controls={`panel-${tab.id}`}
              tabIndex={activeTab === tab.id ? 0 : -1}
              onClick={() => selectTab(tab.id)}
            >
              {tab.label}
              {tab.id === "settings" && dirtyCount > 0 ? <span className="tab-dot" aria-label="unsaved changes" /> : null}
            </button>
          ))}
        </nav>

        <main role="tabpanel" id={`panel-${activeTab}`} aria-labelledby={`tab-${activeTab}`} className="tab-panel">
          {content[activeTab]}
          <div hidden={activeTab !== "api-keys"}>{apiKeysTab}</div>
        </main>
      </div>
    </>
  );
}
