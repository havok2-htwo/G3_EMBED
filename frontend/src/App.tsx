import { FormEvent, startTransition, useEffect, useState } from "react";

import {
  AdminKeyMetadata,
  AdminOption,
  AdminSettings,
  BenchmarkResponse,
  CompareResponse,
  ManagedModel,
  QueueResponse,
  RuntimeInfo,
  SettingsResponse,
  StatsResponse,
  compareTexts,
  deleteModel,
  downloadModel,
  getKeys,
  getModels,
  getQueue,
  getSettings,
  getStats,
  rotateAdminKey,
  runBenchmark,
  saveSettings,
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

const ADMIN_KEY_STORAGE = "genesis_embed_admin_key";
const NUMBER_LOCALE = "de-DE";
const POLL_MS = 5000;
const MODEL_STATUS_POLL_MS = 2000;
const HISTORY_POINTS = 72;

const emptySettings: AdminSettings = {
  default_model: "intfloat/multilingual-e5-small",
  execution_target: "auto",
  backend_override: "auto",
  model_cache_path: ".\\models",
  batching_enabled: true,
  batch_wait_time_ms: 25,
  batch_max_texts: 32,
  batch_max_chars: 64000,
  precision: "auto",
  compile_mode: "off",
  warmup_on_load: false,
  huggingface_token: "",
};

const emptyOptions = {
  models: [] as AdminOption[],
  execution_targets: [] as AdminOption[],
  backends: [] as AdminOption[],
  precisions: [] as AdminOption[],
  compile_modes: [] as AdminOption[],
};

function readStoredAdminKey() {
  try {
    return localStorage.getItem(ADMIN_KEY_STORAGE) || sessionStorage.getItem(ADMIN_KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

function writeStoredAdminKey(value: string) {
  try {
    localStorage.setItem(ADMIN_KEY_STORAGE, value);
    sessionStorage.setItem(ADMIN_KEY_STORAGE, value);
  } catch {}
}

function clearStoredAdminKey() {
  try {
    localStorage.removeItem(ADMIN_KEY_STORAGE);
    sessionStorage.removeItem(ADMIN_KEY_STORAGE);
  } catch {}
}

function formatValue(value: number | null | undefined, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  const maximumFractionDigits = Number.isInteger(value) ? 0 : 2;
  return `${new Intl.NumberFormat(NUMBER_LOCALE, { maximumFractionDigits }).format(value)}${suffix}`;
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

function stringFromRecord(record: Record<string, unknown>, key: string) {
  const value = record[key];
  return typeof value === "string" ? value : "";
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
  const [adminKey, setAdminKey] = useState(readStoredAdminKey);
  const [loginKey, setLoginKey] = useState(readStoredAdminKey);
  const [authorized, setAuthorized] = useState(false);
  const [loading, setLoading] = useState(false);
  const [globalError, setGlobalError] = useState("");
  const [actionMessage, setActionMessage] = useState("");
  const [settingsForm, setSettingsForm] = useState<AdminSettings>(emptySettings);
  const [settingsOptions, setSettingsOptions] = useState(emptyOptions);
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const [managedModels, setManagedModels] = useState<ManagedModel[]>([]);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [queue, setQueue] = useState<QueueResponse | null>(null);
  const [queueHistory, setQueueHistory] = useState<QueuePoint[]>([]);
  const [adminMetadata, setAdminMetadata] = useState<AdminKeyMetadata | null>(null);
  const [newlyCreatedKey, setNewlyCreatedKey] = useState<(AdminKeyMetadata & { token: string }) | null>(null);
  const [saveBusy, setSaveBusy] = useState(false);
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

  const hasDownloadingModels = managedModels.some((model) => model.status === "downloading");
  const history = (stats?.history ?? []) as HistoryEntry[];

  function applySettings(payload: SettingsResponse) {
    setSettingsForm(payload.settings);
    setSettingsOptions(payload.options);
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

  async function refreshDashboard(currentAdminKey = adminKey) {
    if (!currentAdminKey) {
      return;
    }
    try {
      const [settingsResponse, statsResponse, queueResponse, keysResponse] = await Promise.all([
        getSettings(currentAdminKey),
        getStats(currentAdminKey),
        getQueue(currentAdminKey),
        getKeys(currentAdminKey),
      ]);
      startTransition(() => {
        applySettings(settingsResponse);
        setStats(statsResponse);
        setQueue(queueResponse);
        setAdminMetadata(keysResponse.admin_key);
        recordQueuePoint(queueResponse);
        setGlobalError("");
      });
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        setAuthorized(false);
        clearStoredAdminKey();
      }
      setGlobalError(error instanceof Error ? error.message : "Dashboard refresh failed.");
    }
  }

  useEffect(() => {
    if (!adminKey) {
      return;
    }
    setLoading(true);
    Promise.all([getSettings(adminKey), getKeys(adminKey)])
      .then(([settingsResponse, keysResponse]) => {
        writeStoredAdminKey(adminKey);
        applySettings(settingsResponse);
        setAdminMetadata(keysResponse.admin_key);
        setAuthorized(true);
        void refreshDashboard(adminKey);
      })
      .catch(() => {
        setAuthorized(false);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!authorized || !adminKey) {
      return;
    }
    const interval = window.setInterval(() => {
      void refreshDashboard(adminKey);
    }, POLL_MS);
    return () => window.clearInterval(interval);
  }, [authorized, adminKey]);

  useEffect(() => {
    if (!authorized || !adminKey || !hasDownloadingModels) {
      return;
    }
    const interval = window.setInterval(() => {
      void refreshModels();
    }, MODEL_STATUS_POLL_MS);
    return () => window.clearInterval(interval);
  }, [authorized, adminKey, hasDownloadingModels, settingsForm.model_cache_path]);

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const candidate = loginKey.trim();
    if (!candidate) {
      setGlobalError("Admin key is required.");
      return;
    }
    setLoading(true);
    setGlobalError("");
    try {
      const [settingsResponse, keysResponse] = await Promise.all([getSettings(candidate), getKeys(candidate)]);
      writeStoredAdminKey(candidate);
      setAdminKey(candidate);
      applySettings(settingsResponse);
      setAdminMetadata(keysResponse.admin_key);
      setAuthorized(true);
      await refreshDashboard(candidate);
    } catch (error) {
      setGlobalError(error instanceof Error && error.message === "unauthorized" ? "Invalid admin key." : "Login failed.");
    } finally {
      setLoading(false);
    }
  }

  function handleLogout() {
    clearStoredAdminKey();
    setAuthorized(false);
    setAdminKey("");
    setLoginKey("");
    setStats(null);
    setQueue(null);
    setRuntime(null);
    setManagedModels([]);
    setNewlyCreatedKey(null);
  }

  async function handleRotateKey() {
    setActionMessage("");
    setGlobalError("");
    try {
      const response = await rotateAdminKey(adminKey);
      setNewlyCreatedKey(response.key);
      setAdminMetadata(response.keys.admin_key);
      setAdminKey(response.key.token);
      setLoginKey(response.key.token);
      writeStoredAdminKey(response.key.token);
      setActionMessage("Admin key rotated. The new key is active locally.");
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "The admin key could not be rotated.");
    }
  }

  async function handleSaveSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaveBusy(true);
    setActionMessage("");
    setGlobalError("");
    try {
      const response = await saveSettings(adminKey, settingsForm);
      applySettings(response);
      setActionMessage("Settings saved.");
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Settings could not be saved.");
    } finally {
      setSaveBusy(false);
    }
  }

  async function refreshModels() {
    if (!adminKey) {
      return;
    }
    setModelBusyId("__refresh__");
    setModelActionKind("refresh");
    try {
      const response = await getModels(adminKey, settingsForm.model_cache_path);
      setManagedModels(response.models ?? []);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Model refresh failed.");
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
      const response = await downloadModel(adminKey, model.id, settingsForm.model_cache_path, settingsForm.huggingface_token);
      setManagedModels(response.models ?? []);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Model download failed.");
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
      const response = await deleteModel(adminKey, model.id, settingsForm.model_cache_path);
      setManagedModels(response.models ?? []);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Model delete failed.");
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
      const result = await compareTexts(adminKey, compareModel, compareTextA, compareTextB);
      setCompareResult(result);
      setActionMessage(`Compared ${result.dimension} dimensions in ${formatValue(result.total_duration_ms, " ms")}.`);
      await refreshDashboard(adminKey);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Compare failed.");
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
      const result = await runBenchmark(adminKey, compareModel, inputs, benchmarkRepeatCount);
      setBenchmarkResult(result);
      setActionMessage(`Benchmark finished: ${formatValue(result.total_texts)} texts in ${formatValue(result.total_wall_time_ms, " ms")}.`);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Benchmark failed.");
    } finally {
      setBenchmarkBusy(false);
    }
  }

  function updateSetting<K extends keyof AdminSettings>(key: K, value: AdminSettings[K]) {
    setSettingsForm((current) => ({ ...current, [key]: value }));
  }

  if (!authorized) {
    return (
      <main className="centered">
        <section className="panel login-panel">
          <div>
            <span className="eyebrow">G3_EMBED Admin</span>
            <h1>Embedding server control room</h1>
            <p>Enter the startup or persistent admin key to manage models, hardware routing, batching, and local test runs.</p>
          </div>
          <form className="login-form" onSubmit={handleLogin}>
            <label>
              <span>Admin Key</span>
              <input
                type="password"
                value={loginKey}
                onChange={(event) => setLoginKey(event.target.value)}
                placeholder="genesis_embed_admin_..."
                autoFocus
              />
            </label>
            <button type="submit" disabled={loading}>{loading ? "Checking..." : "Open Dashboard"}</button>
            {globalError ? <p className="message error">{globalError}</p> : null}
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
          <button type="button" className="ghost-button" onClick={handleLogout}>Logout</button>
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
          <span>Queue</span>
          <strong>{formatValue(queue?.queue_size ?? 0)}</strong>
          <small>{formatValue(queue?.last_batch_texts_per_second, " texts/s")}</small>
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
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => void copyTextToClipboard(JSON.stringify(compareResult.vectors, null, 2))}
                >
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
            <div className="form-actions full-width">
              <button type="submit" disabled={saveBusy}>{saveBusy ? "Saving..." : "Save Settings"}</button>
            </div>
          </form>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Admin Key</span>
              <h2>Access</h2>
            </div>
          </div>
          {newlyCreatedKey ? (
            <div className="key-token-card">
              <strong>New key</strong>
              <p>The server returns this token only once.</p>
              <div className="key-token-value mono">{newlyCreatedKey.token}</div>
              <button type="button" className="secondary-button" onClick={() => void copyTextToClipboard(newlyCreatedKey.token)}>Copy Key</button>
            </div>
          ) : null}
          <div className="key-token-card">
            <strong>{adminMetadata?.label || "Master Admin Key"}</strong>
            <p>Created {formatDateTime(adminMetadata?.created_at)} | last used {formatDateTime(adminMetadata?.last_used_at)}</p>
            <div className="key-token-value mono">{adminKey}</div>
            <div className="button-row">
              <button type="button" className="secondary-button" onClick={() => void copyTextToClipboard(adminKey)}>Copy Active Key</button>
              <button type="button" className="ghost-button danger-button" onClick={() => void handleRotateKey()}>Rotate Admin Key</button>
            </div>
          </div>

          <div className="panel-divider" />

          <form className="benchmark-form" onSubmit={handleBenchmark}>
            <span className="eyebrow full-width">Benchmark</span>
            <label className="full-width">
              <span>Inputs</span>
              <textarea value={benchmarkInputs} onChange={(event) => setBenchmarkInputs(event.target.value)} rows={5} />
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
                    <td>{formatValue(entry.input_count)} / {formatValue(entry.input_chars)} chars</td>
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
            <div><span>Last Runtime Error</span><strong>{runtime?.last_runtime_error ?? "none"}</strong></div>
          </div>
        </section>
      </div>
    </main>
  );
}
