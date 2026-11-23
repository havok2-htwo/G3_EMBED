// en-US number and date formatting for the admin UI (G3 admin standard 3.9). Unknown values render as "—".

const DASH = "—";
const formatters = new Map<string, Intl.NumberFormat>();

function numberFormat(minimumFractionDigits: number, maximumFractionDigits: number) {
  const key = `${minimumFractionDigits}:${maximumFractionDigits}`;
  let formatter = formatters.get(key);
  if (!formatter) {
    formatter = new Intl.NumberFormat("en-US", { minimumFractionDigits, maximumFractionDigits });
    formatters.set(key, formatter);
  }
  return formatter;
}

const known = (value: number | null | undefined): value is number =>
  value !== null && value !== undefined && Number.isFinite(value);

/** 1,234 or 1,234.5 (up to `maxDigits` decimals). */
export function fmtNum(value: number | null | undefined, maxDigits = 0, unit = "") {
  if (!known(value)) return DASH;
  const text = numberFormat(0, maxDigits).format(value);
  return unit ? `${text} ${unit}` : text;
}

/** Fixed decimals: fmtFixed(0.12345, 4) = "0.1235". */
export function fmtFixed(value: number | null | undefined, digits: number) {
  return known(value) ? numberFormat(digits, digits).format(value) : DASH;
}

/** Latency: 1 decimal below 10 ms, integer ms up to 9,999 ms, then seconds with 1 decimal. */
export function fmtMs(value: number | null | undefined) {
  if (!known(value)) return DASH;
  if (value < 10) return `${numberFormat(1, 1).format(value)} ms`;
  if (value < 10_000) return `${numberFormat(0, 0).format(value)} ms`;
  return `${numberFormat(1, 1).format(value / 1000)} s`;
}

/** Rates such as texts/s: 1 decimal. */
export function fmtRate(value: number | null | undefined, unit: string) {
  return known(value) ? `${numberFormat(1, 1).format(value)} ${unit}` : DASH;
}

/** Sizes in GB with one decimal (two below 10 GB is not needed: 0.5 GB). */
export function fmtGb(value: number | null | undefined) {
  if (!known(value)) return DASH;
  if (value < 1) return `${numberFormat(0, 0).format(value * 1024)} MB`;
  return `${numberFormat(1, 1).format(value)} GB`;
}

const pad = (n: number) => String(n).padStart(2, "0");

/** 2026-11-23 21:04 (local time); withSeconds adds :ss. */
export function fmtDateTime(value: string | number | null | undefined, withSeconds = false, empty = DASH) {
  if (value === null || value === undefined || value === "") return empty;
  const date = typeof value === "number" ? new Date(value) : new Date(value);
  if (Number.isNaN(date.getTime())) return DASH;
  const base = `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  return withSeconds ? `${base}:${pad(date.getSeconds())}` : base;
}

/** 1 h 12 min, 3 min 4 s, 12 s. */
export function fmtDuration(seconds: number | null | undefined) {
  if (!known(seconds)) return DASH;
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h) return `${h} h ${m} min`;
  if (m) return `${m} min ${s} s`;
  return `${s} s`;
}
