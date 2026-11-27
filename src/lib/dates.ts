const formatter = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});

const monthFormatter = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
  timeZone: "UTC",
});

/** "2026-11-27" → "Nov 27, 2026". Returns null for missing or invalid input. */
export function formatDate(value?: string | null) {
  if (!value) return null;
  const d = new Date(value.length === 10 ? `${value}T00:00:00Z` : value);
  if (Number.isNaN(d.getTime())) return null;
  return formatter.format(d);
}

/** "2021-06" → "Jun 2021"; "2021" → "2021"; "present" → "Present". */
export function formatPeriod(value?: string | null) {
  if (!value) return null;
  if (value.toLowerCase() === "present") return "Present";
  if (/^\d{4}$/.test(value)) return value;
  const d = new Date(/^\d{4}-\d{2}$/.test(value) ? `${value}-01T00:00:00Z` : value);
  if (Number.isNaN(d.getTime())) return value;
  return monthFormatter.format(d);
}

export function newestDate(dates: (string | null | undefined)[]) {
  const valid = dates.filter((d): d is string => !!d && !Number.isNaN(Date.parse(d)));
  if (!valid.length) return null;
  return valid.reduce((a, b) => (Date.parse(a) > Date.parse(b) ? a : b));
}
