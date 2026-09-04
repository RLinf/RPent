export function formatValue(value) {
  if (value == null || value === "") return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function errorText(error) {
  if (typeof error === "object" && error !== null) {
    if (typeof error.message === "string") return error.message;
    if (typeof error.detail === "string") return error.detail;
    if (typeof error.error === "string") return error.error;
  }
  return formatValue(error);
}

async function responseErrorText(response) {
  let body = "";
  try {
    body = await response.text();
  } catch {}
  if (!body) return `${response.status} ${response.statusText}`.trim();
  try {
    const parsed = JSON.parse(body);
    return errorText(parsed.detail ?? parsed.error ?? parsed.message ?? parsed);
  } catch {
    return body;
  }
}

export async function requestJSON(url, { method = "GET", body } = {}) {
  const options = { method };
  if (body !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await responseErrorText(response));
  if (response.status === 204) return null;
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
