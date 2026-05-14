const HUGGING_FACE_MODEL_API = "https://huggingface.co/api/models";
const GIB = 1024 ** 3;
const MIB = 1024 ** 2;

export function getBrowserFitContext() {
  const nav = globalThis.navigator ?? {};

  return {
    deviceMemoryGb: typeof nav.deviceMemory === "number" && Number.isFinite(nav.deviceMemory) ? nav.deviceMemory : null,
    hardwareConcurrency: typeof nav.hardwareConcurrency === "number" ? nav.hardwareConcurrency : null
  };
}

export async function searchWebOnnxModels(query, signal) {
  const params = new URLSearchParams({
    filter: "onnx",
    limit: "18",
    sort: "downloads",
    direction: "-1",
    full: "true",
    config: "true"
  });

  const trimmed = query.trim();
  if (trimmed) params.set("search", trimmed);

  const response = await fetch(`${HUGGING_FACE_MODEL_API}?${params.toString()}`, {
    headers: { Accept: "application/json" },
    signal
  });

  if (!response.ok) {
    throw new Error(`Hugging Face model search failed (${response.status})`);
  }

  const entries = await response.json();

  const models = entries
    .filter((entry) => entry?.id?.includes("/"))
    .map(normalizeHubEntry)
    .filter(Boolean)
    .slice(0, 12);

  return Promise.all(models.map((model) => hydrateRemoteSize(model, signal)));
}

export async function createRemoteModelFromUrl(rawUrl, signal) {
  const url = normalizeHttpUrl(rawUrl);
  const sizeBytes = await estimateRemoteSize(url, signal);
  const path = decodeURIComponent(new URL(url).pathname.split("/").pop() || "model.onnx");

  if (!path.toLowerCase().endsWith(".onnx")) {
    throw new Error("Use a direct URL to an .onnx file.");
  }

  return {
    id: url,
    name: path,
    modelId: new URL(url).host,
    source: "Direct URL",
    artifactPath: path,
    downloadUrl: url,
    sizeBytes,
    estimatedBytes: sizeBytes,
    parameterCountB: parseParameterCountB(url, []),
    downloads: null,
    likes: null,
    tags: []
  };
}

export function assessRemoteModelFit(model, context = getBrowserFitContext()) {
  const estimatedBytes = model.estimatedBytes ?? model.sizeBytes ?? null;
  const workingSetBytes = estimateWorkingSetBytes(estimatedBytes);

  if (!estimatedBytes || !workingSetBytes) {
    return {
      label: "Unknown",
      tone: "amber",
      score: 45,
      summary: "Size metadata is unavailable. Load only if you trust the source and expect a small graph.",
      workingSetLabel: "unknown",
      budgetLabel: context.deviceMemoryGb ? `${formatGb(context.deviceMemoryGb * 0.35)} browser budget` : "unknown budget"
    };
  }

  const memoryBudgetBytes = context.deviceMemoryGb ? context.deviceMemoryGb * GIB * 0.35 : null;
  let score = 72;

  if (estimatedBytes <= 150 * MIB) score += 18;
  else if (estimatedBytes <= 500 * MIB) score += 8;
  else if (estimatedBytes <= 1.5 * GIB) score -= 10;
  else score -= 26;

  if (memoryBudgetBytes) {
    const ratio = workingSetBytes / memoryBudgetBytes;
    if (ratio <= 0.6) score += 12;
    else if (ratio <= 1) score += 2;
    else if (ratio <= 1.35) score -= 16;
    else score -= 34;
  }

  if (context.hardwareConcurrency !== null && context.hardwareConcurrency <= 4 && estimatedBytes > 800 * MIB) {
    score -= 8;
  }

  score = clamp(score, 5, 95);
  const workingSetLabel = formatBytes(workingSetBytes);
  const budgetLabel = memoryBudgetBytes ? `${formatBytes(memoryBudgetBytes)} browser budget` : "device memory unavailable";

  if (score >= 82) {
    return {
      label: "Fits",
      tone: "green",
      score,
      summary: "Comfortable for local download and browser-side ONNX parsing.",
      workingSetLabel,
      budgetLabel
    };
  }

  if (score >= 64) {
    return {
      label: "Likely",
      tone: "emerald",
      score,
      summary: "Should parse locally, though download and decode time may be noticeable.",
      workingSetLabel,
      budgetLabel
    };
  }

  if (score >= 42) {
    return {
      label: "Borderline",
      tone: "amber",
      score,
      summary: "May work, but this browser could hit memory pressure while parsing.",
      workingSetLabel,
      budgetLabel
    };
  }

  return {
    label: "Too large",
    tone: "red",
    score,
    summary: "Large for browser-side parsing on this machine.",
    workingSetLabel,
    budgetLabel
  };
}

export function formatBytes(bytes) {
  if (!bytes || !Number.isFinite(bytes)) return "unknown";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < MIB) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < GIB) return `${(bytes / MIB).toFixed(1)} MB`;
  return `${(bytes / GIB).toFixed(2)} GB`;
}

function normalizeHubEntry(entry) {
  const siblings = Array.isArray(entry.siblings) ? entry.siblings : [];
  const artifact = pickOnnxArtifact(siblings);
  if (!artifact) return null;

  const tags = Array.isArray(entry.tags) ? entry.tags : [];
  const parameterCountB = parseParameterCountB(entry.id, tags);
  const usedStorageBytes = typeof entry.usedStorage === "number" && entry.usedStorage > 0 ? entry.usedStorage : null;
  const estimatedBytes = artifact.sizeBytes ?? usedStorageBytes ?? estimateBytesFromParameters(parameterCountB);
  const revision = entry.sha || "main";

  return {
    id: `${entry.id}:${artifact.path}`,
    name: entry.id.split("/").pop()?.replace(/-ONNX$/i, "") || entry.id,
    modelId: entry.id,
    source: "Hugging Face",
    revision,
    artifactPath: artifact.path,
    downloadUrl: `https://huggingface.co/${entry.id}/resolve/${revision}/${artifact.path}`,
    sizeBytes: artifact.sizeBytes,
    estimatedBytes,
    parameterCountB,
    downloads: entry.downloads ?? 0,
    likes: entry.likes ?? 0,
    tags
  };
}

function pickOnnxArtifact(siblings) {
  const candidates = siblings
    .map((sibling) => ({
      path: sibling.rfilename || sibling.path || "",
      sizeBytes: getSiblingSize(sibling)
    }))
    .filter((file) => file.path.toLowerCase().endsWith(".onnx"));

  if (candidates.length === 0) return null;

  return candidates
    .map((file) => ({ ...file, rank: rankOnnxPath(file.path) }))
    .sort((a, b) => a.rank - b.rank || (a.sizeBytes ?? Number.MAX_SAFE_INTEGER) - (b.sizeBytes ?? Number.MAX_SAFE_INTEGER))[0];
}

function rankOnnxPath(path) {
  const name = path.toLowerCase();
  if (/q4|quant|int8|uint8/.test(name)) return 0;
  if (/fp16|float16/.test(name)) return 1;
  if (/decoder_model_merged|decoder_with_past|model/.test(name)) return 2;
  if (name.startsWith("onnx/")) return 3;
  return 4;
}

function getSiblingSize(sibling) {
  const size = sibling?.lfs?.size ?? sibling?.size;
  return typeof size === "number" && Number.isFinite(size) && size > 0 ? size : null;
}

function parseParameterCountB(modelId, tags) {
  for (const value of [modelId, ...tags]) {
    const billion = value.match(/(?:^|[^\d])(\d+(?:\.\d+)?)\s*B(?:$|[^\w])/i);
    if (billion) return Number.parseFloat(billion[1]);

    const million = value.match(/(?:^|[^\d])(\d+(?:\.\d+)?)\s*M(?:$|[^\w])/i);
    if (million) return Number.parseFloat(million[1]) / 1000;
  }

  return null;
}

function estimateBytesFromParameters(parameterCountB) {
  if (!parameterCountB) return null;
  if (parameterCountB <= 0.25) return parameterCountB * GIB * 1.15;
  if (parameterCountB <= 1) return parameterCountB * GIB;
  if (parameterCountB <= 3) return parameterCountB * GIB * 0.9;
  return parameterCountB * GIB * 0.8;
}

function estimateWorkingSetBytes(estimatedBytes) {
  if (!estimatedBytes) return null;
  return estimatedBytes * 2.6 + 180 * MIB;
}

async function estimateRemoteSize(url, signal) {
  try {
    const response = await fetch(url, { method: "HEAD", signal });
    if (!response.ok) return null;
    const contentLength = response.headers.get("Content-Length");
    const parsed = contentLength ? Number.parseInt(contentLength, 10) : null;
    return parsed && Number.isFinite(parsed) && parsed > 0 ? parsed : null;
  } catch {
    return null;
  }
}

async function hydrateRemoteSize(model, signal) {
  if (model.sizeBytes) return model;

  const sizeBytes = await estimateRemoteSize(model.downloadUrl, signal);
  return {
    ...model,
    sizeBytes,
    estimatedBytes: sizeBytes ?? model.estimatedBytes
  };
}

function normalizeHttpUrl(rawUrl) {
  let url;
  try {
    url = new URL(rawUrl.trim());
  } catch {
    throw new Error("Enter a valid HTTPS URL.");
  }

  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new Error("Only HTTP or HTTPS model URLs can be loaded.");
  }

  return url.toString();
}

function formatGb(value) {
  return `${value.toFixed(value >= 10 ? 0 : 1)} GB`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
