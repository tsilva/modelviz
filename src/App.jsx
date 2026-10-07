import React, { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, CircuitBoard, DownloadCloud, ExternalLink, FileUp, Globe2, Search, ShieldCheck, X } from "lucide-react";
import MappingWorkspace from "./components/MappingWorkspace.jsx";
import { createEmptyModelView, createModelViewFromOnnx } from "./lib/modelView.js";
import { parseOnnxModel } from "./lib/onnxParser.js";
import { DEFAULT_MODEL } from "./lib/defaultModel.js";
import { assessRemoteModelFit, createRemoteModelFromUrl, formatBytes, getBrowserFitContext, searchWebOnnxModels } from "./lib/webModelBrowser.js";
const initialLoadProgress = remoteModel => ({
  state: "loading",
  title: `Loading ${remoteModel.name}`,
  detail: remoteModel.artifactPath,
  loadedBytes: 0,
  totalBytes: remoteModel.sizeBytes ?? remoteModel.estimatedBytes ?? null
});
function App() {
  const [query, setQuery] = useState("");
  const [modelView, setModelView] = useState(() => createEmptyModelView());
  const [modelRevision, setModelRevision] = useState(0);
  const [loadStatus, setLoadStatus] = useState({
    state: "loading",
    message: "Loading DistilGPT2"
  });
  const [loadProgress, setLoadProgress] = useState(() => initialLoadProgress(DEFAULT_MODEL));
  const [webQuery, setWebQuery] = useState("distilgpt2");
  const [webResults, setWebResults] = useState([]);
  const [webStatus, setWebStatus] = useState({
    state: "idle",
    message: "Search public ONNX files"
  });
  const [directModel, setDirectModel] = useState(null);
  const [modelBrowserOpen, setModelBrowserOpen] = useState(false);
  const [browserSort, setBrowserSort] = useState("fit");
  const [browserFitFilter, setBrowserFitFilter] = useState("all");
  const inputRef = useRef(null);
  const webSearchAbortRef = useRef(null);
  const modelLoadAbortRef = useRef(null);
  const hasModel = Boolean(modelView.model.fileName);
  const fitContext = useMemo(() => getBrowserFitContext(), []);
  const remoteModels = useMemo(() => {
    const combined = directModel ? [directModel, ...webResults.filter(model => model.id !== directModel.id)] : webResults;
    const withFit = combined.map(model => ({
      model,
      fit: assessRemoteModelFit(model, fitContext)
    }));
    const filtered = withFit.filter(({
      fit
    }) => {
      if (browserFitFilter === "all") return true;
      if (browserFitFilter === "fits") return ["Fits", "Likely"].includes(fit.label);
      if (browserFitFilter === "borderline") return fit.label === "Borderline";
      return fit.label === "Too large";
    });
    return filtered.sort((a, b) => {
      if (browserSort === "downloads") return (b.model.downloads ?? 0) - (a.model.downloads ?? 0);
      if (browserSort === "size") return (a.model.sizeBytes ?? a.model.estimatedBytes ?? Number.MAX_SAFE_INTEGER) - (b.model.sizeBytes ?? b.model.estimatedBytes ?? Number.MAX_SAFE_INTEGER);
      if (browserSort === "name") return a.model.name.localeCompare(b.model.name);
      return b.fit.score - a.fit.score;
    });
  }, [browserFitFilter, browserSort, directModel, fitContext, webResults]);
  const openFile = async event => {
    const [file] = event.target.files ?? [];
    if (!file) return;
    modelLoadAbortRef.current?.abort();
    setLoadProgress(null);
    setLoadStatus({
      state: "loading",
      message: `Loading ${file.name}`
    });
    try {
      const parsed = parseOnnxModel(await file.arrayBuffer());
      const nextView = createModelViewFromOnnx(file.name, parsed);
      setModelView(nextView);
      setModelRevision(value => value + 1);
      setQuery("");
      setLoadStatus({
        state: "ready",
        message: nextView.loadMessage
      });
    } catch (error) {
      setLoadStatus({
        state: "error",
        message: error.message
      });
    } finally {
      event.target.value = "";
    }
  };
  const loadModelBuffer = async ({
    fileName,
    sourcePath,
    buffer
  }) => {
    const parsed = parseOnnxModel(buffer);
    const nextView = createModelViewFromOnnx(fileName, parsed, sourcePath);
    setModelView(nextView);
    setModelRevision(value => value + 1);
    setQuery("");
    setLoadStatus({
      state: "ready",
      message: nextView.loadMessage
    });
  };
  const searchWebModels = async event => {
    event?.preventDefault();
    webSearchAbortRef.current?.abort();
    const controller = new AbortController();
    webSearchAbortRef.current = controller;
    const trimmedQuery = webQuery.trim();
    const directUrlQuery = isDirectOnnxUrl(trimmedQuery);
    setWebStatus({
      state: "loading",
      message: directUrlQuery ? "Checking remote ONNX URL" : "Searching Hugging Face ONNX models"
    });
    try {
      if (directUrlQuery) {
        const remote = await createRemoteModelFromUrl(trimmedQuery, controller.signal);
        setDirectModel(remote);
        setWebResults([]);
        setWebStatus({
          state: "ready",
          message: "Remote URL assessed"
        });
        return;
      }
      const results = await searchWebOnnxModels(trimmedQuery, controller.signal);
      setDirectModel(null);
      setWebResults(results);
      setWebStatus({
        state: "ready",
        message: results.length ? `${results.length} ONNX files found` : "No ONNX files found"
      });
    } catch (error) {
      if (controller.signal.aborted) return;
      setWebStatus({
        state: "error",
        message: error.message
      });
    }
  };
  const openModelBrowser = () => {
    setModelBrowserOpen(true);
    if (webResults.length === 0 && webStatus.state === "idle") {
      searchWebModels();
    }
  };
  const loadRemoteModel = async remoteModel => {
    modelLoadAbortRef.current?.abort();
    const controller = new AbortController();
    modelLoadAbortRef.current = controller;
    setModelBrowserOpen(false);
    setLoadStatus({
      state: "loading",
      message: `Downloading ${remoteModel.artifactPath}`
    });
    setLoadProgress(initialLoadProgress(remoteModel));
    try {
      const response = await fetch(remoteModel.downloadUrl, { signal: controller.signal });
      if (!response.ok) throw new Error(`Remote model download failed (${response.status})`);
      const totalBytes = readContentLength(response) ?? remoteModel.sizeBytes ?? remoteModel.estimatedBytes ?? null;
      const buffer = await readResponseBuffer(response, {
        totalBytes,
        onProgress: loadedBytes => {
          if (controller.signal.aborted) return;
          setLoadProgress({
            state: "loading",
            title: `Loading ${remoteModel.name}`,
            detail: remoteModel.artifactPath,
            loadedBytes,
            totalBytes
          });
        }
      });
      if (controller.signal.aborted) return;
      setLoadProgress({
        state: "loading",
        title: `Parsing ${remoteModel.name}`,
        detail: "Building graph and architecture groups",
        loadedBytes: totalBytes ?? buffer.byteLength,
        totalBytes: totalBytes ?? buffer.byteLength
      });
      await loadModelBuffer({
        fileName: remoteModel.fileName || remoteModel.artifactPath.split("/").pop() || remoteModel.name || "model.onnx",
        sourcePath: remoteModel.downloadUrl,
        buffer
      });
      if (remoteModel !== DEFAULT_MODEL) setWebStatus({
        state: "ready",
        message: `Loaded ${remoteModel.modelId}`
      });
      setLoadProgress(null);
    } catch (error) {
      if (controller.signal.aborted) return;
      setLoadStatus({
        state: "error",
        message: error.message
      });
      if (remoteModel !== DEFAULT_MODEL) setWebStatus({
        state: "error",
        message: error.message
      });
      setLoadProgress({
        state: "error",
        title: "Model load failed",
        detail: error.message,
        loadedBytes: 0,
        totalBytes: null
      });
    } finally {
      if (modelLoadAbortRef.current === controller) modelLoadAbortRef.current = null;
    }
  };
  const dismissModelLoad = () => {
    modelLoadAbortRef.current?.abort();
    setLoadProgress(null);
    setLoadStatus({ state: hasModel ? "ready" : "idle", message: hasModel ? modelView.loadMessage : "Open an ONNX model or reload the example" });
  };
  useEffect(() => {
    loadRemoteModel(DEFAULT_MODEL);
    return () => modelLoadAbortRef.current?.abort();
  }, []);
  return <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <img className="modelviz-icon" src="/brand/icon/icon-1024.png" alt="" />
          <div>
            <strong>ModelViz</strong>
            <span>ONNX architecture explorer</span>
          </div>
        </div>
        <span className="loaded-file" title={modelView.model.fileName}>{hasModel ? modelView.model.fileName : "ONNX architecture explorer"}</span>
        <button className="primary" onClick={() => inputRef.current?.click()}>
          <FileUp size={16} />
          Open ONNX
        </button>
        <button className="secondary-topbar" onClick={openModelBrowser}>
          <Globe2 size={16} />
          Browse web
        </button>
        <input ref={inputRef} type="file" accept=".onnx" hidden onChange={openFile} />
        <label className="search-box">
          <Search size={15} />
          <input value={query} onChange={event => setQuery(event.target.value)} placeholder={hasModel ? "Search raw ops, tensors, groups" : "Load a model before searching"} disabled={!hasModel} />
        </label>
        <div className="status-pill"><ShieldCheck size={14} /> {hasModel ? "Parsed in browser" : "Local browser parser"}</div>
      </header>
      {loadStatus.state === "error" && <div className="model-load-error" role="alert"><AlertTriangle size={16} />{loadStatus.message}</div>}
      {loadStatus.state === "loading" && !loadProgress && <div className="model-load-message" role="status">{loadStatus.message}</div>}
      {hasModel ? <MappingWorkspace key={modelRevision} modelView={modelView} query={query} /> : <section className="model-welcome">
        <div className="welcome-illustration"><CircuitBoard size={48} /></div>
        <span className="welcome-eyebrow">FROM ARCHITECTURE TO OPERATORS</span>
        <h1>See how your model fits together.</h1>
        <p>Open an ONNX model to explore its architecture and trace each block to the operators behind it.</p>
        <div className="empty-actions"><button className="primary" onClick={() => inputRef.current?.click()}><FileUp size={16} /> Open ONNX</button><button className="secondary-topbar" onClick={openModelBrowser}><Globe2 size={16} /> Browse web</button></div>
        <button className="text-action" onClick={() => loadRemoteModel(DEFAULT_MODEL)}>Load the DistilGPT2 example</button>
        <span className="welcome-privacy"><ShieldCheck size={14} /> Your model is parsed in this browser.</span>
      </section>}
      {modelBrowserOpen && <ModelBrowserModal browserFitFilter={browserFitFilter} browserSort={browserSort} fitContext={fitContext} models={remoteModels} onClose={() => setModelBrowserOpen(false)} onFitFilterChange={setBrowserFitFilter} onLoadRemoteModel={loadRemoteModel} onSearch={searchWebModels} onSortChange={setBrowserSort} status={webStatus} query={webQuery} onQueryChange={setWebQuery} />}
      {loadProgress && <LoadingProgressModal progress={loadProgress} onClose={dismissModelLoad} />}
    </main>;
}
async function readResponseBuffer(response, {
  totalBytes,
  onProgress
}) {
  if (!response.body?.getReader) {
    const buffer = await response.arrayBuffer();
    onProgress?.(buffer.byteLength);
    return buffer;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let loadedBytes = 0;
  while (true) {
    const {
      done,
      value
    } = await reader.read();
    if (done) break;
    chunks.push(value);
    loadedBytes += value.byteLength;
    onProgress?.(loadedBytes, totalBytes);
  }
  const merged = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return merged.buffer;
}
function readContentLength(response) {
  const value = Number(response.headers.get("content-length"));
  return Number.isFinite(value) && value > 0 ? value : null;
}
function LoadingProgressModal({
  progress,
  onClose
}) {
  const totalBytes = progress.totalBytes;
  const loadedBytes = progress.loadedBytes ?? 0;
  const hasTotal = Number.isFinite(totalBytes) && totalBytes > 0;
  const percent = hasTotal ? Math.min(100, Math.round(loadedBytes / totalBytes * 100)) : null;
  const isError = progress.state === "error";
  const loadedLabel = loadedBytes === 0 ? "0 B" : formatBytes(loadedBytes);
  return <div className="loading-modal-backdrop" role="presentation">
      <section className={`loading-modal ${progress.state}`} role="dialog" aria-modal="true" aria-labelledby="loading-modal-title">
        <header>
          <div className="loading-modal-icon">
            {isError ? <AlertTriangle size={20} /> : <DownloadCloud size={20} />}
          </div>
          <div>
            <h2 id="loading-modal-title">{progress.title}</h2>
            <p>{progress.detail}</p>
          </div>
          <button className="icon-button" onClick={onClose} aria-label={isError ? "Dismiss load error" : "Cancel model load"}>
              <X size={20} />
            </button>
        </header>
        <div className="progress-track" aria-label="Model loading progress">
          <span className={hasTotal ? "" : "indeterminate"} style={hasTotal ? {
          width: `${percent}%`
        } : undefined} />
        </div>
        <div className="loading-modal-meta">
          <span>{hasTotal ? `${percent}%` : "Downloading"}</span>
          <span>{hasTotal ? `${loadedLabel} / ${formatBytes(totalBytes)}` : loadedLabel}</span>
        </div>
      </section>
    </div>;
}
function ModelBrowserModal({
  browserFitFilter,
  browserSort,
  fitContext,
  models,
  onClose,
  onFitFilterChange,
  onLoadRemoteModel,
  onSearch,
  onSortChange,
  status,
  query,
  onQueryChange
}) {
  return <div className="modal-backdrop" role="presentation" onMouseDown={event => {
    if (event.target === event.currentTarget) onClose();
  }}>
      <section className="model-browser-modal" role="dialog" aria-modal="true" aria-labelledby="model-browser-title">
        <header className="modal-header">
          <div>
            <h2 id="model-browser-title">Browse ONNX models</h2>
            <p>
              Browser parser · {fitContext.deviceMemoryGb ? `${fitContext.deviceMemoryGb} GB RAM hint` : "RAM hint unavailable"} · {fitContext.hardwareConcurrency ?? "?"} CPU threads
            </p>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close model browser">
            <X size={22} />
          </button>
        </header>

        <section className="modal-controls">
          <form className="modal-search-form" onSubmit={onSearch}>
            <label className="modal-search-box">
              <Search size={20} />
              <input value={query} onChange={event => onQueryChange(event.target.value)} placeholder="Search public ONNX models or paste a direct .onnx URL" autoFocus />
            </label>
            <button type="submit" className="modal-primary" disabled={status.state === "loading"}>
              Search
            </button>
          </form>

          <div className="modal-filter-grid">
            <label>
              <span>Sort</span>
              <select value={browserSort} onChange={event => onSortChange(event.target.value)}>
                <option value="fit">Best fit</option>
                <option value="downloads">Downloads</option>
                <option value="size">Smallest file</option>
                <option value="name">Name</option>
              </select>
            </label>
            <label>
              <span>Fit</span>
              <select value={browserFitFilter} onChange={event => onFitFilterChange(event.target.value)}>
                <option value="all">All fits</option>
                <option value="fits">Fits memory</option>
                <option value="borderline">Borderline</option>
                <option value="too-large">Too large</option>
              </select>
            </label>
            <div className={`web-status ${status.state}`}>
              {status.state === "error" ? <AlertTriangle size={14} /> : <Globe2 size={14} />}
              <span>{status.message}</span>
            </div>
          </div>
        </section>

        <div className="modal-results">
          {models.length ? models.map(({
          model,
          fit
        }) => <RemoteModelCard key={model.id} model={model} fit={fit} onLoad={onLoadRemoteModel} />) : <div className="modal-empty-results">
              <Globe2 size={26} />
              <strong>No ONNX models to show</strong>
              <span>Search Hugging Face or check a direct ONNX URL.</span>
            </div>}
        </div>
      </section>
    </div>;
}
function RemoteModelCard({
  model,
  fit,
  onLoad
}) {
  const sizeLabel = formatBytes(model.sizeBytes ?? model.estimatedBytes);
  const paramsLabel = model.parameterCountB ? `${model.parameterCountB < 1 ? `${Math.round(model.parameterCountB * 1000)}M` : `${model.parameterCountB}B`}` : null;
  const sourceLabel = model.source === "Direct URL" ? "Direct URL" : "Hugging Face";
  const popularity = [typeof model.downloads === "number" ? `${formatCount(model.downloads)} downloads` : null, typeof model.likes === "number" ? `${formatCount(model.likes)} likes` : null].filter(Boolean);
  return <article className={`remote-model-card ${fit.tone}`}>
      <div className="remote-model-copy">
        <div className="remote-model-title-row">
          <strong>{model.name}</strong>
          <a href={model.downloadUrl} target="_blank" rel="noreferrer" title="Open model file">
            <ExternalLink size={15} />
          </a>
          <span className={`fit-chip ${fit.tone}`}>{fit.label}</span>
          {paramsLabel && <span className="meta-chip">{paramsLabel}</span>}
          <span className="meta-chip">{sizeLabel}</span>
          <span className="meta-chip">ONNX</span>
        </div>
        <span className="remote-model-id">{model.modelId}</span>
        <div className="remote-model-chips">
          <span>{sourceLabel}</span>
          <span>{model.artifactPath}</span>
          <span>{fit.score}% fit</span>
          <span>{fit.workingSetLabel} working set</span>
          {popularity.map(item => <span key={item}>{item}</span>)}
        </div>
        <p>{fit.summary}</p>
      </div>
      <button type="button" className="use-model-button" onClick={() => onLoad(model)}>
        <DownloadCloud size={15} />
        {fit.tone === "red" ? "Load anyway" : "Use model"}
      </button>
    </article>;
}
function isDirectOnnxUrl(value) {
  if (!value) return false;
  try {
    const url = new URL(value);
    return (url.protocol === "https:" || url.protocol === "http:") && url.pathname.toLowerCase().endsWith(".onnx");
  } catch {
    return false;
  }
}
function formatCount(value) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(value);
}
export default App;
