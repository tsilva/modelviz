import React, { useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Braces,
  CheckCircle2,
  CircuitBoard,
  Database,
  DownloadCloud,
  ExternalLink,
  FileUp,
  GitBranch,
  Globe2,
  HardDrive,
  Layers3,
  Link2,
  Network,
  PanelRight,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  X
} from "lucide-react";
import { classifyRawNode, createArchitectureGroups, createCleanEdges, summarizeCoverage } from "./lib/abstraction.js";
import { createEmptyModelView, createModelViewFromOnnx } from "./lib/modelView.js";
import { parseOnnxModel } from "./lib/onnxParser.js";
import {
  assessRemoteModelFit,
  createRemoteModelFromUrl,
  formatBytes,
  getBrowserFitContext,
  searchWebOnnxModels
} from "./lib/webModelBrowser.js";

const cleanLayout = {
  inputs: { x: 18, y: 22 },
  embeddings: { x: 18, y: 52 },
  ln1: { x: 52, y: 22 },
  attn0: { x: 80, y: 22 },
  mlp0: { x: 52, y: 78 },
  repeat: { x: 80, y: 52 },
  head: { x: 80, y: 82 },
  runtime_support: { x: 28, y: 78 },
  transformer_blocks: { x: 50, y: 46 },
  final_norm: { x: 72, y: 24 },
  lm_head: { x: 84, y: 46 },
  outputs: { x: 72, y: 68 }
};

const groupColors = {
  input: ["#475569", "#f1f5f9"],
  output: ["#be123c", "#ffe4e6"],
  embedding: ["#2563eb", "#dbeafe"],
  norm: ["#7c3aed", "#ede9fe"],
  normalization: ["#7c3aed", "#ede9fe"],
  attention: ["#0f766e", "#ccfbf1"],
  mlp: ["#9333ea", "#f3e8ff"],
  dense: ["#0f766e", "#ccfbf1"],
  activation: ["#be123c", "#ffe4e6"],
  layout: ["#0f7fbd", "#e0f2fe"],
  compute: ["#334155", "#f1f5f9"],
  support: ["#64748b", "#f8fafc"],
  convolution: ["#7c2d12", "#ffedd5"],
  pooling: ["#0369a1", "#e0f2fe"],
  repeat: ["#0f7fbd", "#e0f2fe"],
  head: ["#be123c", "#ffe4e6"]
};

function App() {
  const [viewMode, setViewMode] = useState("Split");
  const [query, setQuery] = useState("");
  const [modelView, setModelView] = useState(() => createEmptyModelView());
  const [selectedGroupId, setSelectedGroupId] = useState(null);
  const [selectedProfile, setSelectedProfile] = useState(null);
  const [loadStatus, setLoadStatus] = useState({ state: "idle", message: "Load an ONNX model file to begin" });
  const [webQuery, setWebQuery] = useState("mnist onnx");
  const [webResults, setWebResults] = useState([]);
  const [webStatus, setWebStatus] = useState({ state: "idle", message: "Search public ONNX files" });
  const [directUrl, setDirectUrl] = useState("");
  const [directModel, setDirectModel] = useState(null);
  const [modelBrowserOpen, setModelBrowserOpen] = useState(false);
  const [browserSort, setBrowserSort] = useState("fit");
  const [browserFitFilter, setBrowserFitFilter] = useState("all");
  const inputRef = useRef(null);
  const webSearchAbortRef = useRef(null);
  const directAbortRef = useRef(null);

  const groups = useMemo(() => createArchitectureGroups(modelView.rawNodes, modelView.model), [modelView]);
  const cleanEdges = useMemo(() => createCleanEdges(groups), [groups]);
  const hasModel = Boolean(modelView.model.fileName);
  const selectedGroup = groups.find((group) => group.id === selectedGroupId) ?? groups[0] ?? null;
  const selectedRawIds = new Set(selectedGroup?.rawNodeIds ?? []);
  const coverage = useMemo(() => summarizeCoverage(groups, modelView.rawNodes), [groups, modelView.rawNodes]);
  const selectedTensors = useMemo(
    () => (selectedGroup ? createSelectedTensorRows(selectedGroup, modelView.tensors) : []),
    [modelView.tensors, selectedGroup]
  );

  const visibleRawNodes = useMemo(() => {
    if (!query.trim()) return modelView.rawNodes;
    const needle = query.toLowerCase();
    return modelView.rawNodes.filter((node) => `${node.name} ${node.opType} ${node.groupHint}`.toLowerCase().includes(needle));
  }, [modelView.rawNodes, query]);

  const fitContext = useMemo(() => getBrowserFitContext(), []);
  const remoteModels = useMemo(() => {
    const combined = directModel ? [directModel, ...webResults.filter((model) => model.id !== directModel.id)] : webResults;
    const withFit = combined.map((model) => ({ model, fit: assessRemoteModelFit(model, fitContext) }));
    const filtered = withFit.filter(({ fit }) => {
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

  const openFile = async (event) => {
    const [file] = event.target.files ?? [];
    if (!file) return;

    setLoadStatus({ state: "loading", message: `Loading ${file.name}` });

    try {
      const parsed = parseOnnxModel(await file.arrayBuffer());
      const nextView = createModelViewFromOnnx(file.name, parsed);
      setModelView(nextView);
      setSelectedGroupId(nextView.rawNodes[0]?.groupId ?? "inputs");
      setSelectedProfile(nextView.modelProfiles[0]?.id ?? "generic");
      setLoadStatus({ state: "ready", message: nextView.loadMessage });
    } catch (error) {
      setLoadStatus({ state: "error", message: error.message });
    } finally {
      event.target.value = "";
    }
  };

  const loadModelBuffer = async ({ fileName, sourcePath, buffer }) => {
    const parsed = parseOnnxModel(buffer);
    const nextView = createModelViewFromOnnx(fileName, parsed, sourcePath);
    setModelView(nextView);
    setSelectedGroupId(nextView.rawNodes[0]?.groupId ?? "inputs");
    setSelectedProfile(nextView.modelProfiles[0]?.id ?? "generic");
    setLoadStatus({ state: "ready", message: nextView.loadMessage });
  };

  const searchWebModels = async (event) => {
    event?.preventDefault();
    webSearchAbortRef.current?.abort();
    const controller = new AbortController();
    webSearchAbortRef.current = controller;
    setWebStatus({ state: "loading", message: "Searching Hugging Face ONNX models" });

    try {
      const results = await searchWebOnnxModels(webQuery, controller.signal);
      setWebResults(results);
      setWebStatus({
        state: "ready",
        message: results.length ? `${results.length} ONNX files found` : "No ONNX files found"
      });
    } catch (error) {
      if (controller.signal.aborted) return;
      setWebStatus({ state: "error", message: error.message });
    }
  };

  const openModelBrowser = () => {
    setModelBrowserOpen(true);
    if (webResults.length === 0 && webStatus.state === "idle") {
      searchWebModels();
    }
  };

  const checkDirectUrl = async (event) => {
    event?.preventDefault();
    directAbortRef.current?.abort();
    const controller = new AbortController();
    directAbortRef.current = controller;
    setDirectModel(null);
    setWebStatus({ state: "loading", message: "Checking remote ONNX URL" });

    try {
      const remote = await createRemoteModelFromUrl(directUrl, controller.signal);
      setDirectModel(remote);
      setWebStatus({ state: "ready", message: "Remote URL assessed" });
    } catch (error) {
      if (controller.signal.aborted) return;
      setWebStatus({ state: "error", message: error.message });
    }
  };

  const loadRemoteModel = async (remoteModel) => {
    setLoadStatus({ state: "loading", message: `Downloading ${remoteModel.artifactPath}` });

    try {
      const response = await fetch(remoteModel.downloadUrl);
      if (!response.ok) throw new Error(`Remote model download failed (${response.status})`);
      const buffer = await response.arrayBuffer();
      await loadModelBuffer({
        fileName: remoteModel.artifactPath.split("/").pop() || remoteModel.name || "model.onnx",
        sourcePath: remoteModel.downloadUrl,
        buffer
      });
      setWebStatus({ state: "ready", message: `Loaded ${remoteModel.modelId}` });
      setModelBrowserOpen(false);
    } catch (error) {
      setLoadStatus({ state: "error", message: error.message });
      setWebStatus({ state: "error", message: error.message });
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><CircuitBoard size={19} /></span>
          <div>
            <strong>ModelViz</strong>
            <span>ONNX architecture explorer</span>
          </div>
        </div>
        <button className="primary" onClick={() => inputRef.current?.click()}>
          <FileUp size={16} />
          Open ONNX
        </button>
        <button className="secondary-topbar" onClick={openModelBrowser}>
          <Globe2 size={16} />
          Browse web
        </button>
        <input ref={inputRef} type="file" accept=".onnx,.json" hidden onChange={openFile} />
        <label className="search-box">
          <Search size={15} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={hasModel ? "Search raw ops, tensors, groups" : "Load a model before searching"}
            disabled={!hasModel}
          />
        </label>
        <div className="view-toggle" aria-label="Graph view mode">
          {["Raw", "Clean", "Split"].map((mode) => (
            <button key={mode} className={viewMode === mode ? "active" : ""} onClick={() => setViewMode(mode)} disabled={!hasModel}>{mode}</button>
          ))}
        </div>
        <div className="status-pill">
          <ShieldCheck size={14} />
          {hasModel ? `${coverage.coverage}% traced` : "No model"}
        </div>
      </header>

      <section className="workspace">
        <aside className="left-panel">
          <div className="panel-title">
            <span>Architecture Profiles</span>
            <SlidersHorizontal size={15} />
          </div>
          {hasModel ? (
            <div className="model-card">
              <Network size={28} />
              <div>
                <strong>{modelView.model.fileName}</strong>
                <span>IR v{modelView.model.irVersion} · opset {modelView.model.opset} · {modelView.model.layers} ops</span>
              </div>
            </div>
          ) : (
            <div className="model-card empty-card">
              <FileUp size={28} />
              <div>
                <strong>No model loaded</strong>
                <span>Open an ONNX file to populate profiles</span>
              </div>
            </div>
          )}
          <div className={`load-note ${loadStatus.state}`}>
            <strong>{hasModel ? modelView.model.family : "Waiting for model file"}</strong>
            <span>{loadStatus.message}</span>
            {modelView.sourcePath && <code>{modelView.sourcePath}</code>}
          </div>
          {hasModel ? (
            <div className="profile-list">
              {modelView.modelProfiles.map((profile) => (
                <button key={profile.id} className={selectedProfile === profile.id ? "selected" : ""} onClick={() => setSelectedProfile(profile.id)}>
                  <span>{profile.name}</span>
                  <em>{profile.confidence}%</em>
                </button>
              ))}
            </div>
          ) : (
            <div className="empty-list">Profiles appear after a model is parsed.</div>
          )}
        </aside>

        <section className={`center-panel ${hasModel ? "" : "empty"}`}>
          <div className="canvas-toolbar">
            <div>
              <strong>{hasModel ? `${viewMode} graph view` : "Load a model to inspect"}</strong>
              <span>
                {hasModel
                  ? `${coverage.rawNodes} graph items · ${coverage.semanticGroups} semantic groups · ${coverage.averageConfidence}% average confidence`
                  : "Open an ONNX file to parse raw operators, tensors, and architecture groups."}
              </span>
            </div>
          </div>

          {hasModel ? (
            <>
              <div className={`graph-stage mode-${viewMode.toLowerCase()}`}>
                {viewMode !== "Clean" && (
                  <section className="graph-pane raw-pane">
                    <div className="pane-label"><Braces size={15} /> Raw ONNX graph</div>
                    <RawGraph nodes={visibleRawNodes} selectedRawIds={selectedRawIds} />
                  </section>
                )}
                {viewMode !== "Raw" && (
                  <section className="graph-pane clean-pane">
                    <div className="pane-label"><Layers3 size={15} /> Clean architecture</div>
                    <CleanGraph groups={groups} edges={cleanEdges} selectedGroupId={selectedGroupId} onSelect={setSelectedGroupId} />
                  </section>
                )}
              </div>

              <section className="group-table-panel">
                <div className="panel-title">
                  <span>Semantic Groups</span>
                </div>
                <div className="group-table">
                  <div className="table-row head">
                    <span>Group</span><span>Kind</span><span>Raw nodes</span><span>Confidence</span><span>Recognizer</span><span>Output</span>
                  </div>
                  {groups.map((group) => (
                    <button key={group.id} className={`table-row ${selectedGroupId === group.id ? "selected" : ""}`} onClick={() => setSelectedGroupId(group.id)}>
                      <span>{group.label}</span>
                      <span>{group.kind}</span>
                      <span>{group.rawNodeCount}</span>
                      <span>{group.confidence}%</span>
                      <span>{group.recognizer}</span>
                      <span>{group.outputs.at(-1) ?? "n/a"}</span>
                    </button>
                  ))}
                </div>
              </section>
            </>
          ) : (
            <section className="empty-state-panel">
              <div className="empty-state-content">
                <FileUp size={34} />
                <strong>Load an ONNX model file first</strong>
                <span>ModelViz starts empty and only renders graph data parsed from a file you choose.</span>
                <div className="empty-actions">
                  <button className="primary" onClick={() => inputRef.current?.click()}>
                    <FileUp size={16} />
                    Open ONNX
                  </button>
                  <button className="secondary-action" onClick={openModelBrowser}>
                    <Globe2 size={16} />
                    Browse web
                  </button>
                </div>
              </div>
            </section>
          )}
        </section>

        <aside className="right-panel">
          {hasModel && selectedGroup ? (
            <>
              <div className="inspector-heading">
                <span><PanelRight size={18} /></span>
                <div>
                  <strong>{selectedGroup.label}</strong>
                  <small>{selectedGroup.kind} · {selectedGroup.rawNodeCount} raw nodes · {selectedGroup.confidence}% confidence</small>
                </div>
              </div>
              <section className="inspector-card">
                <div className="panel-title">
                  <span>Metadata</span>
                  <Link2 size={14} />
                </div>
                {Object.entries(selectedGroup.metadata).map(([key, value]) => (
                  <div className="kv-row" key={key}>
                    <span>{key}</span>
                    <code>{String(value)}</code>
                  </div>
                ))}
              </section>
              <section className="inspector-card">
                <div className="panel-title">
                  <span>Traceability</span>
                  <GitBranch size={14} />
                </div>
                <TraceabilityList ids={selectedGroup.rawNodeIds} />
              </section>
              <section className="tensor-card">
                <div className="panel-title">
                  <span>Group Tensors</span>
                  <Database size={14} />
                </div>
                {selectedTensors.map(({ name, type, shape, relation }) => (
                  <div className="tensor-row" key={`${relation}-${name}`}>
                    <strong>{name}</strong>
                    <span>{relation} · {type} · {shape}</span>
                  </div>
                ))}
              </section>
            </>
          ) : (
            <>
              <div className="inspector-heading muted-heading">
                <span><PanelRight size={18} /></span>
                <div>
                  <strong>Model inspector</strong>
                  <small>Load a model file to inspect groups and tensors</small>
                </div>
              </div>
              <section className="inspector-card inspector-empty">
                <div className="panel-title">
                  <span>Metadata</span>
                  <Link2 size={14} />
                </div>
                <p>No parsed model data yet.</p>
              </section>
            </>
          )}
        </aside>
      </section>
      {modelBrowserOpen && (
        <ModelBrowserModal
          browserFitFilter={browserFitFilter}
          browserSort={browserSort}
          directUrl={directUrl}
          fitContext={fitContext}
          models={remoteModels}
          onCheckDirectUrl={checkDirectUrl}
          onClose={() => setModelBrowserOpen(false)}
          onDirectUrlChange={setDirectUrl}
          onFitFilterChange={setBrowserFitFilter}
          onLoadRemoteModel={loadRemoteModel}
          onSearch={searchWebModels}
          onSortChange={setBrowserSort}
          status={webStatus}
          query={webQuery}
          onQueryChange={setWebQuery}
        />
      )}
    </main>
  );
}

function ModelBrowserModal({
  browserFitFilter,
  browserSort,
  directUrl,
  fitContext,
  models,
  onCheckDirectUrl,
  onClose,
  onDirectUrlChange,
  onFitFilterChange,
  onLoadRemoteModel,
  onSearch,
  onSortChange,
  status,
  query,
  onQueryChange
}) {
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
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
              <input value={query} onChange={(event) => onQueryChange(event.target.value)} placeholder="Search public ONNX models" autoFocus />
            </label>
            <button type="submit" className="modal-primary" disabled={status.state === "loading"}>
              Search
            </button>
          </form>

          <form className="direct-url-row" onSubmit={onCheckDirectUrl}>
            <label>
              <span>Direct ONNX URL</span>
              <input value={directUrl} onChange={(event) => onDirectUrlChange(event.target.value)} placeholder="https://.../model.onnx" />
            </label>
            <button type="submit" className="modal-secondary" disabled={!directUrl.trim() || status.state === "loading"}>
              <CheckCircle2 size={15} />
              Check
            </button>
          </form>

          <div className="modal-filter-grid">
            <label>
              <span>Sort</span>
              <select value={browserSort} onChange={(event) => onSortChange(event.target.value)}>
                <option value="fit">Best fit</option>
                <option value="downloads">Downloads</option>
                <option value="size">Smallest file</option>
                <option value="name">Name</option>
              </select>
            </label>
            <label>
              <span>Fit</span>
              <select value={browserFitFilter} onChange={(event) => onFitFilterChange(event.target.value)}>
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
          {models.length ? (
            models.map(({ model, fit }) => (
              <RemoteModelCard key={model.id} model={model} fit={fit} onLoad={onLoadRemoteModel} />
            ))
          ) : (
            <div className="modal-empty-results">
              <Globe2 size={26} />
              <strong>No ONNX models to show</strong>
              <span>Search Hugging Face or check a direct ONNX URL.</span>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function RemoteModelCard({ model, fit, onLoad }) {
  const sizeLabel = formatBytes(model.sizeBytes ?? model.estimatedBytes);
  const paramsLabel = model.parameterCountB ? `${model.parameterCountB < 1 ? `${Math.round(model.parameterCountB * 1000)}M` : `${model.parameterCountB}B`}` : null;
  const sourceLabel = model.source === "Direct URL" ? "Direct URL" : "Hugging Face";
  const popularity = [
    typeof model.downloads === "number" ? `${formatCount(model.downloads)} downloads` : null,
    typeof model.likes === "number" ? `${formatCount(model.likes)} likes` : null
  ].filter(Boolean);

  return (
    <article className={`remote-model-card ${fit.tone}`}>
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
          {popularity.map((item) => <span key={item}>{item}</span>)}
        </div>
        <p>{fit.summary}</p>
      </div>
      <button type="button" className="use-model-button" onClick={() => onLoad(model)}>
        <DownloadCloud size={15} />
        {fit.tone === "red" ? "Load anyway" : "Use model"}
      </button>
    </article>
  );
}

function TraceabilityList({ ids }) {
  const visibleIds = ids.slice(0, 24);
  const hiddenCount = ids.length - visibleIds.length;

  return (
    <div className="raw-id-list">
      {visibleIds.map((id) => (
        <code key={id}>{id}</code>
      ))}
      {hiddenCount > 0 && <span className="raw-id-more">+{hiddenCount} more raw nodes</span>}
    </div>
  );
}

function formatCount(value) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(value);
}

function RawGraph({ nodes, selectedRawIds }) {
  const edges = useMemo(() => createRawEdges(nodes), [nodes]);

  return (
    <div className="raw-graph">
      <svg viewBox="0 0 100 100" className="edges" preserveAspectRatio="none" aria-hidden="true">
        {edges.map((edge) => (
          <path key={edge.id} className={edge.traced ? "trace-edge" : ""} d={edge.path} />
        ))}
      </svg>
      {nodes.map((node) => {
        const selected = selectedRawIds.has(node.id);
        return (
          <button
            key={node.id}
            className={`raw-node ${classifyRawNode(node)} ${selected ? "traced" : ""}`}
            style={{ left: `clamp(54px, ${node.x}%, calc(100% - 54px))`, top: `clamp(44px, ${node.y}%, calc(100% - 44px))` }}
          >
            <strong>{node.opType}</strong>
            <span>{node.name.split("/").pop()}</span>
          </button>
        );
      })}
    </div>
  );
}

function CleanGraph({ groups, edges, selectedGroupId, onSelect }) {
  const layoutById = useMemo(() => createCleanLayout(groups), [groups]);

  return (
    <div className="clean-graph">
      <svg viewBox="0 0 100 100" className="edges" preserveAspectRatio="none" aria-hidden="true">
        {edges.map(([from, to]) => {
          const a = layoutById[from];
          const b = layoutById[to];
          if (!a || !b) return null;
          return <path key={`${from}-${to}`} d={`M${a.x + 5} ${a.y} C${a.x + 10} ${a.y}, ${b.x - 10} ${b.y}, ${b.x - 5} ${b.y}`} />;
        })}
      </svg>
      {groups.map((group) => {
        const layout = layoutById[group.id];
        const [color, soft] = groupColors[group.kind] ?? groupColors.input;
        return (
          <button
            key={group.id}
            className={`clean-node ${selectedGroupId === group.id ? "selected" : ""}`}
            style={{ left: `clamp(78px, ${layout.x}%, calc(100% - 78px))`, top: `clamp(50px, ${layout.y}%, calc(100% - 50px))`, "--group": color, "--group-soft": soft }}
            onClick={() => onSelect(group.id)}
          >
            <span>{group.kind}</span>
            <strong>{group.label}</strong>
            <em>{group.confidence}% · {group.rawNodeCount} raw</em>
          </button>
        );
      })}
    </div>
  );
}

function createRawEdges(nodes) {
  const byOutput = new Map();
  nodes.forEach((node) => {
    (node.outputs ?? []).forEach((output) => byOutput.set(output, node));
  });

  return nodes.flatMap((node) =>
    (node.inputs ?? []).flatMap((input) => {
      const source = byOutput.get(input);
      if (!source || source.id === node.id) return [];
      const path = `M${source.x} ${source.y} C${source.x + 8} ${source.y}, ${node.x - 8} ${node.y}, ${node.x} ${node.y}`;
      return [{ id: `${source.id}-${node.id}-${input}`, path, traced: selectedEdge(source, node) }];
    })
  );
}

function selectedEdge(source, target) {
  return source.groupId && source.groupId === target.groupId;
}

function createCleanLayout(groups) {
  const staticLayoutApplies = groups.every((group) => cleanLayout[group.id]);
  if (staticLayoutApplies) return cleanLayout;

  const columns = groups.length <= 4 ? groups.length : Math.min(4, Math.ceil(Math.sqrt(groups.length)));
  const rows = Math.ceil(groups.length / columns);

  return Object.fromEntries(
    groups.map((group, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const x = 14 + (columns === 1 ? 36 : (column / (columns - 1)) * 72);
      const y = rows === 1 ? 52 : 33 + (row / Math.max(rows - 1, 1)) * 38;
      return [group.id, { x, y }];
    })
  );
}

function createSelectedTensorRows(group, tensorRows) {
  const knownTensors = new Map(
    tensorRows.map(([name, type, shape, role]) => [name, { name, type, shape, role }])
  );
  const seen = new Set();

  const makeRow = (name, relation) => {
    if (!name || seen.has(`${relation}-${name}`)) return null;
    seen.add(`${relation}-${name}`);

    const known = knownTensors.get(name);
    const displayedRelation = known?.role === "initializer" ? "parameter" : relation;

    return {
      name,
      relation: displayedRelation,
      type: known?.type ?? "tensor",
      shape: known?.shape ?? "intermediate"
    };
  };

  return [
    ...(group.inputs ?? []).map((name) => makeRow(name, "input")),
    ...(group.outputs ?? []).map((name) => makeRow(name, "output"))
  ].filter(Boolean);
}

export default App;
