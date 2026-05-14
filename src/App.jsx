import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Braces,
  CircuitBoard,
  Database,
  FileUp,
  GitBranch,
  Layers3,
  Link2,
  Network,
  PanelRight,
  Search,
  ShieldCheck,
  SlidersHorizontal
} from "lucide-react";
import { classifyRawNode, createArchitectureGroups, createCleanEdges, summarizeCoverage } from "./lib/abstraction.js";
import { createFixtureModelView, createModelViewFromOnnx } from "./lib/modelView.js";
import { parseOnnxModel } from "./lib/onnxParser.js";

const cleanLayout = {
  inputs: { x: 18, y: 22 },
  embeddings: { x: 18, y: 52 },
  ln1: { x: 52, y: 22 },
  attn0: { x: 80, y: 22 },
  mlp0: { x: 52, y: 78 },
  repeat: { x: 80, y: 52 },
  head: { x: 80, y: 82 }
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
  convolution: ["#7c2d12", "#ffedd5"],
  pooling: ["#0369a1", "#e0f2fe"],
  repeat: ["#0f7fbd", "#e0f2fe"],
  head: ["#be123c", "#ffe4e6"]
};

function App() {
  const [viewMode, setViewMode] = useState("Split");
  const [query, setQuery] = useState("");
  const [modelView, setModelView] = useState(() => createFixtureModelView());
  const [selectedGroupId, setSelectedGroupId] = useState("attn0");
  const [selectedProfile, setSelectedProfile] = useState("gpt2");
  const [loadStatus, setLoadStatus] = useState({ state: "idle", message: "Fixture model" });
  const inputRef = useRef(null);

  const groups = useMemo(() => createArchitectureGroups(modelView.rawNodes, modelView.model), [modelView]);
  const cleanEdges = useMemo(() => createCleanEdges(groups), [groups]);
  const selectedGroup = groups.find((group) => group.id === selectedGroupId) ?? groups[0];
  const selectedRawIds = new Set(selectedGroup.rawNodeIds);
  const coverage = useMemo(() => summarizeCoverage(groups, modelView.rawNodes), [groups, modelView.rawNodes]);
  const selectedTensors = useMemo(
    () => createSelectedTensorRows(selectedGroup, modelView.tensors),
    [modelView.tensors, selectedGroup]
  );

  const visibleRawNodes = useMemo(() => {
    if (!query.trim()) return modelView.rawNodes;
    const needle = query.toLowerCase();
    return modelView.rawNodes.filter((node) => `${node.name} ${node.opType} ${node.groupHint}`.toLowerCase().includes(needle));
  }, [modelView.rawNodes, query]);

  useEffect(() => {
    let cancelled = false;

    async function loadDefaultModel() {
      setLoadStatus({ state: "loading", message: "Loading default ONNX" });

      try {
        const response = await fetch("/api/model/default");
        if (!response.ok) throw new Error("Default ONNX endpoint unavailable");
        const fileName = response.headers.get("X-Model-File-Name") ?? "model.onnx";
        const sourcePath = response.headers.get("X-Model-Path") ?? "";
        const parsed = parseOnnxModel(await response.arrayBuffer());
        const nextView = createModelViewFromOnnx(fileName, parsed, sourcePath);
        if (cancelled) return;
        setModelView(nextView);
        setSelectedGroupId(nextView.rawNodes[0]?.groupId ?? "inputs");
        setSelectedProfile(nextView.modelProfiles[0]?.id ?? "generic");
        setLoadStatus({ state: "ready", message: nextView.loadMessage });
      } catch (error) {
        if (!cancelled) setLoadStatus({ state: "idle", message: error.message });
      }
    }

    loadDefaultModel();
    return () => {
      cancelled = true;
    };
  }, []);

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
        <input ref={inputRef} type="file" accept=".onnx,.json" hidden onChange={openFile} />
        <label className="search-box">
          <Search size={15} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search raw ops, tensors, groups" />
        </label>
        <div className="view-toggle" aria-label="Graph view mode">
          {["Raw", "Clean", "Split"].map((mode) => (
            <button key={mode} className={viewMode === mode ? "active" : ""} onClick={() => setViewMode(mode)}>{mode}</button>
          ))}
        </div>
        <div className="status-pill">
          <ShieldCheck size={14} />
          {coverage.coverage}% traced
        </div>
      </header>

      <section className="workspace">
        <aside className="left-panel">
          <div className="panel-title">
            <span>Architecture Profiles</span>
            <SlidersHorizontal size={15} />
          </div>
          <div className="model-card">
            <Network size={28} />
            <div>
              <strong>{modelView.model.fileName}</strong>
              <span>IR v{modelView.model.irVersion} · opset {modelView.model.opset} · {modelView.model.layers} ops</span>
            </div>
          </div>
          <div className={`load-note ${loadStatus.state}`}>
            <strong>{modelView.model.family}</strong>
            <span>{loadStatus.message}</span>
            {modelView.sourcePath && <code>{modelView.sourcePath}</code>}
          </div>
          <div className="profile-list">
            {modelView.modelProfiles.map((profile) => (
              <button key={profile.id} className={selectedProfile === profile.id ? "selected" : ""} onClick={() => setSelectedProfile(profile.id)}>
                <span>{profile.name}</span>
                <em>{profile.confidence}%</em>
              </button>
            ))}
          </div>
        </aside>

        <section className="center-panel">
          <div className="canvas-toolbar">
            <div>
              <strong>{viewMode} graph view</strong>
              <span>{coverage.rawNodes} graph items · {coverage.semanticGroups} semantic groups · {coverage.averageConfidence}% average confidence</span>
            </div>
          </div>

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
        </section>

        <aside className="right-panel">
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
            <div className="raw-id-list">
              {selectedGroup.rawNodeIds.map((id) => (
                <code key={id}>{id}</code>
              ))}
            </div>
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
        </aside>
      </section>
    </main>
  );
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
            style={{ left: `clamp(68px, ${layout.x}%, calc(100% - 68px))`, top: `clamp(50px, ${layout.y}%, calc(100% - 50px))`, "--group": color, "--group-soft": soft }}
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
