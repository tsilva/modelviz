import React, { useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, GitBranch, Layers3, PanelLeft, ShieldCheck, X } from "lucide-react";
import { createArchitectureGroups, summarizeCoverage, classifyRawNode } from "../lib/abstraction.js";
import { createGraphIndex, deriveGroupEdges, flattenGroups, ownersByNode, visibleGroups } from "../lib/graph.js";
import GraphCanvas from "./GraphCanvas.jsx";
const recognitionLabels = {
  exact: "Exact operator",
  pattern: "Inferred pattern",
  scope: "Exporter scope",
  fused: "Fused operator"
};
export default function MappingWorkspace({
  modelView,
  query
}) {
  const index = useMemo(() => createGraphIndex(modelView.rawNodes, modelView.tensors, modelView.initializers), [modelView]);
  const roots = useMemo(() => createArchitectureGroups(modelView.rawNodes, modelView.model, index), [index, modelView.model]);
  const all = useMemo(() => flattenGroups(roots), [roots]);
  const searchLabels = useMemo(() => {
    const labels = new Map();
    for (const group of all) for (const id of group.rawNodeIds) {
      labels.set(id, `${labels.get(id) ?? ""} ${group.label} ${group.roles?.[id] ?? ""}`);
    }
    return labels;
  }, [all]);
  const [selectedId, setSelectedId] = useState(() => roots.find(g => g.kind !== "input")?.id ?? roots[0]?.id);
  const [activeRaw, setActiveRaw] = useState(null);
  const [expanded, setExpanded] = useState(new Set());
  const [mode, setMode] = useState("Split");
  const [navigatorOpen, setNavigatorOpen] = useState(false);
  const [inspectorTab, setInspectorTab] = useState("Overview");
  const [rawPage, setRawPage] = useState(0);
  const [focusKey, setFocusKey] = useState(0);
  const [selectionOnly, setSelectionOnly] = useState(false);
  const stage = useRef(null);
  const [rawAnchor, setRawAnchor] = useState(null);
  const [cleanAnchor, setCleanAnchor] = useState(null);
  const displayed = useMemo(() => visibleGroups(roots, expanded), [roots, expanded]);
  const edges = useMemo(() => deriveGroupEdges(displayed, index), [displayed, index]);
  const owners = useMemo(() => ownersByNode(displayed), [displayed]);
  const selected = all.find(group => group.id === selectedId) ?? roots[0];
  const selectedRawIds = useMemo(() => new Set(selected?.rawNodeIds ?? []), [selected]);
  const selectedVisible = useMemo(() => new Set(displayed.filter(group => group.rawNodeIds.some(id => selectedRawIds.has(id))).map(group => group.id)), [displayed, selectedRawIds]);
  const stats = summarizeCoverage(roots, modelView.rawNodes);
  const rawNodes = useMemo(() => modelView.rawNodes.filter(node => (!selectionOnly || selectedRawIds.has(node.id)) && (!query.trim() || `${node.name} ${node.opType} ${node.inputs.join(" ")} ${node.outputs.join(" ")} ${searchLabels.get(node.id) ?? ""}`.toLowerCase().includes(query.toLowerCase().trim()))).map(node => ({
    ...node,
    label: node.opType,
    kind: classifyRawNode(node)
  })), [modelView, query, selectionOnly, selectedRawIds, searchLabels]);
  const rawEdges = useMemo(() => {
    const ids = new Set(rawNodes.map(node => node.id));
    return index.edges.filter(edge => ids.has(edge.from) && ids.has(edge.to));
  }, [rawNodes, index]);
  const trace = selected?.rawNodeIds.map(id => index.byId.get(id)).filter(Boolean) ?? [];
  const selectedNode = index.byId.get(activeRaw);
  const selectedParents = selected ? all.filter(group => group.id !== selected.id && group.children?.length && selected.rawNodeIds.every(id => group.rawNodeIds.includes(id))) : [];
  const selectedRole = activeRaw ? all.find(group => group.roles?.[activeRaw])?.roles[activeRaw] : null;
  const chooseGroup = id => {
    setSelectedId(id);
    setActiveRaw(null);
    setRawPage(0);
    setFocusKey(value => value + 1);
  };
  const chooseRaw = id => {
    setActiveRaw(id);
    setRawPage(0);
    const owner = owners.get(id)?.[0];
    if (owner) setSelectedId(owner);
    setFocusKey(value => value + 1);
  };
  const toggle = id => {
    const next = new Set(expanded);
    if (id === null) next.clear();else if (next.has(id)) {
      next.delete(id);
      const parent = all.find(group => group.id === id);
      if (parent && selected?.rawNodeIds.every(rawId => parent.rawNodeIds.includes(rawId))) setSelectedId(id);
    } else next.add(id);
    setExpanded(next);
    if (id === null) {
      const parent = roots.find(group => selected?.rawNodeIds.every(rawId => group.rawNodeIds.includes(rawId)));
      if (parent) setSelectedId(parent.id);
    }
    setFocusKey(value => value + 1);
  };
  const tensors = [...(selected?.inputs ?? []).map(name => ({
    name,
    relation: "Input"
  })), ...(selected?.outputs ?? []).map(name => ({
    name,
    relation: "Output"
  })), ...(selected?.parameters ?? []).map(name => ({
    name,
    relation: "Parameter"
  }))];
  const stageRect = stage.current?.getBoundingClientRect();
  const link = stageRect && rawAnchor && cleanAnchor && mode === "Split" && stageRect.width > 720 ? {
    x1: rawAnchor.x - stageRect.left,
    y1: rawAnchor.y - stageRect.top,
    x2: cleanAnchor.x - stageRect.left,
    y2: cleanAnchor.y - stageRect.top
  } : null;
  return <section className="mapping-workspace">
    <div className="mapping-toolbar">
      <button className={`navigator-toggle ${navigatorOpen ? "selected" : ""}`} onClick={() => setNavigatorOpen(value => !value)} aria-expanded={navigatorOpen}><PanelLeft size={16} /> Architecture</button>
      <div className="mapping-summary"><strong>{modelView.model.family}</strong><span>{stats.rawNodes} operators <i>/</i> {roots.length} groups <i>/</i> {stats.recognizedNodes} operators in recognized patterns</span></div>
      <div className="view-toggle" aria-label="Graph view mode">{["Raw", "Clean", "Split"].map(item => <button key={item} onClick={() => setMode(item)} className={mode === item ? "active" : ""} aria-pressed={mode === item}>{item}</button>)}</div>
    </div>
    <div className={`mapping-main ${navigatorOpen ? "with-navigator" : ""}`}>
      {navigatorOpen && <aside className="architecture-navigator" aria-label="Architecture navigator"><header><strong>Architecture groups</strong><button className="canvas-icon" aria-label="Close architecture navigator" onClick={() => setNavigatorOpen(false)}><X size={16} /></button></header>
        <p>Expand a group to follow its primitives.</p><GroupTree groups={roots} expanded={expanded} selectedId={selectedId} onSelect={chooseGroup} onToggle={toggle} />
        <footer><ShieldCheck size={14} /> {stats.coverage}% of top-level graph items mapped</footer>
      </aside>}
      <div className={`mapping-stage mode-${mode.toLowerCase()}`} ref={stage}>
        {mode !== "Clean" && <div className="raw-canvas-wrapper"><GraphCanvas title="Raw ONNX graph" subtitle={selectionOnly ? "Selected group · exact tensor connections" : "Exact operators · select to trace architecture"} nodes={rawNodes} edges={rawEdges} selectedIds={selectedRawIds} activeId={activeRaw} onSelect={chooseRaw} focusKey={focusKey} onAnchor={setRawAnchor} raw />
          <label className="isolate-control"><input type="checkbox" checked={selectionOnly} onChange={event => setSelectionOnly(event.target.checked)} /> Selected group only</label>
        </div>}
        {mode !== "Raw" && <GraphCanvas title="Clean architecture" subtitle="Inferred groups · connections from tensor flow" nodes={displayed} edges={edges} selectedIds={selectedVisible} activeId={selectedId} onSelect={chooseGroup} onToggle={toggle} expanded={expanded} focusKey={focusKey} onAnchor={setCleanAnchor} />}
        {link && <svg className="cross-mapping" width="100%" height="100%" aria-hidden="true"><path d={`M ${link.x1} ${link.y1} C ${(link.x1 + link.x2) / 2} ${link.y1}, ${(link.x1 + link.x2) / 2} ${link.y2}, ${link.x2} ${link.y2}`} /><circle cx={link.x1} cy={link.y1} r="3" /><circle cx={link.x2} cy={link.y2} r="3" /></svg>}
      </div>
    </div>
    {selected && <section className="mapping-inspector" aria-label="Mapping inspector">
      <header className="mapping-inspector-header"><div><GitBranch size={18} />{selectedParents.length > 0 && <span className="selection-breadcrumb">{selectedParents.map(group => group.label).join(" / ")} /</span>}<strong>{selected.label}</strong><span className={`recognition-badge recognition-${selected.recognition}`}>{recognitionLabels[selected.recognition]}</span><span>{selected.rawNodeCount} contributing {selected.rawNodeCount === 1 ? "operator" : "operators"}</span></div>
        <nav aria-label="Inspector tabs">{["Overview", "Raw nodes", "Tensors"].map(tab => <button key={tab} className={inspectorTab === tab ? "active" : ""} onClick={() => setInspectorTab(tab)} aria-pressed={inspectorTab === tab}>{tab}</button>)}</nav>
      </header>
      {selectedNode && <div className="active-primitive"><strong>{selectedNode.domain || "ai.onnx"}::{selectedNode.opType}</strong>{selectedRole && <em>{selectedRole}</em>}<code>{selectedNode.name}</code><span>{selectedNode.inputs.length} inputs → {selectedNode.outputs.length} outputs</span><button aria-label="Clear operator selection" onClick={() => setActiveRaw(null)}><X size={13} /></button></div>}
      {inspectorTab === "Overview" && <div className="mapping-overview">
        <section><h3>Recognition evidence</h3>{selected.evidence.filter(Boolean).map(item => <p key={item}>{item}</p>)}<span className="muted-copy">Pattern labels describe a match; they are not calibrated confidence scores.</span></section>
        <section><h3>Boundary tensors</h3><TensorRows rows={tensors.filter(row => row.relation !== "Parameter")} index={index} /><span className="muted-copy">{selected.parameters.length} parameter tensors · <button className="text-action" onClick={() => setInspectorTab("Tensors")}>Inspect tensors</button></span></section>
        <section><h3>Contributing operators <span>{trace.length > 6 ? `first 6 of ${trace.length}` : trace.length}</span></h3><div className="trace-preview">{trace.slice(0, 6).map(node => <TraceRow key={node.id} node={node} active={activeRaw === node.id} onSelect={chooseRaw} />)}</div>{trace.length > 6 && <button className="text-action" onClick={() => setInspectorTab("Raw nodes")}>View all {trace.length} raw nodes →</button>}</section>
      </div>}
      {inspectorTab === "Raw nodes" && <div className="inspector-list-tab"><div className="trace-full">{trace.slice(rawPage * 30, (rawPage + 1) * 30).map(node => <TraceRow key={node.id} node={node} active={activeRaw === node.id} onSelect={chooseRaw} />)}</div><div className="list-pagination"><button disabled={!rawPage} onClick={() => setRawPage(value => value - 1)}>Previous</button><span>{Math.min(trace.length, rawPage * 30 + 1)}–{Math.min(trace.length, (rawPage + 1) * 30)} of {trace.length}</span><button disabled={(rawPage + 1) * 30 >= trace.length} onClick={() => setRawPage(value => value + 1)}>Next</button></div></div>}
      {inspectorTab === "Tensors" && <div className="inspector-list-tab"><h3>Group boundaries and parameters</h3><TensorRows rows={tensors} index={index} />{selectedNode && <><h3 className="tensor-subheading">Selected operator tensors</h3><TensorRows rows={[...selectedNode.inputs.filter(Boolean).map(name => ({
            name,
            relation: index.parameters.has(name) ? "Parameter" : "Input"
          })), ...selectedNode.outputs.filter(Boolean).map(name => ({
            name,
            relation: "Output"
          }))]} index={index} /><h3 className="tensor-subheading">Selected operator attributes</h3><div className="operator-attributes">{(selectedNode.attributes ?? []).map(attr => <p key={attr.name}><strong>{attr.name}</strong><code>{typeof attr.value === "object" ? JSON.stringify(attr.value, (key, value) => key === "nodes" ? `${value.length} operators (nested graph)` : value).slice(0, 200) : String(attr.value)}</code></p>)}{!selectedNode.attributes?.length && <p>No attributes recorded.</p>}</div></>}</div>}
    </section>}
  </section>;
}
function GroupTree({
  groups,
  expanded,
  selectedId,
  onSelect,
  onToggle
}) {
  return <ul className="group-tree">{groups.map(group => <li key={group.id}><div className={selectedId === group.id ? "selected" : ""}>
    {!!group.children?.length ? <button className="tree-expand" aria-label={`${expanded.has(group.id) ? "Collapse" : "Expand"} ${group.label} in navigator`} aria-expanded={expanded.has(group.id)} onClick={() => onToggle(group.id)}>{expanded.has(group.id) ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</button> : <span className="tree-indent" />}
    <button className="tree-select" onClick={() => onSelect(group.id)} aria-pressed={selectedId === group.id}><Layers3 size={14} /><span>{group.label}</span><small>{group.rawNodeCount}</small></button>
    </div>{expanded.has(group.id) && group.children?.length > 0 && <GroupTree groups={group.children} expanded={expanded} selectedId={selectedId} onSelect={onSelect} onToggle={onToggle} />}</li>)}</ul>;
}
function TraceRow({
  node,
  active,
  onSelect
}) {
  return <button className={`trace-row ${active ? "active" : ""}`} onClick={() => onSelect(node.id)} aria-label={`Trace operator ${node.name}`}><span>{node.opType}</span><code title={node.name}>{node.name}</code><ChevronRight size={13} /></button>;
}
function TensorRows({
  rows,
  index
}) {
  return <div className="boundary-tensors">{rows.map(({
      name,
      relation
    }) => {
      const info = index.tensorInfo.get(name);
      return <div key={`${relation}:${name}`} className="boundary-tensor"><small>{relation}</small><code title={name}>{name}</code><span>{info && info.shape !== "not recorded" ? `${info.type} [${info.shape}]` : info ? `${info.type} · shape not recorded` : "Shape not recorded"}</span></div>;
    })}{!rows.length && <p>No external tensor boundaries.</p>}</div>;
}
