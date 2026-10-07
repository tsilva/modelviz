import React, { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Focus, Maximize2, Minus, Plus } from "lucide-react";
import { graphEdgePath, layoutGraph } from "../lib/graph.js";
export default function GraphCanvas({
  title,
  subtitle,
  nodes,
  edges,
  selectedIds,
  activeId,
  onSelect,
  onToggle,
  expanded,
  focusKey,
  onAnchor,
  raw = false
}) {
  const viewport = useRef(null);
  const drag = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [view, setView] = useState({
    width: 640,
    height: 450,
    left: 0,
    top: 0
  });
  const layout = useMemo(() => layoutGraph(nodes, edges), [nodes, edges]);
  const nodeById = useMemo(() => new Map(nodes.map(node => [node.id, node])), [nodes]);
  const horizontalInset = Math.max(0, (view.width - layout.width * zoom) / 2);
  useEffect(() => {
    const element = viewport.current;
    const update = () => setView({
      width: element.clientWidth,
      height: element.clientHeight,
      left: element.scrollLeft,
      top: element.scrollTop
    });
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const frame = ids => {
    const points = nodes.filter(node => ids?.has(node.id)).map(node => layout.points[node.id]);
    if (!points.length || !viewport.current) return;
    const left = Math.min(...points.map(p => p.x)) - 110;
    const top = Math.min(...points.map(p => p.y)) - 60;
    const width = Math.max(...points.map(p => p.x)) + 110 - left;
    const height = Math.max(...points.map(p => p.y)) + 60 - top;
    const viewportWidth = viewport.current.clientWidth;
    const viewportHeight = viewport.current.clientHeight;
    const nextZoom = Math.max(0.18, Math.min(1, (viewportWidth - 60) / width, (viewportHeight - 80) / height));
    setZoom(nextZoom);
    requestAnimationFrame(() => viewport.current?.scrollTo({
      left: Math.max(0, (left + width / 2) * nextZoom + Math.max(0, (viewportWidth - layout.width * nextZoom) / 2) - viewportWidth / 2),
      top: Math.max(0, (top + height / 2) * nextZoom - viewportHeight / 2)
    }));
  };
  useEffect(() => {
    frame(activeId && nodes.some(node => node.id === activeId) ? new Set([activeId]) : selectedIds);
  }, [focusKey, layout]);
  const selectedPoints = nodes.filter(node => selectedIds.has(node.id)).map(node => layout.points[node.id]);
  const selectionBounds = selectedPoints.length ? {
    left: Math.min(...selectedPoints.map(p => p.x)) - 103,
    top: Math.min(...selectedPoints.map(p => p.y)) - 55,
    right: Math.max(...selectedPoints.map(p => p.x)) + 103,
    bottom: Math.max(...selectedPoints.map(p => p.y)) + 55
  } : null;
  useEffect(() => {
    if (!onAnchor || !viewport.current) return;
    const rect = viewport.current.getBoundingClientRect();
    const bounds = selectionBounds;
    const visibleTop = bounds ? Math.max(0, bounds.top * zoom - view.top) : 0;
    const visibleBottom = bounds ? Math.min(view.height, bounds.bottom * zoom - view.top) : 0;
    const point = bounds && visibleBottom > visibleTop ? {
      x: rect.left + Math.max(10, Math.min(view.width - 10, (raw ? bounds.right : bounds.left) * zoom + horizontalInset - view.left)),
      y: rect.top + (visibleTop + visibleBottom) / 2
    } : null;
    onAnchor(previous => previous?.x === point?.x && previous?.y === point?.y ? previous : point);
  }, [layout, zoom, view, selectedIds, onAnchor]);
  const visible = point => point.x * zoom + horizontalInset + 125 > view.left && point.x * zoom + horizontalInset - 125 < view.left + view.width && point.y * zoom + 100 > view.top && point.y * zoom - 100 < view.top + view.height;
  const renderNodes = nodes.filter(node => visible(layout.points[node.id]));
  const startPan = event => {
    if (event.target.closest("button") || event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = {
      x: event.clientX,
      y: event.clientY,
      left: viewport.current.scrollLeft,
      top: viewport.current.scrollTop
    };
  };
  const movePan = event => {
    if (!drag.current) return;
    viewport.current.scrollTo(drag.current.left + drag.current.x - event.clientX, drag.current.top + drag.current.y - event.clientY);
  };
  const endPan = () => {
    drag.current = null;
  };
  return <section className={`mapping-canvas ${raw ? "primitive-canvas" : "architecture-canvas"}`} aria-label={title}>
      <header className="graph-heading"><div><strong>{title}</strong><span>{subtitle}</span></div>
        <button className="canvas-icon" title="Fit graph" aria-label={`Fit ${title}`} onClick={() => frame(new Set(nodes.map(node => node.id)))}><Maximize2 size={16} /></button>
      </header>
      <div className="mapping-viewport" ref={viewport} tabIndex={0} aria-label={`Pan and scroll ${title}`} onScroll={() => setView(current => ({
      ...current,
      left: viewport.current.scrollLeft,
      top: viewport.current.scrollTop
    }))} onPointerDown={startPan} onPointerMove={movePan} onPointerUp={endPan} onPointerCancel={endPan}>
        <div style={{
        width: Math.max(view.width, layout.width * zoom),
        height: Math.max(view.height, layout.height * zoom)
      }}>
          <div className="mapping-surface" style={{
          width: layout.width,
          height: layout.height,
          left: horizontalInset,
          transform: `scale(${zoom})`
        }}>
            {raw && selectionBounds && selectedPoints.length > 1 && <div className="selection-enclosure" style={{
            left: selectionBounds.left,
            top: selectionBounds.top,
            width: selectionBounds.right - selectionBounds.left,
            height: selectionBounds.bottom - selectionBounds.top
          }}><span>{selectedPoints.length} mapped operators</span></div>}
            <svg className="mapping-edges" width={layout.width} height={layout.height} aria-hidden="true">
              <defs><marker id={raw ? "raw-link-arrow" : "architecture-link-arrow"} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" /></marker></defs>
              {edges.map(edge => {
              const a = layout.points[edge.from];
              const b = layout.points[edge.to];
              if (!a || !b || !visible(a) && !visible(b) && (Math.max(a.y, b.y) * zoom < view.top || Math.min(a.y, b.y) * zoom > view.top + view.height)) return null;
              const selected = selectedIds.has(edge.from) && selectedIds.has(edge.to);
              return <path key={edge.id} className={selected ? "selected-link" : ""} d={graphEdgePath(a, b, nodeById.get(edge.from)?.children?.length ? 51 : 40, nodeById.get(edge.to)?.children?.length ? 51 : 40)} markerEnd={`url(#${raw ? "raw-link-arrow" : "architecture-link-arrow"})`}><title>{edge.tensor ?? edge.tensors?.join(", ")}</title></path>;
            })}
            </svg>
            {renderNodes.map(node => {
            const point = layout.points[node.id];
            return <div key={node.id} className={`graph-object kind-${node.kind ?? "compute"} ${selectedIds.has(node.id) ? "mapped" : ""} ${activeId === node.id ? "active" : ""}`} style={{
              left: point.x,
              top: point.y
            }}>
                <button className="graph-object-select" onClick={() => onSelect(node.id)} aria-label={`${raw ? "Operator" : "Block"}: ${node.label}`} aria-pressed={selectedIds.has(node.id)} title={node.name ?? node.label}>
                  <small>{raw ? node.domain || "ONNX" : node.recognition === "scope" ? "Exporter scope" : node.kind}</small>
                  <strong>{node.label}</strong><span>{raw ? node.name : node.role ?? (["input", "output"].includes(node.kind) ? `Graph ${node.kind}` : `${node.rawNodeCount} ${node.rawNodeCount === 1 ? "operator" : "operators"}`)}</span>
                </button>
                {!!node.children?.length && <button className="graph-expand" onClick={() => onToggle(node.id)} aria-label={`Expand ${node.label}`} aria-expanded={expanded.has(node.id)}><ChevronRight size={14} /> Expand</button>}
              </div>;
          })}
          </div>
        </div>
      </div>
      {!nodes.length && <div className="graph-no-results">No operators match this search.</div>}
      <div className="mapping-zoom">
        <button aria-label={`Zoom out ${title}`} onClick={() => setZoom(value => Math.max(.18, value - .1))}><Minus size={14} /></button>
        <span>{Math.round(zoom * 100)}%</span>
        <button aria-label={`Zoom in ${title}`} onClick={() => setZoom(value => Math.min(1.6, value + .1))}><Plus size={14} /></button>
        <i /> <button aria-label={`Frame selected in ${title}`} title="Frame selected" onClick={() => frame(activeId ? new Set([activeId]) : selectedIds)}><Focus size={16} /></button>
      </div>
      {onToggle && expanded.size > 0 && <button className="collapse-graph" onClick={() => onToggle(null)}><ChevronDown size={14} /> Collapse all</button>}
    </section>;
}
