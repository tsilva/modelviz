// Tensor names define dataflow. Node names are display metadata, never identity.
export function createGraphIndex(nodes, tensors = [], initializers = []) {
  const byId = new Map(nodes.map(node => [node.id, node]));
  const producer = new Map();
  const consumers = new Map();
  const tensorInfo = new Map(tensors.map(([name, type, shape, role]) => [name, {
    name,
    type,
    shape,
    role
  }]));
  const parameters = new Map(initializers.map(tensor => [tensor.name, tensor]));
  for (const node of nodes) {
    for (const name of node.outputs ?? []) {
      if (name) producer.set(name, node.id);
    }
    for (const name of node.inputs ?? []) {
      if (!name) continue;
      if (!consumers.has(name)) consumers.set(name, []);
      consumers.get(name).push(node.id);
    }
  }
  const edges = nodes.flatMap(target => (target.inputs ?? []).flatMap((tensor, inputIndex) => {
    const from = producer.get(tensor);
    return from && from !== target.id ? [{
      id: `${from}:${target.id}:${inputIndex}`,
      from,
      to: target.id,
      tensor
    }] : [];
  }));
  const constantValues = new Set(parameters.keys());
  for (const node of nodes) if ((!node.domain || node.domain === "ai.onnx") && node.opType === "Constant") {
    node.outputs.filter(Boolean).forEach(name => constantValues.add(name));
  }
  const constantTransforms = new Set(["Reshape", "Transpose", "Flatten", "Cast", "Squeeze", "Unsqueeze", "Concat", "Slice", "Gather", "Shape", "Add", "Sub", "Mul", "Div"]);
  const queue = [...constantValues];
  for (let offset = 0; offset < queue.length; offset += 1) {
    for (const id of consumers.get(queue[offset]) ?? []) {
      const node = byId.get(id);
      if (node.domain && node.domain !== "ai.onnx") continue;
      if (!constantTransforms.has(node.opType) || !node.inputs.filter(Boolean).every(name => constantValues.has(name))) continue;
      for (const name of node.outputs.filter(Boolean)) if (!constantValues.has(name)) {
        constantValues.add(name);
        queue.push(name);
      }
    }
  }
  return {
    nodes,
    byId,
    producer,
    consumers,
    edges,
    tensorInfo,
    parameters,
    constantValues
  };
}
export function groupBoundary(ids, index) {
  const members = new Set(ids);
  const inputs = new Set();
  const outputs = new Set();
  const parameters = new Set();
  for (const id of ids) {
    const node = index.byId.get(id);
    for (const name of node?.inputs ?? []) {
      if (!name || members.has(index.producer.get(name))) continue;
      if (index.constantValues.has(name)) parameters.add(name);else inputs.add(name);
    }
    for (const name of node?.outputs ?? []) {
      if (name && (index.consumers.get(name) ?? []).some(consumer => !members.has(consumer))) outputs.add(name);
    }
  }
  return {
    inputs: [...inputs],
    outputs: [...outputs],
    parameters: [...parameters]
  };
}
export function flattenGroups(groups) {
  return groups.flatMap(group => [group, ...flattenGroups(group.children ?? [])]);
}
export function visibleGroups(groups, expanded) {
  return groups.flatMap(group => group.children?.length && expanded.has(group.id) ? visibleGroups(group.children, expanded) : [group]);
}
export function ownersByNode(groups) {
  const owners = new Map();
  for (const group of groups) for (const id of group.rawNodeIds) {
    if (!owners.has(id)) owners.set(id, []);
    owners.get(id).push(group.id);
  }
  return owners;
}

// Derive the quotient graph from real tensors crossing the visible groups.
export function deriveGroupEdges(groups, index) {
  const owners = ownersByNode(groups);
  const edges = new Map();
  for (const edge of index.edges) {
    for (const from of owners.get(edge.from) ?? []) for (const to of owners.get(edge.to) ?? []) {
      if (from === to) continue;
      const key = `${from}:${to}`;
      if (!edges.has(key)) edges.set(key, {
        id: key,
        from,
        to,
        tensors: []
      });
      const tensors = edges.get(key).tensors;
      if (!tensors.includes(edge.tensor)) tensors.push(edge.tensor);
    }
  }
  return [...edges.values()];
}

// Layered DAG layout with a bounded number of columns; long graphs scroll vertically.
export function layoutGraph(nodes, edges) {
  const ids = new Set(nodes.map(node => node.id));
  const next = new Map(nodes.map(node => [node.id, []]));
  const incoming = new Map(nodes.map(node => [node.id, 0]));
  const rank = new Map(nodes.map(node => [node.id, 0]));
  const pairs = new Set();
  for (const edge of edges) {
    const key = `${edge.from}:${edge.to}`;
    if (!ids.has(edge.from) || !ids.has(edge.to) || pairs.has(key)) continue;
    pairs.add(key);
    next.get(edge.from).push(edge.to);
    incoming.set(edge.to, incoming.get(edge.to) + 1);
  }
  const queue = nodes.filter(node => incoming.get(node.id) === 0).map(node => node.id);
  for (let i = 0; i < queue.length; i += 1) {
    const id = queue[i];
    for (const to of next.get(id)) {
      rank.set(to, Math.max(rank.get(to), rank.get(id) + 1));
      incoming.set(to, incoming.get(to) - 1);
      if (incoming.get(to) === 0) queue.push(to);
    }
  }
  const rows = new Map();
  for (const node of nodes) {
    const row = rank.get(node.id);
    if (!rows.has(row)) rows.set(row, []);
    rows.get(row).push(node);
  }
  const columns = Math.max(1, Math.min(5, Math.max(0, ...[...rows.values()].map(row => row.length))));
  const width = Math.max(480, columns * 218 + 80);
  let y = 100;
  const points = {};
  for (const [, row] of [...rows].sort(([a], [b]) => a - b)) {
    for (let offset = 0; offset < row.length; offset += 5) {
      const chunk = row.slice(offset, offset + 5);
      chunk.forEach((node, column) => {
        points[node.id] = {
          x: width / 2 + (column - (chunk.length - 1) / 2) * 218,
          y
        };
      });
      y += 116;
    }
  }
  return {
    points,
    width,
    height: Math.max(380, y + 35)
  };
}

export function graphEdgePath(source, target, sourceHalfHeight = 40, targetHalfHeight = 40) {
  // Route skip connections beside the column, so they remain visible around nodes.
  if (Math.abs(target.x - source.x) < 60 && target.y - source.y > 175) {
    const route = source.x + 122 + Math.min(48, (target.y - source.y) / 116 * 6);
    return `M ${source.x + 90} ${source.y} H ${route - 8} Q ${route} ${source.y} ${route} ${source.y + 8} V ${target.y - 8} Q ${route} ${target.y} ${route - 8} ${target.y} H ${target.x + 90}`;
  }
  const start = source.y + sourceHalfHeight;
  const end = target.y - targetHalfHeight;
  const mid = (start + end) / 2;
  return `M ${source.x} ${start} C ${source.x} ${mid}, ${target.x} ${mid}, ${target.x} ${end}`;
}
