import { createGraphIndex, deriveGroupEdges, flattenGroups, groupBoundary } from "./graph.js";
const kinds = {
  Input: "input",
  Output: "output",
  Constant: "support",
  Shape: "support",
  Cast: "layout",
  Gather: "embedding",
  Flatten: "layout",
  Reshape: "layout",
  Transpose: "layout",
  Gemm: "dense",
  MatMul: "compute",
  Conv: "convolution",
  Relu: "activation",
  Gelu: "activation",
  Softmax: "activation",
  LayerNormalization: "normalization",
  BatchNormalization: "normalization",
  Attention: "attention",
  MultiHeadAttention: "attention"
};
const layouts = new Set(["Reshape", "Transpose", "Split", "Squeeze", "Unsqueeze", "Identity", "Cast"]);
const activations = new Set(["Relu", "Gelu", "Sigmoid", "Tanh", "LeakyRelu"]);
const standard = node => !node.domain || node.domain === "ai.onnx";
export const classifyRawNode = node => kinds[node.opType] ?? "compute";
export function createArchitectureGroups(nodes, model = {}, graphIndex) {
  const index = graphIndex ?? createGraphIndex(nodes);
  const assigned = new Set();
  const groups = [];
  const make = (id, label, kind, raw, recognition, evidence, children = []) => {
    const ids = raw.map(node => node.id);
    return {
      id,
      label,
      kind,
      rawNodeIds: ids,
      rawNodeCount: ids.length,
      ...groupBoundary(ids, index),
      recognition,
      evidence,
      children,
      metadata: {
        operators: raw.length,
        ...Object.fromEntries(raw.length === 1 ? (raw[0].attributes ?? []).filter(a => a.value !== null && typeof a.value !== "object").map(a => [a.name, a.value]) : [])
      }
    };
  };
  const primitive = node => make(`raw:${node.id}`, node.opType === "Input" || node.opType === "Output" ? node.name : node.opType, classifyRawNode(node), [node], "exact", [`ONNX ${node.domain || "ai.onnx"}::${node.opType}`, node.name]);
  const add = (label, kind, raw, evidence, recognition = "pattern", roles = {}) => {
    const unique = [...new Map(raw.map(node => [node.id, node])).values()];
    if (!unique.length || unique.some(node => assigned.has(node.id))) return false;
    unique.forEach(node => assigned.add(node.id));
    const group = make(`group:${unique[0].id}`, label, kind, unique, recognition, evidence, unique.length > 1 ? unique.map(primitive) : []);
    group.roles = roles;
    for (const child of group.children) child.role = roles[child.rawNodeIds[0]];
    groups.push(group);
    return true;
  };
  const producer = name => index.byId.get(index.producer.get(name));
  const successors = node => [...new Set((node.outputs ?? []).flatMap(name => index.consumers.get(name) ?? []))].map(id => index.byId.get(id)).filter(Boolean);
  const isParameter = name => index.constantValues.has(name);
  const isProjection = node => standard(node) && ["MatMul", "Gemm"].includes(node.opType) && node.inputs.some(isParameter);
  const bias = node => standard(node) && node.opType === "Add" && node.inputs.some(isParameter);

  // Require connected QK -> softmax -> AV dataflow; a Softmax alone is not attention.
  for (const softmax of nodes.filter(node => standard(node) && node.opType === "Softmax")) {
    const scoreNodes = [];
    const visited = new Set();
    const findScore = (node, depth = 0) => {
      if (!node || visited.has(node.id) || depth > 8 || !standard(node)) return null;
      visited.add(node.id);
      if (node.opType === "MatMul" && !node.inputs.some(isParameter)) return [node];
      if (!["Add", "Mul", "Div", "Cast", "Reshape", "Transpose", "Where"].includes(node.opType)) return null;
      for (const input of node.inputs) {
        const path = findScore(producer(input), depth + 1);
        if (path) return [node, ...path];
      }
      return null;
    };
    const path = findScore(producer(softmax.inputs[0]));
    if (!path) continue;
    scoreNodes.push(...path);
    let probability = softmax;
    const after = [];
    for (let step = 0; step < 4; step += 1) {
      const children = successors(probability);
      if (children.length !== 1 || !standard(children[0]) || !layouts.has(children[0].opType)) break;
      probability = children[0];
      after.push(probability);
    }
    const av = successors(probability).find(node => standard(node) && node.opType === "MatMul" && probability.outputs.includes(node.inputs[0]) && !node.inputs.some(isParameter));
    if (!av) continue;
    const qk = path.at(-1);
    const raw = [softmax, ...scoreNodes, ...after, av];
    const roles = {
      [qk.id]: "Query × key scores",
      [softmax.id]: "Attention weights",
      [av.id]: "Weighted value mixing"
    };
    const seen = new Set(raw.map(node => node.id));
    const collectProjection = (name, role, depth = 0) => {
      const node = producer(name);
      if (!node || depth > 8 || !standard(node)) return;
      if (!layouts.has(node.opType) && !isProjection(node) && !bias(node)) return;
      const roleLabel = `${role} ${isProjection(node) ? "projection" : "layout / bias"}`;
      if (roles[node.id] && roles[node.id] !== roleLabel) roles[node.id] = "Shared query / key / value projection";else roles[node.id] = roleLabel;
      if (seen.has(node.id)) return;
      seen.add(node.id);
      raw.push(node);
      if (!isProjection(node)) for (const input of node.inputs) if (!isParameter(input)) collectProjection(input, role, depth + 1);
    };
    collectProjection(qk.inputs[0], "Query");
    collectProjection(qk.inputs[1], "Key");
    collectProjection(av.inputs[1], "Value");
    let tail = av;
    for (let step = 0; step < 6; step += 1) {
      const next = successors(tail);
      if (next.length !== 1) break;
      const node = next[0];
      if (!standard(node) || !layouts.has(node.opType) && !isProjection(node) && !bias(node)) break;
      raw.push(node);
      tail = node;
      roles[node.id] = isProjection(node) ? "Output projection" : "Output layout / bias";
      if (isProjection(node)) {
        const nextBias = successors(tail);
        if (nextBias.length === 1 && bias(nextBias[0])) {
          raw.push(nextBias[0]);
          roles[nextBias[0].id] = "Output bias";
        }
        break;
      }
    }
    add("Attention", "attention", raw, ["Connected score MatMul → Softmax → value MatMul", "Projection and layout operators traced through tensor connections", "Self vs cross attention is not established by this pattern"], "pattern", roles);
  }
  for (const node of nodes) {
    if (assigned.has(node.id)) continue;
    if (["Attention", "MultiHeadAttention"].includes(node.opType) && (standard(node) || node.domain === "com.microsoft")) {
      add("Attention (fused)", "attention", [node], [`${node.domain || "ai.onnx"}::${node.opType} is stored as one operator`, "Internal primitives are not present in this file"], "fused");
      continue;
    }
    if (isProjection(node) || standard(node) && node.opType === "Conv") {
      const raw = [node];
      let tail = node;
      let hasActivation = false;
      let projections = 1;
      for (let step = 0; step < 5; step += 1) {
        const next = successors(tail);
        if (next.length !== 1 || assigned.has(next[0].id) || !standard(next[0])) break;
        const candidate = next[0];
        if (bias(candidate) || activations.has(candidate.opType) || node.opType === "Conv" && candidate.opType === "BatchNormalization") {
          raw.push(candidate);
          tail = candidate;
          hasActivation ||= activations.has(candidate.opType);
        } else if (node.opType !== "Conv" && hasActivation && projections === 1 && isProjection(candidate)) {
          raw.push(candidate);
          tail = candidate;
          projections += 1;
        } else break;
      }
      const kind = node.opType === "Conv" ? "convolution" : projections > 1 ? "mlp" : "dense";
      add(kind === "mlp" ? "Feed-forward network" : kind === "convolution" ? "Convolution" : "Dense projection", kind, raw, ["Connected operators with constant/initializer parameters", `Pattern: ${raw.map(item => item.opType).join(" → ")}`]);
      continue;
    }
    if (standard(node) && node.opType === "Gather" && index.parameters.get(node.inputs[0])?.dims?.length === 2) {
      add("Embedding lookup", "embedding", [node], ["Gather reads a rank-2 initializer", "Embedding role inferred from lookup structure"]);
      continue;
    }
    assigned.add(node.id);
    const leaf = primitive(node);
    if (node.attributes?.some(attr => [5, 10].includes(attr.type))) {
      leaf.evidence.push("Contains nested graphs; this view maps the parent operator and its top-level tensors");
    }
    if (standard(node) && node.opType === "Add" && node.inputs.length === 2 && node.inputs.every(name => index.producer.has(name) && !isParameter(name))) {
      leaf.label = "Residual / tensor add";
      leaf.recognition = "pattern";
      leaf.evidence.push("Two computed tensors merge; residual role requires surrounding context");
    }
    groups.push(leaf);
  }

  // Exporter scopes add hierarchy without pretending that names prove architecture.
  const scopes = new Map();
  const root = [];
  const order = new Map(nodes.map((node, i) => [node.id, i]));
  groups.sort((a, b) => Math.min(...a.rawNodeIds.map(id => order.get(id))) - Math.min(...b.rawNodeIds.map(id => order.get(id))));
  for (const group of groups) {
    const hints = group.rawNodeIds.map(id => blockScope(index.byId.get(id))).filter(Boolean);
    const scope = hints[0];
    if (!scope || hints.length !== group.rawNodeIds.length || hints.some(hint => hint.key !== scope.key)) {
      root.push(group);
      continue;
    }
    if (!scopes.has(scope.key)) {
      const parent = make(`scope:${scope.key}`, `Block ${scope.number}`, "block", [], "scope", [`Exporter namespace: ${scope.key}`, "Children recognized independently from operator dataflow"], []);
      scopes.set(scope.key, parent);
      root.push(parent);
    }
    scopes.get(scope.key).children.push(group);
  }
  for (const parent of scopes.values()) {
    parent.rawNodeIds = parent.children.flatMap(child => child.rawNodeIds);
    parent.rawNodeCount = parent.rawNodeIds.length;
    Object.assign(parent, groupBoundary(parent.rawNodeIds, index));
    parent.metadata = {
      operators: parent.rawNodeCount,
      scope: parent.evidence[0].replace("Exporter namespace: ", "")
    };
  }
  return root;
}
function blockScope(node) {
  if (["Input", "Output"].includes(node.opType)) return null;
  const match = node.name?.match(/(?:^|[/.])(?:h|layers?|blocks?)[/.](\d+)(?=[/.]|$)/);
  if (!match) return null;
  return {
    key: node.name.slice(0, match.index + match[0].length),
    number: Number(match[1])
  };
}
export function createCleanEdges(groups, index) {
  return index ? deriveGroupEdges(groups, index) : [];
}
export function summarizeCoverage(groups, nodes) {
  const leaves = flattenGroups(groups).filter(group => !group.children?.length);
  const recognized = new Set(flattenGroups(groups).filter(group => ["pattern", "fused"].includes(group.recognition)).flatMap(group => group.rawNodeIds));
  const operators = nodes.filter(node => !["Input", "Output"].includes(node.opType));
  return {
    rawNodes: operators.length,
    semanticGroups: groups.length,
    recognizedNodes: operators.filter(node => recognized.has(node.id)).length,
    coverage: nodes.length ? Math.round(new Set(leaves.flatMap(g => g.rawNodeIds)).size / nodes.length * 100) : 0
  };
}
