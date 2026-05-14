const opWeights = {
  Input: "input",
  Output: "output",
  Constant: "utility",
  Cast: "utility",
  Shape: "utility",
  Gather: "data",
  Flatten: "layout",
  Reshape: "layout",
  Transpose: "layout",
  Gemm: "dense",
  MatMul: "compute",
  Add: "compute",
  Mul: "compute",
  Relu: "activation",
  Softmax: "attention",
  LayerNormalization: "normalization",
  BatchNormalization: "normalization",
  Gelu: "activation"
};

export function classifyRawNode(node) {
  return opWeights[node.opType] ?? "other";
}

export function createArchitectureGroups(nodes, model) {
  const byHint = nodes.reduce((acc, node) => {
    const key = node.groupId ?? node.groupHint ?? "unknown";
    acc[key] = acc[key] ? [...acc[key], node] : [node];
    return acc;
  }, {});

  const orderedKeys = Object.keys(byHint);

  const makeGroup = (id, label, kind, raw, metadata, confidence, recognizer) => ({
    id,
    label,
    kind,
    rawNodeIds: raw.map((node) => node.id),
    rawNodeCount: raw.length,
    inputs: [...new Set(raw.flatMap((node) => node.inputs ?? []))].slice(0, 4),
    outputs: [...new Set(raw.flatMap((node) => node.outputs ?? []))].slice(-4),
    metadata,
    confidence,
    recognizer
  });

  if (model.family !== "GPT-2") {
    return orderedKeys.map((key, index) => {
      const raw = byHint[key];
      const first = raw[0] ?? {};
      const kind = first.groupHint ?? first.opType?.toLowerCase() ?? "unknown";
      const label = first.groupLabel ?? labelForKind(kind, index);
      const metadata = first.metadata ?? { opType: first.opType ?? "unknown" };
      const confidence = confidenceForKind(kind);
      const recognizer = first.recognizer ?? `onnx.${first.opType ?? kind}`;
      return makeGroup(key, label, kind, raw, metadata, confidence, recognizer);
    });
  }

  const attentionRaw = byHint.attention ?? [];
  const mlpRaw = [...(byHint.mlp ?? []), ...(byHint.residual ?? []).filter((node) => node.id === "resid2")];

  return [
    makeGroup("inputs", "Inputs", "input", byHint.input ?? [], { tensors: "input_ids, position_ids, attention_mask" }, 100, "io"),
    makeGroup("embeddings", "Token + Position Embeddings", "embedding", byHint.embedding ?? [], { vocab: model.vocab, hiddenSize: model.hiddenSize }, 96, "gpt.embedding"),
    makeGroup("ln1", "LayerNorm h.0/ln_1", "norm", byHint.norm ?? [], { normalizedShape: model.hiddenSize }, 93, "onnx.LayerNormalization"),
    makeGroup("attn0", "h.0 Self-Attention", "attention", attentionRaw, { hiddenSize: model.hiddenSize, heads: model.heads, headDim: model.hiddenSize / model.heads, qkv: "768 -> 2304", causalMask: true }, 94, "gpt.attention.qkv"),
    makeGroup("mlp0", "h.0 MLP", "mlp", mlpRaw, { expansion: "4x", activation: "GELU", projection: "768 -> 3072 -> 768" }, 91, "gpt.mlp.gelu"),
    makeGroup("repeat", "Transformer Blocks h.1-h.10", "repeat", byHint.repeat ?? [], { repeatedBlocks: model.layers - 2, pattern: "LayerNorm + Attention + MLP" }, 87, "namespace.repetition"),
    makeGroup("head", "Final Norm + LM Head", "head", byHint.head ?? [], { tiedEmbedding: true, logits: `batch x sequence x ${model.vocab}` }, 95, "gpt.lm_head")
  ];
}

export function createCleanEdges(groups) {
  if (groups?.length) {
    return groups.slice(0, -1).map((group, index) => [group.id, groups[index + 1].id]);
  }

  return [];
}

export function summarizeCoverage(groups, nodes) {
  const covered = new Set(groups.flatMap((group) => group.rawNodeIds));
  const computeNodes = nodes.filter((node) => classifyRawNode(node) !== "utility");
  const coveredCompute = computeNodes.filter((node) => covered.has(node.id));
  const denominator = computeNodes.length || 1;
  const groupCount = groups.length || 1;
  return {
    rawNodes: nodes.length,
    semanticGroups: groups.length,
    coverage: Math.round((coveredCompute.length / denominator) * 100),
    averageConfidence: Math.round(groups.reduce((sum, group) => sum + group.confidence, 0) / groupCount)
  };
}

function labelForKind(kind, index) {
  const labels = {
    input: "Inputs",
    output: "Outputs",
    dense: "Dense Layer",
    activation: "Activation",
    layout: "Layout",
    compute: "Compute"
  };

  return labels[kind] ?? `Group ${index + 1}`;
}

function confidenceForKind(kind) {
  const confidence = {
    input: 100,
    output: 100,
    dense: 92,
    activation: 94,
    layout: 90,
    normalization: 90,
    convolution: 91,
    pooling: 88,
    compute: 78
  };

  return confidence[kind] ?? 70;
}
