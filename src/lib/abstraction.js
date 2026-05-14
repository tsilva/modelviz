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
  const transformerGroups = createTransformerGroups(nodes, model);
  if (transformerGroups) return transformerGroups;

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
  const byId = new Set(groups.map((group) => group.id));
  if (byId.has("transformer_blocks")) {
    return [
      ["inputs", "embeddings"],
      ["runtime_support", "embeddings"],
      ["embeddings", "transformer_blocks"],
      ["runtime_support", "transformer_blocks"],
      ["transformer_blocks", "final_norm"],
      ["final_norm", "lm_head"],
      ["lm_head", "outputs"]
    ].filter(([from, to]) => byId.has(from) && byId.has(to));
  }

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

function createTransformerGroups(nodes, model) {
  const blockIndexes = [...new Set(nodes.map(transformerBlockIndex).filter((index) => index !== null))].sort((a, b) => a - b);
  const hasTransformerPath = nodes.some((node) => nodeMatches(node, /(?:^|\/)transformer(?:\/|\.|$)/));
  const hasAttentionOps = nodes.filter((node) => node.opType === "Softmax" || nodeMatches(node, /\/attn\//)).length >= 2;

  if (model.family !== "Attention model" && !blockIndexes.length && !hasTransformerPath && !hasAttentionOps) {
    return null;
  }

  const buckets = {
    inputs: [],
    embeddings: [],
    transformer_blocks: [],
    final_norm: [],
    lm_head: [],
    outputs: [],
    runtime_support: []
  };
  const assigned = new Map();

  nodes.forEach((node) => {
    const bucket = primaryTransformerBucket(node);
    if (bucket) {
      buckets[bucket].push(node);
      assigned.set(node.id, bucket);
    }
  });

  assignSupportNodes(nodes, buckets, assigned);

  const blockRange = formatBlockRange(blockIndexes);
  const rawBlockNodes = buckets.transformer_blocks;
  const blockCount = blockIndexes.length || estimateBlockCount(rawBlockNodes);
  const attentionOps = rawBlockNodes.filter((node) => nodeMatches(node, /\/attn\//) || node.opType === "Softmax").length;
  const mlpOps = rawBlockNodes.filter((node) => nodeMatches(node, /\/mlp\//) || nodeMatches(node, /c_fc|c_proj|Gelu|Tanh/)).length;
  const cacheOutputs = nodes.filter((node) => node.opType === "Output" && /^present\.\d+\.(key|value)$/.test(node.name)).length;

  const groups = [
    makeSemanticGroup("inputs", "Inputs", "input", buckets.inputs, { tensors: inputTensorNames(buckets.inputs) }, 100, "io.inputs"),
    makeSemanticGroup("embeddings", "Token + Position Embeddings", "embedding", buckets.embeddings, { pattern: "token lookup + position lookup + attention mask", parameters: model.parameterCount }, 94, "transformer.embeddings"),
    makeSemanticGroup("transformer_blocks", `${blockCount || "Repeated"} Transformer Blocks`, "attention", rawBlockNodes, { blocks: blockRange || String(blockCount || "?"), pattern: "LayerNorm + self-attention + MLP + residual", attentionOps, mlpOps, cacheOutputs }, 94, "transformer.decoder.blocks"),
    makeSemanticGroup("final_norm", "Final LayerNorm", "norm", buckets.final_norm, { role: "normalize decoder hidden states" }, 90, "transformer.final_norm"),
    makeSemanticGroup("lm_head", "LM Head", "head", buckets.lm_head, { projection: "hidden states -> vocabulary logits", tiedEmbedding: hasTiedEmbeddingHead(nodes) }, 92, "transformer.lm_head"),
    makeSemanticGroup("outputs", "Logits + Cache Outputs", "output", buckets.outputs, { outputs: outputTensorSummary(buckets.outputs), cacheOutputs }, 100, "io.outputs")
  ];

  if (buckets.runtime_support.length) {
    groups.splice(2, 0, makeSemanticGroup("runtime_support", "Shape + Mask Support", "support", buckets.runtime_support, { role: "dynamic sequence shape, constants, casts, and causal mask helpers" }, 82, "onnx.runtime_support"));
  }

  const usefulGroups = groups.filter((group) => group.rawNodeCount > 0);
  return usefulGroups.length >= 3 ? usefulGroups : null;
}

function makeSemanticGroup(id, label, kind, raw, metadata, confidence, recognizer) {
  return {
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
  };
}

function primaryTransformerBucket(node) {
  if (node.opType === "Input") return "inputs";
  if (node.opType === "Output") return "outputs";
  if (nodeMatches(node, /\/lm_head\//) || nodeMatches(node, /logits|weight_transposed/)) return "lm_head";
  if (nodeMatches(node, /\/transformer\/ln_f\//)) return "final_norm";
  if (transformerBlockIndex(node) !== null) return "transformer_blocks";
  if (nodeMatches(node, /\/transformer\/(?:wte|wpe)\//) || nodeMatches(node, /(?:^|\/)transformer\/(?:Shape|Range|Add|Sub|Mul|Unsqueeze|Cast|Gather)(?:_|$|\/)/)) {
    return "embeddings";
  }
  return null;
}

function assignSupportNodes(nodes, buckets, assigned) {
  const consumerBuckets = new Map();
  nodes.forEach((node) => {
    const bucket = assigned.get(node.id);
    if (!bucket) return;
    (node.inputs ?? []).forEach((input) => {
      if (!consumerBuckets.has(input)) consumerBuckets.set(input, bucket);
    });
  });

  nodes.forEach((node) => {
    if (assigned.has(node.id)) return;
    const outputConsumers = (node.outputs ?? []).map((output) => consumerBuckets.get(output)).filter(Boolean);
    const bucket = outputConsumers.find((candidate) => candidate !== "outputs") ?? "runtime_support";
    buckets[bucket].push(node);
    assigned.set(node.id, bucket);
  });
}

function transformerBlockIndex(node) {
  const match = searchableNodeText(node).match(/(?:^|[/.])h\.(\d+)(?:[/.]|$)/);
  return match ? Number.parseInt(match[1], 10) : null;
}

function nodeMatches(node, pattern) {
  return pattern.test(searchableNodeText(node));
}

function searchableNodeText(node) {
  return [node.name, node.opType, ...(node.inputs ?? []), ...(node.outputs ?? [])].filter(Boolean).join(" ");
}

function formatBlockRange(indexes) {
  if (!indexes.length) return "";
  if (indexes.length === 1) return `h.${indexes[0]}`;
  return `h.${indexes[0]}-h.${indexes.at(-1)}`;
}

function estimateBlockCount(nodes) {
  const layerNorms = nodes.filter((node) => nodeMatches(node, /\/ln_[12]\//)).length;
  return layerNorms ? Math.max(1, Math.round(layerNorms / 10)) : null;
}

function inputTensorNames(nodes) {
  const names = nodes.map((node) => node.name).filter(Boolean);
  return names.length ? names.join(", ") : "model inputs";
}

function outputTensorSummary(nodes) {
  const names = nodes.map((node) => node.name).filter(Boolean);
  const visible = names.filter((name) => !/^present\.\d+\.(key|value)$/.test(name));
  const cacheCount = names.length - visible.length;
  return [...visible, cacheCount ? `${cacheCount} key/value cache tensors` : null].filter(Boolean).join(", ") || "model outputs";
}

function hasTiedEmbeddingHead(nodes) {
  return nodes.some((node) => nodeMatches(node, /wte\.weight_transposed|transformer\.wte\.weight/));
}
