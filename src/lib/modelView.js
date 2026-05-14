import { modelProfiles, rawNodes, sampleModel, tensors } from "../data/modelData.js";

const OP_KIND = {
  Input: "input",
  Output: "output",
  Flatten: "layout",
  Reshape: "layout",
  Transpose: "layout",
  Gemm: "dense",
  MatMul: "dense",
  Conv: "convolution",
  MaxPool: "pooling",
  AveragePool: "pooling",
  Relu: "activation",
  Gelu: "activation",
  Sigmoid: "activation",
  Tanh: "activation",
  Softmax: "activation",
  Add: "compute",
  Mul: "compute",
  BatchNormalization: "normalization",
  LayerNormalization: "normalization"
};

export function createFixtureModelView() {
  return {
    model: sampleModel,
    rawNodes,
    tensors,
    modelProfiles,
    sourcePath: "",
    loadMessage: "Fixture model"
  };
}

export function createModelViewFromOnnx(fileName, parsed, sourcePath = "") {
  const graph = parsed.graph ?? {};
  const initializers = graph.initializers ?? [];
  const initializerNames = new Set(initializers.map((tensor) => tensor.name));
  const graphInputs = (graph.inputs ?? []).filter((input) => !initializerNames.has(input.name));
  const graphOutputs = graph.outputs ?? [];
  const ops = graph.nodes ?? [];
  const outputNames = new Set(graphOutputs.map((output) => output.name));
  const raw = [];

  graphInputs.forEach((input, index) => {
    raw.push({
      id: `input_${safeId(input.name || index)}`,
      name: input.name || `input_${index + 1}`,
      opType: "Input",
      groupHint: "input",
      groupId: "inputs",
      groupLabel: "Inputs",
      inputs: [],
      outputs: [input.name].filter(Boolean),
      metadata: {
        shape: formatShape(input.shape),
        dataType: input.dataType
      }
    });
  });

  ops.forEach((node, index) => {
    const kind = outputIntersects(node, outputNames) ? "output" : inferKind(node.opType);
    raw.push({
      id: `node_${index}_${safeId(node.name || node.opType || "op")}`,
      name: node.name || `${node.opType || "Op"} ${index + 1}`,
      opType: node.opType || "Unknown",
      groupHint: kind,
      groupId: `op_${index}`,
      groupLabel: formatGroupLabel(node, index, ops.length, kind),
      inputs: node.inputs ?? [],
      outputs: node.outputs ?? [],
      attributes: node.attributes ?? [],
      metadata: nodeMetadata(node, kind),
      recognizer: `onnx.${node.opType || "Unknown"}`
    });
  });

  graphOutputs.forEach((output, index) => {
    raw.push({
      id: `output_${safeId(output.name || index)}`,
      name: output.name || `output_${index + 1}`,
      opType: "Output",
      groupHint: "output",
      groupId: "outputs",
      groupLabel: "Outputs",
      inputs: [output.name].filter(Boolean),
      outputs: [],
      metadata: {
        shape: formatShape(output.shape),
        dataType: output.dataType
      }
    });
  });

  const laidOut = layoutRawNodes(raw);
  const opCounts = countBy(ops.map((node) => node.opType || "Unknown"));
  const primaryOpset = (parsed.opsets ?? []).find((opset) => !opset.domain) ?? parsed.opsets?.[0];
  const model = {
    fileName,
    family: inferFamily(opCounts),
    opset: primaryOpset?.version ?? "?",
    irVersion: parsed.irVersion ?? "?",
    layers: ops.length,
    graphName: graph.name || "unnamed graph",
    producer: [parsed.producerName, parsed.producerVersion].filter(Boolean).join(" ") || "unknown",
    parameterCount: estimateParameterCount(initializers),
    initializerCount: initializers.length,
    inputCount: graphInputs.length,
    outputCount: graphOutputs.length
  };

  return {
    model,
    rawNodes: laidOut,
    tensors: createTensorRows(graphInputs, graphOutputs, initializers),
    modelProfiles: inferProfiles(model.family, opCounts),
    sourcePath,
    loadMessage: `${ops.length} ONNX ops parsed`
  };
}

function inferKind(opType) {
  return OP_KIND[opType] ?? "compute";
}

function outputIntersects(node, outputNames) {
  return (node.outputs ?? []).some((output) => outputNames.has(output));
}

function nodeMetadata(node, kind) {
  const attrs = Object.fromEntries(
    (node.attributes ?? [])
      .filter((attribute) => attribute.name && attribute.value !== null)
      .slice(0, 6)
      .map((attribute) => [attribute.name, Array.isArray(attribute.value) ? attribute.value.join(", ") : attribute.value])
  );

  return {
    opType: node.opType || "Unknown",
    kind,
    inputs: (node.inputs ?? []).length,
    outputs: (node.outputs ?? []).length,
    ...attrs
  };
}

function formatGroupLabel(node, index, total, kind) {
  if (node.opType === "Gemm" && kind === "output") return "Output Dense";
  if (node.opType === "Gemm") return "Dense Layer";
  if (node.opType === "Relu") return "ReLU Activation";
  if (node.opType === "Flatten") return "Flatten";
  if (kind === "output") return "Output Layer";
  return node.opType || `Layer ${index + 1} of ${total}`;
}

function layoutRawNodes(nodes) {
  if (nodes.length <= 1) return nodes.map((node) => ({ ...node, x: 50, y: 50 }));
  const columns = nodes.length <= 4 ? nodes.length : Math.min(4, Math.ceil(Math.sqrt(nodes.length)));
  const rows = Math.ceil(nodes.length / columns);

  return nodes.map((node, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    return {
      ...node,
      x: 14 + (columns === 1 ? 36 : (column / (columns - 1)) * 72),
      y: rows === 1 ? 50 : 33 + (row / Math.max(rows - 1, 1)) * 38
    };
  });
}

function createTensorRows(inputs, outputs, initializers) {
  const rows = [];

  inputs.forEach((input) => {
    rows.push([input.name, input.dataType, formatShape(input.shape), "input", "dynamic"]);
  });

  outputs.forEach((output) => {
    rows.push([output.name, output.dataType, formatShape(output.shape), "output", "dynamic"]);
  });

  initializers.slice(0, 12).forEach((tensor) => {
    rows.push([tensor.name, tensor.dataType, formatShape(tensor.dims), "initializer", formatBytes(tensor.byteSize)]);
  });

  return rows;
}

function inferProfiles(family, opCounts) {
  const hasDense = (opCounts.Gemm ?? 0) + (opCounts.MatMul ?? 0) > 0;
  const hasConv = (opCounts.Conv ?? 0) > 0;

  if (family === "MLP classifier" || hasDense) {
    return [
      { id: "mlp", name: "MLP classifier", status: "active", confidence: family === "MLP classifier" ? 96 : 82 },
      { id: "generic", name: "ONNX feed-forward", status: "ready", confidence: 78 },
      { id: "unknown", name: "Unknown fallback", status: "generic", confidence: 48 }
    ];
  }

  if (hasConv) {
    return [
      { id: "cnn", name: "CNN", status: "active", confidence: 90 },
      { id: "generic", name: "ONNX vision", status: "ready", confidence: 75 },
      { id: "unknown", name: "Unknown fallback", status: "generic", confidence: 48 }
    ];
  }

  return [
    { id: "generic", name: "Generic ONNX graph", status: "active", confidence: 72 },
    { id: "unknown", name: "Unknown fallback", status: "generic", confidence: 50 }
  ];
}

function inferFamily(opCounts) {
  const dense = (opCounts.Gemm ?? 0) + (opCounts.MatMul ?? 0);
  const activations = (opCounts.Relu ?? 0) + (opCounts.Gelu ?? 0) + (opCounts.Sigmoid ?? 0) + (opCounts.Tanh ?? 0);
  if ((opCounts.Flatten ?? 0) > 0 && dense >= 2 && activations >= 1) return "MLP classifier";
  if ((opCounts.Conv ?? 0) > 0) return "Convolutional network";
  if ((opCounts.Attention ?? 0) > 0 || (opCounts.Softmax ?? 0) > 1) return "Attention model";
  return "Generic ONNX graph";
}

function estimateParameterCount(initializers) {
  const total = initializers.reduce((sum, tensor) => {
    if (!tensor.dims.length || tensor.dims.some((dim) => typeof dim !== "number")) return sum;
    return sum + tensor.dims.reduce((product, dim) => product * dim, 1);
  }, 0);

  return total ? total.toLocaleString() : "unknown";
}

function countBy(values) {
  return values.reduce((acc, value) => {
    acc[value] = (acc[value] ?? 0) + 1;
    return acc;
  }, {});
}

function formatShape(shape = []) {
  return shape.length ? shape.join(" x ") : "scalar";
}

function formatBytes(bytes) {
  if (!bytes) return "unknown";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function safeId(value) {
  return String(value).replace(/[^a-zA-Z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "item";
}
