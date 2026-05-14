export const modelProfiles = [
  { id: "gpt2", name: "GPT-2", status: "active", confidence: 94 },
  { id: "bert", name: "BERT", status: "ready", confidence: 88 },
  { id: "vit", name: "ViT", status: "ready", confidence: 82 },
  { id: "resnet", name: "ResNet", status: "ready", confidence: 79 },
  { id: "unknown", name: "Unknown fallback", status: "generic", confidence: 52 }
];

export const pipelineSteps = [
  { name: "Parse", detail: "ONNX nodes, tensors, attrs" },
  { name: "Normalize", detail: "producer/consumer graph" },
  { name: "Collapse", detail: "shape ops and constants" },
  { name: "Recognize", detail: "attention, MLP, norm" },
  { name: "Group", detail: "semantic raw-node sets" },
  { name: "Layout", detail: "clean architecture graph" }
];

export const sampleModel = {
  fileName: "gpt2-small.onnx",
  family: "GPT-2",
  opset: 17,
  irVersion: 8,
  hiddenSize: 768,
  layers: 12,
  heads: 12,
  vocab: 50257
};

export const rawNodes = [
  { id: "input_ids", name: "input_ids", opType: "Input", groupHint: "input", x: 6, y: 34, outputs: ["ids"] },
  { id: "position_ids", name: "position_ids", opType: "Input", groupHint: "input", x: 6, y: 62, outputs: ["pos"] },
  { id: "wte_gather", name: "/transformer/wte/Gather", opType: "Gather", groupHint: "embedding", x: 20, y: 29, inputs: ["ids", "wte.weight"], outputs: ["tok_emb"] },
  { id: "wpe_gather", name: "/transformer/wpe/Gather", opType: "Gather", groupHint: "embedding", x: 20, y: 63, inputs: ["pos", "wpe.weight"], outputs: ["pos_emb"] },
  { id: "emb_add", name: "/transformer/Add", opType: "Add", groupHint: "embedding", x: 33, y: 46, inputs: ["tok_emb", "pos_emb"], outputs: ["hidden_0"] },
  { id: "ln1", name: "/transformer/h.0/ln_1/LayerNormalization", opType: "LayerNormalization", groupHint: "norm", x: 46, y: 29, inputs: ["hidden_0"], outputs: ["ln1_out"] },
  { id: "qkv_matmul", name: "/transformer/h.0/attn/c_attn/MatMul", opType: "MatMul", groupHint: "attention", x: 58, y: 22, inputs: ["ln1_out", "c_attn.weight"], outputs: ["qkv_mm"] },
  { id: "qkv_add", name: "/transformer/h.0/attn/c_attn/Add", opType: "Add", groupHint: "attention", x: 58, y: 34, inputs: ["qkv_mm", "c_attn.bias"], outputs: ["qkv"] },
  { id: "qkv_split", name: "/transformer/h.0/attn/Split", opType: "Split", groupHint: "attention", x: 58, y: 46, inputs: ["qkv"], outputs: ["q", "k", "v"] },
  { id: "attn_reshape", name: "/transformer/h.0/attn/Reshape", opType: "Reshape", groupHint: "attention", x: 70, y: 22, inputs: ["q", "k", "v"], outputs: ["heads"] },
  { id: "attn_transpose", name: "/transformer/h.0/attn/Transpose", opType: "Transpose", groupHint: "attention", x: 70, y: 34, inputs: ["heads"], outputs: ["heads_t"] },
  { id: "attn_scores", name: "/transformer/h.0/attn/MatMul_QK", opType: "MatMul", groupHint: "attention", x: 70, y: 46, inputs: ["q", "k"], outputs: ["scores"] },
  { id: "causal_mask", name: "/transformer/h.0/attn/Where_causal_mask", opType: "Where", groupHint: "attention", x: 70, y: 58, inputs: ["scores", "mask"], outputs: ["masked_scores"] },
  { id: "softmax", name: "/transformer/h.0/attn/Softmax", opType: "Softmax", groupHint: "attention", x: 82, y: 34, inputs: ["masked_scores"], outputs: ["probs"] },
  { id: "attn_value", name: "/transformer/h.0/attn/MatMul_V", opType: "MatMul", groupHint: "attention", x: 82, y: 46, inputs: ["probs", "v"], outputs: ["context"] },
  { id: "attn_proj", name: "/transformer/h.0/attn/c_proj/MatMul", opType: "MatMul", groupHint: "attention", x: 82, y: 58, inputs: ["context", "c_proj.weight"], outputs: ["attn_out"] },
  { id: "resid1", name: "/transformer/h.0/Add_residual_attn", opType: "Add", groupHint: "residual", x: 92, y: 42, inputs: ["hidden_0", "attn_out"], outputs: ["hidden_1"] },
  { id: "ln2", name: "/transformer/h.0/ln_2/LayerNormalization", opType: "LayerNormalization", groupHint: "mlp", x: 46, y: 72, inputs: ["hidden_1"], outputs: ["ln2_out"] },
  { id: "mlp_fc", name: "/transformer/h.0/mlp/c_fc/MatMul", opType: "MatMul", groupHint: "mlp", x: 58, y: 72, inputs: ["ln2_out", "c_fc.weight"], outputs: ["fc"] },
  { id: "gelu", name: "/transformer/h.0/mlp/Gelu", opType: "Gelu", groupHint: "mlp", x: 70, y: 72, inputs: ["fc"], outputs: ["gelu"] },
  { id: "mlp_proj", name: "/transformer/h.0/mlp/c_proj/MatMul", opType: "MatMul", groupHint: "mlp", x: 82, y: 72, inputs: ["gelu", "c_proj.weight"], outputs: ["mlp_out"] },
  { id: "resid2", name: "/transformer/h.0/Add_residual_mlp", opType: "Add", groupHint: "residual", x: 92, y: 72, inputs: ["hidden_1", "mlp_out"], outputs: ["hidden_2"] },
  { id: "blocks", name: "/transformer/h.1...h.10/RepeatedBlocks", opType: "Subgraph", groupHint: "repeat", x: 52, y: 88, inputs: ["hidden_2"], outputs: ["hidden_11"] },
  { id: "ln_f", name: "/transformer/ln_f/LayerNormalization", opType: "LayerNormalization", groupHint: "head", x: 70, y: 88, inputs: ["hidden_11"], outputs: ["final_hidden"] },
  { id: "lm_head", name: "/lm_head/MatMul", opType: "MatMul", groupHint: "head", x: 86, y: 88, inputs: ["final_hidden", "wte.weight"], outputs: ["logits"] }
];

export const tensors = [
  ["input_ids", "int64", "batch x sequence", "input", "dynamic"],
  ["attention_mask", "int64", "batch x sequence", "input", "dynamic"],
  ["wte.weight", "float32", "50257 x 768", "initializer", "147.2 MB"],
  ["wpe.weight", "float32", "1024 x 768", "initializer", "3.0 MB"],
  ["h.0.attn.c_attn.weight", "float32", "768 x 2304", "initializer", "6.8 MB"],
  ["present.0.key", "float32", "batch x 12 x seq x 64", "cache output", "dynamic"],
  ["present.0.value", "float32", "batch x 12 x seq x 64", "cache output", "dynamic"],
  ["logits", "float32", "batch x sequence x 50257", "output", "dynamic"]
];
