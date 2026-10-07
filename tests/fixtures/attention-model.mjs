// Small, deterministic ONNX fixture with explicit weights and two residual blocks.
// Uses the field numbers from the public ONNX protobuf schema.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
export const vi = (value) => {
  let number = BigInt.asUintN(64, BigInt(value)); const bytes = [];
  do { const byte = Number(number & 127n); number >>= 7n; bytes.push(byte | (number ? 128 : 0)); } while (number);
  return Buffer.from(bytes);
};
export const field = (number, wire, bytes) => Buffer.concat([vi((number << 3) | wire), wire === 2 ? vi(bytes.length) : Buffer.alloc(0), bytes]);
export const int = (number, value) => field(number, 0, vi(value));
export const str = (number, value) => field(number, 2, Buffer.from(value));
export const msg = (number, parts) => field(number, 2, Buffer.concat(parts));
export function valueInfo(name, dims) {
  return Buffer.concat([str(1, name), msg(2, [msg(1, [int(1, 1), msg(2, dims.map((dim) => msg(1, [typeof dim === "number" ? int(1, dim) : str(2, dim)])))])])]);
}
export function node(opType, inputs, outputs, name = "", attrs = [], domain = "") {
  return Buffer.concat([...inputs.map((s) => str(1, s)), ...outputs.map((s) => str(2, s)), str(3, name), str(4, opType), ...attrs.map((a) => field(5, 2, a)), ...(domain ? [str(7, domain)] : [])]);
}
export function attribute(name, value) { return Buffer.concat([str(1, name), int(20, 2), int(3, value)]); }
export function makeAttentionModel() {
  const nodes = []; const params = []; const intermediates = [];
  const weight = (name, dims) => { params.push(Buffer.concat([...dims.map((dim) => int(1, dim)), int(2, 1), str(8, name), field(9, 2, Buffer.alloc(dims.reduce((a, b) => a * b, 1) * 4))])); return name; };
  let input = "hidden_states";
  for (let block = 0; block < 2; block += 1) {
    const prefix = `/transformer/h.${block}/`;
    const output = (name) => `${prefix}${name}`;
    const addNode = (op, ins, name, attrs = []) => { const out = output(name); nodes.push(node(op, ins, [out], `${prefix}${name}`, attrs)); intermediates.push(valueInfo(out, [1, 2, 4])); return out; };
    const norm = addNode("LayerNormalization", [input, weight(`${prefix}ln_scale`, [4]), weight(`${prefix}ln_bias`, [4])], "ln_1", [attribute("axis", -1)]);
    const q = addNode("MatMul", [norm, weight(`${prefix}q_weight`, [4, 4])], "attn/q_proj");
    const k = addNode("MatMul", [norm, weight(`${prefix}k_weight`, [4, 4])], "attn/k_proj");
    const v = addNode("MatMul", [norm, weight(`${prefix}v_weight`, [4, 4])], "attn/v_proj");
    const kt = addNode("Transpose", [k], "attn/key_transpose", [Buffer.concat([str(1, "perm"), int(20, 7), int(8, 0), int(8, 2), int(8, 1)])]);
    intermediates[intermediates.length - 1] = valueInfo(kt, [1, 4, 2]);
    const scores = addNode("MatMul", [q, kt], "attn/scores");
    intermediates[intermediates.length - 1] = valueInfo(scores, [1, 2, 2]);
    const probs = addNode("Softmax", [scores], "attn/softmax", [attribute("axis", -1)]);
    intermediates[intermediates.length - 1] = valueInfo(probs, [1, 2, 2]);
    const context = addNode("MatMul", [probs, v], "attn/value_mix");
    const projected = addNode("MatMul", [context, weight(`${prefix}o_weight`, [4, 4])], "attn/out_proj");
    const residual = addNode("Add", [input, projected], "residual_1");
    const norm2 = addNode("LayerNormalization", [residual, weight(`${prefix}ln2_scale`, [4]), weight(`${prefix}ln2_bias`, [4])], "ln_2", [attribute("axis", -1)]);
    const fc = addNode("MatMul", [norm2, weight(`${prefix}fc_weight`, [4, 8])], "mlp/fc");
    intermediates[intermediates.length - 1] = valueInfo(fc, [1, 2, 8]);
    const activation = addNode("Relu", [fc], "mlp/relu");
    intermediates[intermediates.length - 1] = valueInfo(activation, [1, 2, 8]);
    const down = addNode("MatMul", [activation, weight(`${prefix}down_weight`, [8, 4])], "mlp/down");
    input = addNode("Add", [residual, down], "residual_2");
  }
  const graph = Buffer.concat([...nodes.map((n) => field(1, 2, n)), str(2, "two_block_attention"), ...params.map((p) => field(5, 2, p)), field(11, 2, valueInfo("hidden_states", [1, 2, 4])), field(12, 2, valueInfo(input, [1, 2, 4])), ...intermediates.map((v) => field(13, 2, v))]);
  return Buffer.concat([int(1, 9), str(2, "ModelViz test fixture"), field(7, 2, graph), msg(8, [int(2, 17)])]);
}
if (process.argv[1] === fileURLToPath(import.meta.url)) writeFileSync(new URL("attention.onnx", import.meta.url), makeAttentionModel());
