import assert from "node:assert/strict";
import test from "node:test";
import { createArchitectureGroups, createCleanEdges, summarizeCoverage } from "../src/lib/abstraction.js";
import { createGraphIndex, deriveGroupEdges, flattenGroups, groupBoundary, graphEdgePath, layoutGraph, ownersByNode, visibleGroups } from "../src/lib/graph.js";
import { createModelViewFromOnnx } from "../src/lib/modelView.js";
import { parseOnnxModel } from "../src/lib/onnxParser.js";
import { makeAttentionModel, node, field, int, msg, str, valueInfo } from "../tests/fixtures/attention-model.mjs";
const raw = (id, opType, inputs, outputs, extra = {}) => ({ id, opType, inputs, outputs, name: id, ...extra });
const graph = (nodes, initializers = []) => createGraphIndex(nodes, [], initializers);
const fixture = () => createModelViewFromOnnx("attention.onnx", parseOnnxModel(makeAttentionModel()));

test("parser preserves negative attributes, repeated integers, domains and intermediate shapes", () => {
  const view = fixture();
  assert.equal(view.rawNodes.find((n) => n.opType === "LayerNormalization").attributes[0].value, -1);
  assert.deepEqual(view.rawNodes.find((n) => n.opType === "Transpose").attributes[0].value, [0, 2, 1]);
  assert.equal(view.tensors.find(([name]) => name.endsWith("attn/scores"))[2], "1 x 2 x 2");
  assert.equal(view.initializers.length, 20);
  assert.ok(view.tensors.filter((t) => t[3] === "initializer").length > 12);
});

test("two residual blocks have real attention and feed-forward groups with expansion and complete source membership", () => {
  const view = fixture();
  const index = createGraphIndex(view.rawNodes, view.tensors, view.initializers);
  const roots = createArchitectureGroups(view.rawNodes, view.model, index);
  const all = flattenGroups(roots);
  const attention = all.filter((g) => g.label === "Attention");
  assert.equal(attention.length, 2);
  assert.equal(attention[0].rawNodeCount, 8);
  assert.equal(all.filter((g) => g.label === "Feed-forward network").length, 2);
  assert.equal(roots.filter((g) => g.kind === "block").length, 2);
  const leaves = all.filter((g) => !g.children.length);
  assert.deepEqual(new Set(leaves.flatMap((g) => g.rawNodeIds)), new Set(view.rawNodes.map((n) => n.id)));
  assert.equal(leaves.flatMap((g) => g.rawNodeIds).length, view.rawNodes.length);
  assert.equal(summarizeCoverage(roots, view.rawNodes).coverage, 100);
});

test("attention is recognized through tensor flow without exporter names", () => {
  const view = fixture();
  const nodes = view.rawNodes.map((n, i) => ({ ...n, name: `unnamed_${i}` }));
  const index = createGraphIndex(nodes, view.tensors, view.initializers);
  const groups = createArchitectureGroups(nodes, view.model, index);
  assert.equal(groups.filter((g) => g.label === "Attention").length, 2);
  assert.equal(groups.filter((g) => g.kind === "block").length, 0);
});

test("standalone classifier Softmax and custom-domain lookalikes remain exact primitives", () => {
  const nodes = [raw("in", "Input", [], ["x"]), raw("scores", "Softmax", ["x"], ["y"]), raw("out", "Output", ["y"], []), raw("custom", "MatMul", ["x", "w"], ["z"], { domain: "vendor.custom" })];
  const groups = createArchitectureGroups(nodes, {}, graph(nodes, [{ name: "w", dims: [4,4] }]));
  assert.equal(groups.filter((g) => g.kind === "attention").length, 0);
  assert.equal(groups.find((g) => g.rawNodeIds.includes("custom")).recognition, "exact");
});

test("fused attention stays a single operator with no invented expansion", () => {
  const nodes = [raw("fused", "Attention", ["x", "w"], ["y"], { domain: "com.microsoft" })];
  const [group] = createArchitectureGroups(nodes, {}, graph(nodes));
  assert.equal(group.recognition, "fused"); assert.equal(group.children.length, 0);
  assert.match(group.evidence.join(" "), /not present/);
});

test("real boundary tensors exclude internal intermediates and preserve shared weights", () => {
  const nodes = [raw("a", "MatMul", ["x", "w", ""], ["h"]), raw("b", "Relu", ["h"], ["y"]), raw("c", "MatMul", ["y", "w"], ["z"])];
  const index = graph(nodes, [{ name: "w", dims: [4,4] }]);
  assert.deepEqual(groupBoundary(["a", "b"], index), { inputs: ["x"], outputs: ["y"], parameters: ["w"] });
  assert.equal(index.edges.length, 2);
});

test("group connections preserve branching and residual bypasses, never fabricate list-order edges", () => {
  const nodes = [raw("a", "Input", [], ["x"]), raw("b", "Relu", ["x"], ["y"]), raw("c", "Add", ["x", "y"], ["z"]), raw("unrelated", "Constant", [], ["unused"])];
  const index = graph(nodes);
  const groups = nodes.map((n) => ({ id: `g_${n.id}`, rawNodeIds: [n.id] }));
  const edges = deriveGroupEdges(groups, index);
  assert.deepEqual(edges.map((e) => [e.from, e.to]), [["g_a", "g_b"], ["g_a", "g_c"], ["g_b", "g_c"]]);
  assert.equal(edges.some((e) => e.to === "g_unrelated"), false);
});

test("parallel architecture groups retain a fork and join when display order changes", () => {
  const nodes = [raw("input", "Input", [], ["x"]), raw("left", "MatMul", ["x", "w1"], ["l"]), raw("right", "MatMul", ["x", "w2"], ["r"]), raw("join", "Add", ["l", "r"], ["y"])];
  const index = graph(nodes, [{ name: "w1", dims: [4, 4] }, { name: "w2", dims: [4, 4] }]);
  const groups = createArchitectureGroups(nodes, {}, index);
  const owners = ownersByNode(groups);
  const connections = createCleanEdges([...groups].reverse(), index).map(({ from, to }) => [from, to]);
  const owner = (id) => owners.get(id)[0];
  assert.deepEqual(connections, [[owner("input"), owner("left")], [owner("input"), owner("right")], [owner("left"), owner("join")], [owner("right"), owner("join")]]);
});

test("expansion changes the quotient graph and supports reverse operator ownership", () => {
  const view = fixture(); const index = createGraphIndex(view.rawNodes, view.tensors, view.initializers);
  const roots = createArchitectureGroups(view.rawNodes, view.model, index);
  const parent = roots.find((g) => g.kind === "block");
  const groups = visibleGroups(roots, new Set([parent.id]));
  assert.equal(groups.some((g) => g.id === parent.id), false);
  const attn = groups.find((g) => g.label === "Attention");
  assert.deepEqual(ownersByNode(groups).get(attn.rawNodeIds[0]), [attn.id]);
  const edges = deriveGroupEdges(groups, index);
  const residual = groups.find((g) => g.label === "Residual / tensor add");
  assert.ok(edges.some((e) => e.from === attn.id && e.to === residual.id));
});

test("dense grouping stops at shared output branches", () => {
  const nodes = [raw("a", "MatMul", ["x", "w"], ["h"]), raw("b", "Relu", ["h"], ["y"]), raw("c", "Sigmoid", ["h"], ["z"])];
  const groups = createArchitectureGroups(nodes, {}, graph(nodes, [{ name: "w", dims: [4,4] }]));
  assert.equal(groups.find((g) => g.kind === "dense").rawNodeCount, 1);
});

test("unnamed and duplicate node names retain unique raw IDs", () => {
  const n = node("Relu", ["x"], ["y"], "duplicate");
  const n2 = node("Relu", ["y"], ["z"], "duplicate");
  const bytes = Buffer.concat([int(1, 9), msg(7, [field(1, 2, n), field(1, 2, n2), field(11, 2, valueInfo("x", [4])), field(12, 2, valueInfo("z", [4]))])]);
  const view = createModelViewFromOnnx("duplicates.onnx", parseOnnxModel(bytes));
  assert.equal(new Set(view.rawNodes.map((n) => n.id)).size, view.rawNodes.length);
});

test("DAG layout follows dependencies even when source arrays are reordered", () => {
  const nodes = [raw("c", "Add", [], []), raw("a", "Input", [], []), raw("b", "Relu", [], [])];
  const layout = layoutGraph(nodes, [{ from: "a", to: "b" }, { from: "b", to: "c" }]);
  assert.ok(layout.points.a.y < layout.points.b.y && layout.points.b.y < layout.points.c.y);
  const wide = layoutGraph(Array.from({ length: 20 }, (_, i) => ({ id: String(i) })), []);
  assert.ok(Object.values(wide.points).every((p) => p.x >= 90 && p.x <= wide.width - 90));
});

test("truncated model fields reject rather than silently build a partial model", () => {
  assert.throws(() => parseOnnxModel(Buffer.from([58, 100, 1])), /Truncated/);
});

test("dense weights remain recognizable after constant reshapes and transposes", () => {
  const nodes = [raw("shape", "Constant", [], ["dims"]), raw("reshape", "Reshape", ["w", "dims"], ["wr"]), raw("dense", "MatMul", ["x", "wr"], ["h"]), raw("bias", "Add", ["h", "b"], ["y"])];
  const index = graph(nodes, [{ name: "w", dims: [4,4] }, { name: "b", dims: [4] }]);
  assert.ok(index.constantValues.has("wr"));
  const dense = createArchitectureGroups(nodes, {}, index).find((g) => g.kind === "dense");
  assert.deepEqual(dense.rawNodeIds, ["dense", "bias"]);
  assert.deepEqual(dense.inputs, ["x"]);
  assert.deepEqual(dense.parameters, ["wr", "b"]);
});

test("graph inputs and outputs remain separate from named transformer scopes", () => {
  const view = fixture(); const index = createGraphIndex(view.rawNodes, view.tensors, view.initializers);
  const roots = createArchitectureGroups(view.rawNodes, view.model, index);
  assert.equal(roots.find((g) => g.kind === "output").rawNodeCount, 1);
  assert.ok(roots.filter((g) => g.kind === "block").every((g) => g.rawNodeCount === 15));
});

test("unknown rank stays unknown and does not masquerade as a scalar", () => {
  const unknownShape = Buffer.concat([str(1, "x"), msg(2, [msg(1, [int(1, 1)])])]);
  const bytes = Buffer.concat([int(1, 9), msg(7, [field(11, 2, unknownShape), field(12, 2, valueInfo("scalar", []))])]);
  const view = createModelViewFromOnnx("shapes.onnx", parseOnnxModel(bytes));
  assert.equal(view.tensors.find(([name]) => name === "x")[2], "not recorded");
  assert.equal(view.tensors.find(([name]) => name === "scalar")[2], "scalar");
});

test("convolution normalization and activation form a connected group while pooling stays separate", () => {
  const nodes = [raw("conv", "Conv", ["x", "w"], ["c"]), raw("bn", "BatchNormalization", ["c", "scale", "bias", "mean", "var"], ["n"]), raw("relu", "Relu", ["n"], ["r"]), raw("pool", "MaxPool", ["r"], ["y"])];
  const params = ["w", "scale", "bias", "mean", "var"].map((name) => ({ name, dims: [4] }));
  const groups = createArchitectureGroups(nodes, {}, graph(nodes, params));
  assert.deepEqual(groups.find((g) => g.kind === "convolution").rawNodeIds, ["conv", "bn", "relu"]);
  assert.equal(groups.find((g) => g.rawNodeIds.includes("pool")).recognition, "exact");
});

test("embedding inference requires an actual rank-2 initializer", () => {
  const nodes = [raw("embedding", "Gather", ["w", "ids"], ["vectors"]), raw("shape_gather", "Gather", ["shape", "position"], ["dim"])];
  const groups = createArchitectureGroups(nodes, {}, graph(nodes, [{ name: "w", dims: [100, 4] }, { name: "shape", dims: [3] }]));
  assert.equal(groups.find((g) => g.rawNodeIds.includes("embedding")).label, "Embedding lookup");
  assert.equal(groups.find((g) => g.rawNodeIds.includes("shape_gather")).recognition, "exact");
});

test("empty files reject with a clear missing-graph error", () => {
  assert.throws(() => parseOnnxModel(new Uint8Array()), /does not contain an ONNX graph/);
});


test("residual paths route beside intermediate nodes and expanded nodes get exterior ports", () => {
  const skip = graphEdgePath({ x: 240, y: 100 }, { x: 240, y: 448 });
  assert.match(skip, /^M 330 100 H/);
  assert.match(skip, /H 330$/);
  assert.match(graphEdgePath({ x: 240, y: 100 }, { x: 240, y: 216 }, 51, 51), /^M 240 151 C .*240 165$/);
});
