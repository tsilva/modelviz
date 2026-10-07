import assert from "node:assert/strict";
import test from "node:test";
import { parseOnnxModel } from "../src/lib/onnxParser.js";
import { field, int, msg, node, str, vi } from "../tests/fixtures/attention-model.mjs";

const model = (...graphFields) => msg(7, graphFields);
const floats = (...values) => {
  const bytes = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
  return bytes;
};
const doubles = (...values) => {
  const bytes = Buffer.alloc(values.length * 8);
  values.forEach((value, index) => bytes.writeDoubleLE(value, index * 8));
  return bytes;
};
const attribute = (name, type, ...values) => Buffer.concat([str(1, name), int(20, type), ...values]);
const attributes = (...values) => parseOnnxModel(model(field(1, 2, node("Custom", [], [], "", values)))).graph.nodes[0].attributes;
const initializer = (name, type, ...values) => msg(5, [int(1, 2), int(2, type), str(8, name), ...values]);

test("ONNX scalar and list attributes use the schema field numbers", () => {
  const parsed = attributes(
    attribute("axis", 2, int(3, 1)),
    attribute("epsilon", 1, field(2, 5, floats(0.125))),
    attribute("auto_pad", 3, str(4, "SAME_UPPER")),
    attribute("scales", 6, field(7, 2, floats(1.5, 2.5))),
    attribute("perm", 7, field(8, 2, Buffer.concat([vi(0), vi(2), vi(1)])))
  );
  assert.deepEqual(parsed.map(({ type, value }) => [type, value]), [
    [2, 1], [1, 0.125], [3, "SAME_UPPER"], [6, [1.5, 2.5]], [7, [0, 2, 1]]
  ]);
});

test("repeated attributes concatenate mixed packed and unpacked segments", () => {
  const parsed = attributes(
    attribute("floats", 6, field(7, 2, floats(1.5)), field(7, 5, floats(2.5)), field(7, 2, floats(3.5))),
    attribute("ints", 7, field(8, 2, Buffer.concat([vi(0), vi(-1)])), int(8, 2), field(8, 2, vi(3)))
  );
  assert.deepEqual(parsed.map(({ value }) => value), [[1.5, 2.5, 3.5], [0, -1, 2, 3]]);
});

test("a graph claiming 127 missing bytes is rejected", () => {
  assert.throws(() => parseOnnxModel(Buffer.from([58, 127])), /Truncated/);
});

test("unknown fields cannot skip beyond either the file or their parent message", () => {
  const truncatedFields = [
    Buffer.concat([vi(100 * 8 + 2), vi(127)]),
    field(100, 1, Buffer.alloc(7)),
    field(100, 5, Buffer.alloc(3))
  ];
  for (const bytes of truncatedFields) {
    assert.throws(() => parseOnnxModel(Buffer.concat([model(), bytes])), /Truncated/);
    // The following outer field must not satisfy a read past the nested graph.
    assert.throws(() => parseOnnxModel(Buffer.concat([model(bytes), str(2, "producer")])), /Truncated/);
  }
});

test("known fixed-width attributes respect their enclosing message boundary", () => {
  assert.throws(() => attributes(attribute("epsilon", 1, field(2, 5, Buffer.alloc(3)))), /Truncated/);
});

test("packed fields reject incomplete fixed-width elements and varints", () => {
  assert.throws(() => attributes(attribute("scales", 6, field(7, 2, Buffer.alloc(5)))), /Truncated/);
  assert.throws(() => attributes(attribute("axes", 7, field(8, 2, Buffer.from([128])))), /Unexpected end/);
  assert.throws(() => parseOnnxModel(model(initializer("w", 11, field(10, 2, Buffer.alloc(9))))), /Truncated/);
  assert.throws(() => parseOnnxModel(model(initializer("w", 6, field(5, 2, Buffer.from([128]))))), /Unexpected end/);
});

test("invalid tags, overflowing integers and unsafe lengths are rejected", () => {
  const malformed = [
    Buffer.from([0]),
    Buffer.concat([vi(100 * 8), Buffer.alloc(10, 128), Buffer.from([0])]),
    Buffer.concat([vi(100 * 8), Buffer.alloc(9, 255), Buffer.from([2])]),
    Buffer.concat([vi(100 * 8 + 2), vi(2n ** 53n)]),
    Buffer.concat([vi(2n ** 32n), Buffer.from([0])])
  ];
  for (const bytes of malformed) assert.throws(() => parseOnnxModel(Buffer.concat([model(), bytes])), /Invalid|Truncated/);
  assert.throws(() => attributes(attribute("axis", 2, field(3, 0, Buffer.alloc(10, 255)))), /Invalid/);
});

test("unknown protobuf groups require matching end tags", () => {
  const start = vi(100 * 8 + 3);
  const end = vi(100 * 8 + 4);
  assert.throws(() => parseOnnxModel(Buffer.concat([model(), start, int(1, 1)])), /Truncated/);
  assert.throws(() => parseOnnxModel(Buffer.concat([model(), start, vi(101 * 8 + 4)])), /Invalid/);
  assert.throws(() => parseOnnxModel(Buffer.concat([model(), end])), /Unexpected/);
});

test("valid unknown fields and uint64 values remain skippable", () => {
  const unknown = Buffer.concat([
    int(100, 2n ** 64n - 1n), field(101, 1, doubles(1)), str(102, "future"),
    field(103, 5, floats(2)), vi(104 * 8 + 3), int(1, 1), vi(104 * 8 + 4)
  ]);
  const parsed = parseOnnxModel(Buffer.concat([model(unknown), unknown]));
  assert.deepEqual(parsed.graph.nodes, []);
});

test("initializer byte sizes include every inline storage field and encoding", () => {
  // Sizes are serialized data payload bytes, excluding protobuf tags and lengths.
  const cases = [
    ["double packed", 11, 16, [field(10, 2, doubles(1, 2))]],
    ["double unpacked", 11, 16, [field(10, 1, doubles(1)), field(10, 1, doubles(2))]],
    ["uint64 packed", 13, 11, [field(11, 2, Buffer.concat([vi(1), vi(2n ** 64n - 1n)]))]],
    ["uint64 unpacked", 13, 11, [int(11, 1), int(11, 2n ** 64n - 1n)]],
    ["float packed", 1, 8, [field(4, 2, floats(1, 2))]],
    ["float unpacked", 1, 8, [field(4, 5, floats(1)), field(4, 5, floats(2))]],
    ["int32 packed", 6, 3, [field(5, 2, Buffer.concat([vi(1), vi(128)]))]],
    ["int32 unpacked", 6, 3, [int(5, 1), int(5, 128)]],
    ["int64 packed", 7, 11, [field(7, 2, Buffer.concat([vi(1), vi(-1)]))]],
    ["int64 unpacked", 7, 11, [int(7, 1), int(7, -1)]],
    ["string", 8, 3, [str(6, "a"), str(6, "bc")]],
    ["raw", 11, 16, [field(9, 2, doubles(1, 2))]],
    ["multiple packed segments", 11, 16, [field(10, 2, doubles(1)), field(10, 2, doubles(2))]],
    ["metadata only", 1, 0, [str(12, "documentation"), field(100, 5, floats(1)), field(101, 1, doubles(1))]]
  ];
  const parsed = parseOnnxModel(model(...cases.map(([name, type, , values]) => initializer(name, type, ...values))));
  cases.forEach(([name, , expected], index) => assert.equal(parsed.graph.initializers[index].byteSize, expected, name));
});
