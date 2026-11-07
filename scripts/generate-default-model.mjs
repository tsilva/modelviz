import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { DEFAULT_MODEL } from "../src/lib/defaultModel.js";
import { parseOnnxModel } from "../src/lib/onnxParser.js";

const input = process.argv[2];
if (!input) throw new Error("Pass the pinned DistilGPT2 ONNX export as the first argument.");

const source = await readFile(input);
const sha256 = createHash("sha256").update(source).digest("hex");
assert.equal(sha256, "360ea17f6533bbbd9ab86b13dbc279e53cf977196f360a2f0da01f8f8b329803",
  "The source must match the pinned ONNX export.");

const model = parseOnnxModel(source);
const json = JSON.stringify(model) + "\n";
assert.deepEqual(JSON.parse(json), model, "Graph metadata must round-trip without loss.");
assert.equal(Buffer.byteLength(json), DEFAULT_MODEL.graphSizeBytes,
  "Update the example graph size after verifying a parser or model change.");
await writeFile(new URL("../public" + DEFAULT_MODEL.graphUrl, import.meta.url), json);
