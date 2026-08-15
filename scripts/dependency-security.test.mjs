import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import test from "node:test";

import { transformSync } from "@babel/core";
import { expand } from "brace-expansion";
import { customAlphabet } from "nanoid";
import postcss from "postcss";

const require = createRequire(import.meta.url);

function installedVersion(dependency) {
  let directory = dirname(require.resolve(dependency));
  while (directory !== dirname(directory)) {
    try {
      const metadata = JSON.parse(readFileSync(join(directory, "package.json"), "utf8"));
      if (metadata.name === dependency && metadata.version) return metadata.version;
    } catch {
      // Walk upward until the package root is found.
    }
    directory = dirname(directory);
  }
  throw new Error(`Unable to find package metadata for ${dependency}`);
}

function tuple(value) {
  return value.split(".").map(Number);
}

test("patched dependency floors are installed", () => {
  assert.deepEqual(tuple(installedVersion("@babel/core")), [7, 29, 7]);
  assert.deepEqual(tuple(installedVersion("brace-expansion")), [5, 0, 9]);
  assert.deepEqual(tuple(installedVersion("nanoid")), [3, 3, 18]);
  assert.deepEqual(tuple(installedVersion("postcss")), [8, 5, 26]);
});

test("Brace Expansion handles consecutive empty groups in bounded time", () => {
  const malicious = "{}".repeat(2_000);
  const started = performance.now();
  assert.deepEqual(expand(malicious), [malicious]);
  assert.ok(performance.now() - started < 1_000);
  assert.deepEqual(expand("model-{small,large}.onnx"), ["model-small.onnx", "model-large.onnx"]);
});

test("Nano ID preserves zero-size and normal generator behavior", () => {
  assert.equal(customAlphabet("abcdef", 0)(), "");
  assert.match(customAlphabet("abcdef", 12)(), /^[a-f]{12}$/);
});

test("PostCSS and Babel preserve legitimate transformations", async () => {
  const css = await postcss([]).process(".model { color: #fff; }", { from: undefined });
  assert.match(css.css, /\.model/);

  const javascript = transformSync("const model = value => value + 1;", {
    ast: false,
    code: true,
    sourceMaps: false,
  });
  assert.match(javascript?.code ?? "", /const model/);
});
