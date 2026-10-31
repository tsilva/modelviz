import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import test from "node:test";

const require = createRequire(import.meta.url);
const pluginRequire = createRequire(require.resolve("@sentry/vite-plugin"));
const sentryEntry = pluginRequire.resolve("@sentry/bundler-plugins/core");
const sentryRequire = createRequire(sentryEntry);
// Load the exact offline helper used for upload selection and post-upload deletion.
// Importing this module does not initialize the plugin, CLI, or an upload.
const { globFiles } = sentryRequire(join(dirname(sentryEntry), "glob.js"));
const glob = sentryRequire("glob");
const globRequire = createRequire(sentryRequire.resolve("glob"));
const { Minimatch, minimatch } = globRequire("minimatch");

test("Sentry resolves the reviewed glob dependency graph", () => {
  assert.equal(sentryRequire("glob/package.json").version, "14.0.2");
  assert.equal(globRequire("minimatch/package.json").version, "11.0.1");
  assert.equal(globRequire("path-scurry/package.json").version, "3.0.0");
  const matcherRequire = createRequire(globRequire.resolve("minimatch"));
  assert.equal(matcherRequire("@isaacs/nonrecursive-glob-match/package.json").version, "1.0.1");
});

test("Sentry preserves asset, source-map deletion, and exclusion selection offline", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "modelviz-sentry-assets-"));
  try {
    const files = ["dist/assets/app.js", "dist/assets/app.js.map", "dist/assets/nested/chunk.mjs",
      "dist/assets/nested/chunk.mjs.map", "dist/assets/style.css", "dist/assets/.private.js",
      "dist/assets/excluded/skip.js", "dist/other.js", "outside.map"];
    for (const file of files) {
      await mkdir(dirname(join(cwd, file)), { recursive: true });
      await writeFile(join(cwd, file), "synthetic asset");
    }
    const config = fs.readFileSync(new URL("../vite.config.js", import.meta.url), "utf8");
    assert.ok(config.includes('assets: "./dist/assets/**"'));
    assert.ok(config.includes('filesToDeleteAfterUpload: ["./dist/assets/**/*.map"]'));
    const selected = async (patterns, options = {}) =>
      (await globFiles(patterns, { cwd, ...options })).map(file => relative(cwd, file)).sort();
    assert.deepEqual(await selected("./dist/assets/**"), files.slice(0, 5).concat(files[6]).sort());
    assert.deepEqual(await selected(["./dist/assets/**/*.map"]),
      ["dist/assets/app.js.map", "dist/assets/nested/chunk.mjs.map"]);
    assert.deepEqual(await selected(["./dist/assets/**/*.{js,mjs}"], { ignore: "**/excluded/**" }),
      ["dist/assets/app.js", "dist/assets/nested/chunk.mjs"]);
    assert.deepEqual(await selected("./dist/assets/missing/**"), []);
  } finally {
    await rm(cwd, { recursive: true, force: true });
  }
});

test("Minimatch bounds globstar-parent pattern amplification", () => {
  const matcher = new Minimatch("", { optimizationLevel: 2, braceExpandMax: 32 });
  const started = performance.now();
  const expanded = matcher.firstPhasePreProcess([("**/../a/b/".repeat(8) + "asset.js").split("/")]);
  assert.ok(expanded.length <= 32, `expanded to ${expanded.length} patterns`);
  assert.ok(performance.now() - started < 1_000);
  assert.equal(minimatch("dist/assets/nested/app.js", "dist/**/app.{js,mjs}", { optimizationLevel: 2 }), true);
  assert.equal(minimatch("dist/assets/app.css", "dist/**/app.{js,mjs}", { optimizationLevel: 2 }), false);
});

test("Minimatch handles adversarial wildcard and extglob nonmatches in bounded time", () => {
  // A subprocess deadline also bounds a future regression to a backtracking matcher.
  const result = spawnSync(process.execPath, ["-e", `
    const assert = require("node:assert/strict");
    const { minimatch } = require(${JSON.stringify(globRequire.resolve("minimatch"))});
    const candidate = "a".repeat(160) + "b";
    assert.equal(minimatch(candidate, "*a".repeat(24) + "c"), false);
    assert.equal(minimatch(candidate, "+(a|aa)*c"), false);
  `], { timeout: 2_000, encoding: "utf8" });
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(minimatch("chunk.mjs", "@(app|chunk).@(js|mjs)"), true);
  assert.equal(minimatch(".private.js", "*.js"), false);
});

for (const api of ["glob", "globStream"]) {
  test(`${api} stops directory reads after cancellation`, async () => {
    const cwd = await mkdtemp(join(tmpdir(), "modelviz-glob-abort-"));
    try {
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        const directory = join(cwd, `a${a}`, `b${b}`, "nested");
        await mkdir(directory, { recursive: true });
        await writeFile(join(directory, "asset.js"), "synthetic asset");
      }
      const controller = new AbortController();
      const reason = new Error("synthetic traversal budget");
      let reads = 0;
      const instrumentedFs = {
        readdir(path, options, callback) {
          reads++;
          fs.readdir(path, options, (error, entries) => {
            if (reads === 1) controller.abort(reason);
            callback(error, entries);
          });
        }
      };
      const options = { cwd, fs: instrumentedFs, signal: controller.signal, nodir: true };
      const walk = api === "glob" ? glob.glob("**/*.js", options) : (async () => {
        for await (const file of glob.globStream("**/*.js", options)) assert.ok(file);
      })();
      await assert.rejects(walk, error => error === reason);
      await delay(100); // Include work that a broken walker schedules after rejection.
      assert.equal(reads, 1, "aborted traversal continued reading directories");
      const preAborted = AbortSignal.abort(reason);
      await assert.rejects(glob.glob("**/*.js", { cwd, signal: preAborted }), error => error === reason);
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });
}
