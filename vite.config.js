import fs from "node:fs/promises";
import { execSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { sentryVitePlugin } from "@sentry/vite-plugin";
import { defineConfig, loadEnv } from "vite";

function resolveModelPath() {
  const configuredPath = process.env.MODELVIZ_MODEL_PATH?.trim();
  const modelPath = configuredPath || path.join(os.homedir(), "Desktop", "mnist_mlp_best_seed1.onnx");

  if (modelPath === "~") return os.homedir();
  if (modelPath.startsWith("~/")) return path.join(os.homedir(), modelPath.slice(2));
  return path.resolve(modelPath);
}

function getGitSha() {
  try {
    return execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim();
  } catch {
    return "local";
  }
}

function localModelEndpoint() {
  return {
    name: "modelviz-local-model-endpoint",
    configureServer(server) {
      server.middlewares.use("/api/model/default", async (req, res, next) => {
        if (req.method !== "GET") {
          next();
          return;
        }

        const modelPath = resolveModelPath();

        try {
          const bytes = await fs.readFile(modelPath);
          res.statusCode = 200;
          res.setHeader("Content-Type", "application/octet-stream");
          res.setHeader("X-Model-File-Name", path.basename(modelPath));
          res.setHeader("X-Model-Path", modelPath);
          res.end(bytes);
        } catch (error) {
          res.statusCode = 404;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: error.message, path: modelPath }));
        }
      });
    }
  };
}

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const releaseName = env.SENTRY_RELEASE || env.VITE_SENTRY_RELEASE || getGitSha();
  const hasSentryUploadConfig = env.SENTRY_AUTH_TOKEN && env.SENTRY_ORG && env.SENTRY_PROJECT;

  return {
    define: {
      "import.meta.env.VITE_SENTRY_RELEASE": JSON.stringify(releaseName)
    },
    build: {
      sourcemap: command === "build" && hasSentryUploadConfig ? "hidden" : false
    },
    plugins: [
      localModelEndpoint(),
      command === "build" &&
        hasSentryUploadConfig &&
        sentryVitePlugin({
          authToken: env.SENTRY_AUTH_TOKEN,
          org: env.SENTRY_ORG,
          project: env.SENTRY_PROJECT,
          telemetry: false,
          release: {
            name: releaseName,
            inject: true
          },
          sourcemaps: {
            assets: "./dist/assets/**",
            filesToDeleteAfterUpload: ["./dist/assets/**/*.map"]
          }
        })
    ].filter(Boolean)
  };
});
