import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { defineConfig } from "vite";

function resolveModelPath() {
  const configuredPath = process.env.MODELVIZ_MODEL_PATH?.trim();
  const modelPath = configuredPath || path.join(os.homedir(), "Desktop", "mnist_mlp_best_seed1.onnx");

  if (modelPath === "~") return os.homedir();
  if (modelPath.startsWith("~/")) return path.join(os.homedir(), modelPath.slice(2));
  return path.resolve(modelPath);
}

export default defineConfig({
  plugins: [
    {
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
    }
  ]
});
