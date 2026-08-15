<div align="center">
  <img src="./public/brand/logo/logo-1024.png" alt="ModelViz" width="360" />

  **🔎 Turn raw ONNX graphs into clean model views 🔎**
</div>

ModelViz is a local React app for inspecting ONNX model structure. It parses model bytes in the browser, groups raw operators into semantic architecture blocks, and shows the raw graph beside a cleaner model-level view.

Use it to open an ONNX file from disk or the web, compare raw nodes with inferred groups, and inspect tensors, metadata, traceability, confidence, and model-profile matches.

## Install

```bash
git clone git@github.com:tsilva/modelviz.git
cd modelviz
pnpm install --frozen-lockfile
pnpm dev
```

Open the local Vite URL printed by the dev server, usually [http://127.0.0.1:5173](http://127.0.0.1:5173).

## Commands

```bash
pnpm dev      # start the local Vite dev server
pnpm audit --audit-level low  # audit the complete dependency graph
pnpm test     # run dependency security regressions
pnpm build    # build the production bundle
pnpm preview  # preview the production bundle locally
```

## Notes

- ModelViz starts with an empty state and prompts you to open an ONNX file.
- Uploaded files are parsed in the browser with the local ONNX protobuf parser.
- The web model browser searches Hugging Face ONNX model files and estimates browser parsing fit from file size, device memory, and CPU thread count before loading.
- The dev server still exposes `/api/model/default` for local experiments.
- Set `MODELVIZ_MODEL_PATH=/path/to/model.onnx` to choose the file served by that endpoint.
- If `MODELVIZ_MODEL_PATH` is not set, the dev server looks for `~/Desktop/mnist_mlp_best_seed1.onnx`.
- The package is marked `private` and does not declare a license.

## Local credentials

Private local values declared in `.keyenv.toml` live in macOS Keychain. Run
`keyenv doctor` to verify them and launch credential-dependent commands with
`keyenv run -- <command>`. Python, Node, and their child processes receive the
values through their normal environment APIs. Keep only public or non-secret
configuration in dotenv files.

## Architecture

![ModelViz architecture diagram](./architecture.png)

## License

No license file is currently included.
