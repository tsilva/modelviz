<p align="center">
  <img src="./logo.png" alt="ModelViz" width="360" />
  <br />
  <!-- repo-tagline:start -->
  <strong>🔎 Turn raw ONNX graphs into clean model views 🔎</strong>
  <!-- repo-tagline:end -->
</p>

ModelViz turns an ONNX file into an interactive graph. It parses model bytes
in your browser and groups related operators into architecture blocks.

## Try it

Open [ModelViz](https://modelviz.tsilva.eu/) to inspect the included
DistilGPT2 example. In **Split** view, select an architecture block to
highlight the raw operators it contains. The inspector shows tensor metadata
and the evidence for inferred groups. You can also open an ONNX file from
disk or use the model browser.

## Run locally

```bash
git clone https://github.com/tsilva/modelviz.git
cd modelviz
pnpm install --frozen-lockfile
pnpm dev --port auto
```

Open the printed local URL. See the [inspection guide](docs/model-inspection.md)
for mapping limits, example assets, and development checks, or
[production delivery](docs/production-delivery.md) for deployment.

## License

[MIT License](LICENSE).
