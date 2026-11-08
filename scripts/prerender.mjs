import { readFile, writeFile } from "node:fs/promises";
import { createElement } from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";
import { createDefaultModelView, DEFAULT_MODEL } from "../src/lib/defaultModel.js";

const renderer = await createServer({
  configFile: false,
  envDir: false,
  mode: "production",
  appType: "custom",
  server: { middlewareMode: true, ws: false, watch: null },
});

try {
  const { default: App } = await renderer.ssrLoadModule("/src/App.jsx");
  const graph = await readFile(new URL(`../public${DEFAULT_MODEL.graphUrl}`, import.meta.url), "utf8");
  const initialModelView = createDefaultModelView(JSON.parse(graph));
  const markup = renderToString(createElement(App, { initialModelView }));
  const output = new URL("../dist/index.html", import.meta.url);
  let html = await readFile(output, "utf8");
  const placeholder = '<div id="root"></div>';
  if (!html.includes(placeholder)) throw new Error("Missing ModelViz prerender root");

  const stylesheet = html.match(/<link rel="stylesheet" crossorigin href="(\/assets\/[^"/]+\.css)">/u);
  if (stylesheet) {
    const css = await readFile(new URL(`.${stylesheet[1]}`, output), "utf8");
    // Keep linked CSS when it needs its own URL for resource resolution.
    if (!/(?:@import\b|url\s*\(|<\/style\b)/iu.test(css)) {
      html = html.replace(stylesheet[0], () => `<style>${css}</style>`);
    }
  }

  const snapshot = graph.replaceAll("<", "\\u003c");
  await writeFile(output, html.replace(placeholder, () =>
    `<div id="root">${markup}</div><script type="application/json" id="default-model-graph">${snapshot}</script>`
  ));
} finally {
  await renderer.close();
}
