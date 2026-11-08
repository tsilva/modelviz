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
  const html = await readFile(output, "utf8");
  const placeholder = '<div id="root"></div>';
  if (!html.includes(placeholder)) throw new Error("Missing ModelViz prerender root");

  const snapshot = graph.replaceAll("<", "\\u003c");
  await writeFile(output, html.replace(placeholder, () =>
    `<div id="root">${markup}</div><script type="application/json" id="default-model-graph">${snapshot}</script>`
  ));
} finally {
  await renderer.close();
}
