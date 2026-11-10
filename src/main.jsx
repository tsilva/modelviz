import React from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import { Analytics } from "@vercel/analytics/react";
import App from "./App.jsx";
import { initAnalytics } from "./lib/analytics.js";
import { createDefaultModelView } from "./lib/defaultModel.js";
import { initSentry, SentryErrorBoundary } from "./lib/sentry.js";
import "./styles.css";

initSentry();
initAnalytics();

const root = document.getElementById("root");
const snapshot = document.getElementById("default-model-graph");
let initialModelView;
if (snapshot) {
  try {
    initialModelView = createDefaultModelView(JSON.parse(snapshot.textContent));
  } catch {
    // Let the application load the default graph from its normal URL.
  }
  snapshot.remove();
}

const app = (
  <React.StrictMode>
    <SentryErrorBoundary fallback={<div className="app-fallback">Something went wrong.</div>}>
      <App initialModelView={initialModelView} />
      <Analytics />
    </SentryErrorBoundary>
  </React.StrictMode>
);

if (initialModelView && root.hasChildNodes()) {
  hydrateRoot(root, app);
} else {
  createRoot(root).render(app);
}
