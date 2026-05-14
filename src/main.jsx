import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import { initAnalytics } from "./lib/analytics.js";
import { initSentry, SentryErrorBoundary } from "./lib/sentry.js";
import "./styles.css";

initSentry();
initAnalytics();

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <SentryErrorBoundary fallback={<div className="app-fallback">Something went wrong.</div>}>
      <App />
    </SentryErrorBoundary>
  </React.StrictMode>
);
