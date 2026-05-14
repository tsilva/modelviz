import * as Sentry from "@sentry/react";

const SENTRY_DSN =
  import.meta.env.VITE_SENTRY_DSN ||
  "https://bfdc0775e16fcae8874a1256b9867ac6@o4511061698478080.ingest.de.sentry.io/4511388609675344";

let sentryInitialized = false;

export function initSentry() {
  if (sentryInitialized || !import.meta.env.PROD || !SENTRY_DSN) return;

  sentryInitialized = true;

  Sentry.init({
    dsn: SENTRY_DSN,
    enabled: true,
    environment: import.meta.env.MODE,
    release: import.meta.env.VITE_SENTRY_RELEASE,
    integrations: [Sentry.browserTracingIntegration()],
    tracesSampleRate: 0.1
  });
}

export const SentryErrorBoundary = Sentry.ErrorBoundary;
