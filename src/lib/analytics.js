const GA_MEASUREMENT_ID = "G-JV1DRFZCL6";

let analyticsInitialized = false;

export function initAnalytics() {
  if (analyticsInitialized || !import.meta.env.PROD || typeof window === "undefined") return;

  analyticsInitialized = true;
  window.dataLayer = window.dataLayer || [];
  window.gtag = window.gtag || function gtag() {
    window.dataLayer.push(arguments);
  };

  const script = document.createElement("script");
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`;
  document.head.appendChild(script);

  window.gtag("js", new Date());
  window.gtag("config", GA_MEASUREMENT_ID);
}
