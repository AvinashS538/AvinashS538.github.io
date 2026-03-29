// Service Worker for Stock Analyzer PWA
const CACHE_NAME = "stock-analyzer-v1";
const STATIC_ASSETS = ["/", "/static/index.html"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))));
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  // Only cache static assets, not API calls
  if (e.request.url.includes("/api/")) {
    return e.respondWith(fetch(e.request));
  }
  e.respondWith(caches.match(e.request).then((r) => r || fetch(e.request)));
});
