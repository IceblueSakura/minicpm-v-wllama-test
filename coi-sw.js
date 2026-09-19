// Minimal cross-origin-isolation service worker for static hosts that cannot set
// response headers (GitHub Pages). Injects COOP/COEP into same-origin navigations
// so SharedArrayBuffer becomes available; the page reloads itself (bounded retries).
// Required regardless of thread count: both wllama 3.6.1 WASM builds import shared memory.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {
  if (event.request.mode !== 'navigate') return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  event.respondWith(fetch(event.request).then(response => {
    const headers = new Headers(response.headers);
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  }));
});
