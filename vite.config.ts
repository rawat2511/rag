import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Cross-origin isolation lets ONNX Runtime use multi-threaded WASM (3-4x faster embedding).
// "credentialless" keeps the Hugging Face / jsDelivr model downloads working.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

export default defineConfig({
  base: './',
  plugins: [react()],
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: 'es' },
  build: { target: 'esnext', chunkSizeWarningLimit: 2500 },
});
