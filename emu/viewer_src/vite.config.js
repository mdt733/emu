import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  base: './',
  build: { rollupOptions: { input: {
    main: fileURLToPath(new URL('./index.html', import.meta.url)),
    prismLab: fileURLToPath(new URL('./prism-lab.html', import.meta.url)),
  } } },
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
  server: { host: '0.0.0.0' },
  preview: { host: '0.0.0.0' },
});
