import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
  server: { host: '0.0.0.0' },
  preview: { host: '0.0.0.0' },
});
