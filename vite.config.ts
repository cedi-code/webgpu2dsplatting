import { resolve } from 'path'
import { defineConfig } from 'vite'

export default defineConfig({
  base: '/webgpu2dsplatting/',
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        journey: resolve(__dirname, 'journey.html'),
      },
    },
  },
})