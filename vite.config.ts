import { resolve } from 'path'
import { defineConfig } from 'vite'

export default defineConfig({
  base: '/webgpu-img2splat/',
  build: {
    rollupOptions: {
      input: {
        demo: resolve(__dirname, 'index.html'),
        motivation: resolve(__dirname, 'journey-intro.html'),
        playground: resolve(__dirname, 'playground.html'),
        primitive: resolve(__dirname, 'journey-primitive.html'),
        optimization: resolve(__dirname, 'journey-optimization.html'),
        parallel: resolve(__dirname, 'journey-parallel.html'),
        densification: resolve(__dirname, 'journey-densification.html'),
        depth : resolve(__dirname, 'journey-depth.html'),
        test: resolve(__dirname, 'test.html'),

      },
    },
  },
})