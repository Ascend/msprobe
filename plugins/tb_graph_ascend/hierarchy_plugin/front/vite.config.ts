import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
  plugins: [
    react(),
    viteSingleFile(), // 必须放在最后
  ],
  css: {
    preprocessorOptions: {
      less: {
        javascriptEnabled: true,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/data/plugin/graph_ascend': {
        target: 'http://127.0.0.1:6006',
        changeOrigin: true,
        secure: false,
      },
      '/getConvertProgress': {
        //SSE 路径转换白名单
        target: 'http://127.0.0.1:6006',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => {
          return '/data/plugin/graph_ascend' + path;
        },
      },
    },
  },
});
