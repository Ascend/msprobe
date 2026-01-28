/* -------------------------------------------------------------------------
 *  This file is part of the MindStudio project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  mode: 'development', // 明确指定开发模式
  devtool: 'eval-cheap-source-map', // 开发环境推荐使用的 source map 类型

  entry: {
    app: './src/index.tsx', // 保持与生产环境一致的入口
  },

  output: {
    filename: '[name].bundle.js', // 使用带 hash 的文件名
    path: path.resolve(__dirname, 'dist'),
    publicPath: '/', // 确保 dev server 的静态资源路径正确
  },

  devServer: {
    static: {
      directory: path.join(__dirname, 'dist'), // 服务资源目录
    },
    compress: false,
    proxy: [
      {
        context: (pathname) => {
          return !pathname.match(/(\.js|\.css|\.html|\.ico|\.svg)$/);
        },
        target: 'http://127.0.0.1:6006',
        changeOrigin: true,
        secure: false,
        pathRewrite: {
          '^/(.*)': '/data/plugin/TrendVis/$1', // 路径转换核心逻辑
        },
        on: {
          error: (err, req, res) => {
            // 安全处理响应对象
            if (res && !res.headersSent) {
              res.writeHead(500, { 'Content-Type': 'text/plain' });
              res.end('Proxy Error');
            }
          },
          proxyReqWs: (proxyReq, req, socket) => {
            // WebSocket 错误专属处理
            socket.on('error', (error) => {});
          },
        },
      },
    ],
    webSocketServer: {
      type: 'ws',
      options: {
        path: '/ws',
        noInfo: true,
      },
    },
    http2: false, // 推荐启用HTTP/2
    https: false, // 根据实际需要配置
    hot: true, // 启用热模块替换
    liveReload: true, // 启用实时重新加载
    port: 8080, // 自定义端口号
    open: true, // 自动打开浏览器
    client: {
      overlay: {
        errors: true,
        warnings: false,
      }, // 在浏览器中显示编译错误
    },
    headers: {
      'X-Proxy-Source': 'webpack-dev-server',
    },
  },

  module: {
    rules: [
      {
        test: /\.html$/,
        use: [
          {
            loader: 'html-loader',
            options: {
              sources: false, // 开发环境不需要优化资源路径
            },
          },
        ],
      },
      {
        test: /\.tsx?$/, // 支持 .ts 和 .tsx 文件
        use: {
          loader: 'ts-loader',
          options: {
            transpileOnly: true,
            experimentalWatchApi: true,
          },
        },
        exclude: /node_modules/,
      },
      {
        test: /\.css$/i,
        use: [
          'style-loader',
          {
            loader: 'css-loader',
            options: {
              sourceMap: true, // 启用 CSS source maps
            },
          },
        ],
      },
      {
        test: /\.less$/i,
        use: [
          'style-loader', // 将 JS 字符串生成为 <style> 标签
          'css-loader', // 解析 @import 和 url() 等引用
          {
            loader: 'less-loader', // 编译 Less 文件为 CSS
            options: {
              lessOptions: {
                javascriptEnabled: true, // 如果你使用了 JavaScript 内嵌的 Less 函数，则需要此选项
              },
            },
          },
        ],
      },
    ],
  },

  resolve: {
    extensions: ['.ts', '.tsx', '.js', '.json'], // 添加 .json 扩展名解析
  },

  plugins: [
    new HtmlWebpackPlugin({
      inject: 'body',
      template: './public/index.html',
      minify: false, // 开发环境不压缩 HTML
    }),
  ],

  optimization: {
    removeAvailableModules: false,
    removeEmptyChunks: false,
    splitChunks: false, // 禁用代码拆分加速编译
  },
};
