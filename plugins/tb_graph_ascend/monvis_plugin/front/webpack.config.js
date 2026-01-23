/* -------------------------------------------------------------------------
 Copyright (c) 2025, Huawei Technologies.
 All rights reserved.

 Licensed under the Apache License, Version 2.0  (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
--------------------------------------------------------------------------------------------*/

const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const InlineChunkHtmlPlugin = require('inline-chunk-html-plugin');

module.exports = {
  entry: {
    app: './src/index.tsx',
  },
  output: {
    filename: 'index.js',
    path: path.resolve(__dirname, 'dist'),
  },
  module: {
    rules: [
      {
        test: /\.(html)$/,
        use: {
          loader: 'html-loader',
        },
      },
      {
        test: /\.tsx?$/,
        use: {
          loader: 'ts-loader',
          options: {
            transpileOnly: true,
          },
        },
        exclude: /node_modules/,
      },
      {
        test: /\.css$/i,
        use: ['style-loader', 'css-loader'],
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
    extensions: ['.ts', '.js', '.tsx'],
  },
  plugins: [
    new CleanWebpackPlugin(),
    new HtmlWebpackPlugin({
      inject: 'body',
      template: './public/index.html',
    }),
    new InlineChunkHtmlPlugin(HtmlWebpackPlugin, [/.*/]),
  ],
};
