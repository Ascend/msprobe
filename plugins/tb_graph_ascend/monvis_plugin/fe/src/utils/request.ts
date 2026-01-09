/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import axios, { AxiosResponse, AxiosError } from 'axios';
interface RequestOptions {
  url: string; // 请求地址
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE'; // 请求方法，默认为 GET
  data?: any; // 请求体数据（适用于 POST、PUT）
  params?: any; // URL 参数（适用于 GET）
  headers?: Record<string, string>; // 自定义请求头
  timeout?: number; // 超时时间（毫秒），默认 10 秒
}
interface ApiResponse<T = any> {
  success: boolean; // 是否成功
  data?: T; // 响应数据
  error?: string; // 错误信息
}
export default async function request<T = any>(options: RequestOptions): Promise<ApiResponse<T>> {
  const { url, method = 'GET', data = null, params = null, headers = {}, timeout = 60000 * 3 } = options;

  try {
    const controller = new AbortController();
    const signal = controller.signal;
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    const response: AxiosResponse<T> = await axios({
      url,
      method,
      data,
      params,
      headers: {
        'Content-Type': 'application/json', // 默认 Content-Type
        ...headers, // 自定义请求头覆盖默认值
      },
      maxBodyLength: Infinity, // 允许发送大文件
      signal, // 绑定信号以支持超时
    });

    clearTimeout(timeoutId);
    if (response.status >= 200 && response.status < 300) {
      return response.data as ApiResponse<T>;
    } else {
      return {
        success: false,
        error: `HTTP Error: ${response.status} - ${response.statusText}`,
      };
    }
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      if (axiosError.code === 'ECONNABORTED') {
        return {
          success: false,
          error: '请求超时，请稍后重试。',
        };
      } else if (axiosError.response) {
        const { status, statusText } = axiosError.response;
        return {
          success: false,
          error: `HTTP Error: ${status} - ${statusText}`,
        };
      } else if (axiosError.request) {
        return {
          success: false,
          error: '网络错误，未收到服务器响应。',
        };
      } else {
        return {
          success: false,
          error: axiosError.message || '未知错误',
        };
      }
    } else {
      return {
        success: false,
        error: (error as Error).message || '未知错误',
      };
    }
  }
}
