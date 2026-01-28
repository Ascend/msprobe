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
import axios, { type AxiosResponse, AxiosError } from 'axios';
import type { ApiResponse } from './type';
interface RequestOptions {
  url: string; // 请求地址
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE'; // 请求方法，默认为 GET
  data?: any; // 请求体数据（适用于 POST、PUT）
  params?: any; // URL 参数（适用于 GET）
  headers?: Record<string, string>; // 自定义请求头
  timeout?: number; // 超时时间（毫秒），默认 10 秒
}

export default async function request<T = any>(options: RequestOptions): Promise<ApiResponse<T>> {
  const { url, method = 'GET', data = null, params = null, headers = {}, timeout = 60000 * 3 } = options;
  const controller = new AbortController();
  const signal = controller.signal;
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  try {
    if (typeof params === 'object' && params !== null) {
      if ('metaData' in params) {
        params.metaData.type = 'rank' in params.metaData ? 'db' : 'json';
      }
    }
    const baseURL = '/data/plugin/graph_ascend';
    const fullUrl = baseURL ? `${baseURL.replace(/\/$/, '')}/${url.replace(/^\//, '')}` : url;
    const response: AxiosResponse<T> = await axios({
      url: fullUrl,
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
    clearTimeout(timeoutId);
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
