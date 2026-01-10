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

export interface ApiResponse<T = any> {
  success: boolean; // 是否成功
  data?: T; // 响应数据
  error?: string; // 错误信息
}

export interface CurrentMetaDataType {
  run: string;
  tag: string;
  type: 'json' | 'db';
  lang: 'zh' | 'en';
  microStep?: number;
  step?: number;
  rank?: number;
}
