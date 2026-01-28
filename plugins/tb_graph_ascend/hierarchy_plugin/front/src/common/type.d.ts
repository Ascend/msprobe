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

export interface GraphAllNodeType {
  npuNodeList: string[];
  benchNodeList: string[];
}

export interface GraphMatchedRelationsType {
  npuUnMatchNodes: string[];
  benchUnMatchNodes: string[];
  npuMatchNodes: Record<string, string>;
  benchMatchNodes: Record<string, string>;
}
