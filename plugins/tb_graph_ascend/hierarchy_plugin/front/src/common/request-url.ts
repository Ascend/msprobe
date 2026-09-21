/*
 * -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
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

const DEVELOPMENT_PLUGIN_BASE_URL = '/data/plugin/graph_ascend';

export function buildRequestUrl(url: string, isDevelopment: boolean): string {
  const normalizedUrl = url.replace(/^\/+/, '');
  if (isDevelopment) {
    return `${DEVELOPMENT_PLUGIN_BASE_URL}/${normalizedUrl}`;
  }
  return normalizedUrl;
}
