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

export enum SIDER_TYPE {
  FILE,
  SEARCH,
  PRECISION,
  MATCH,
  THEME,
  LANGUAGE,
}

export const enum DIRS {
  COMPARE_DIR = 'Compare',
  SINGLE_DIR = 'Single',
  MD5_DIR = 'Md5',
  COMMUNICATION_DIR = 'Communication',
  OVERFLOW_DIR = 'Overflow',
}

export const enum FILES {
  COMPARE_FILE = 'compare_20250902171456',
  SINGLE_FILE = 'build_20251227174822',
  MD5_FILE = 'compare_20260104171250',
  COMMUNICATION_FILE = 'build_collective',
  OVERFLOW_FILE = 'build_20260110160141',
}

export const MAX_DIFF_PIXELS = 100;
