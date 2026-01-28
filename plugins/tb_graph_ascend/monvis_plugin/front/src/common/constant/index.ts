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

export const STEP_DIMENSION = 'step';
export const RANK_DIMENSION = 'rank';
export const MODULE_NAME_DIMENSION = 'module_name';

export const CONTINUOUS = 'continuous';
export const PIECEWISE = 'piecewise';

export const DIMENSIONS_OPTIONS = [
  { value: STEP_DIMENSION, label: 'Step' },
  { value: RANK_DIMENSION, label: 'Rank' },
  { value: MODULE_NAME_DIMENSION, label: 'Module Name' },
];

export const HEATMAP_TYPE = [
  { value: CONTINUOUS, label: '渐变模式' },
  { value: PIECEWISE, label: '分段模式' },
];

export const DIMENSIONS_AXIS_MAP = {
  [STEP_DIMENSION]: {
    x: 'Rank',
    y: 'Module Name',
  },
  [RANK_DIMENSION]: {
    x: 'Step',
    y: 'Module Name',
  },
  [MODULE_NAME_DIMENSION]: {
    x: 'Step',
    y: 'Rank',
  },
};

export const CLEAR_ICON = 'path://M10 10 H90 V90 H10 Z M25 25 L75 75 M75 25 L25 75';
