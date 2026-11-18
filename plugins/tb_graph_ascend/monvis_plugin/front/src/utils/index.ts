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
import { DIMENSIONS_AXIS_MAP, MODULE_NAME_DIMENSION } from '../common/constant';
import type { TreadDataType } from '../common/type';

/** 原始曲线显示名（图例/下拉/标签共用）：如 "Step: 0 / Rank: 1" */
export const formatCurveLabel = (trend: TreadDataType, dimension: string): string => {
  const axisNames = DIMENSIONS_AXIS_MAP[dimension] || { x: dimension, y: dimension };
  return `${axisNames.x}: ${trend.dimX ?? ' '} / ${axisNames.y}: ${trend.dimY ?? ' '}`;
};

/** diff 差值曲线显示名（图例/标签共用）：如 "A - B" */
export const formatDiffLabel = (labelA: string, labelB: string): string => `${labelA} - ${labelB}`;

export const formatSegmentLabel = (min, max) => {
  const format = (value) => {
    if (value === Infinity) return '∞';
    if (value === -Infinity) return '-∞';
    return value.toExponential(2);
  };

  if (min === -Infinity) return `< ${format(max)}`;
  if (max === Infinity) return `> ${format(min)}`;
  return `${format(min)} ~ ${format(max)}`;
};

export const escapeHTML = (input: string): string => {
  const htmlEscapeMap = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
    '/': '&#x2F;',
  };
  return String(input)?.replace(/[&<>"'/]/g, (match) => htmlEscapeMap[match as keyof typeof htmlEscapeMap]);
};

/**
 * 折线图数据点构造：与 LineChart 完全一致的 x 映射。
 * module_name 维度下 dimensions 是 "<id>_<name>" 形态，取 id 段作为 x 坐标；
 * 值做 ±1e8 钳制，extra 绑定原始维度串（悬浮框维度信息用）。
 */
export const buildLineSeriesData = (
  trend: TreadDataType,
  dimension: string,
): Array<{ value: [number, number]; extra: string }> => {
  return trend.values.map((value, index) => {
    const dimValue =
      dimension === MODULE_NAME_DIMENSION
        ? String(trend.dimensions[index])?.split('_')?.[0] || ''
        : String(trend.dimensions[index]);
    if (Number(value) > 100000000) value = '100000000';
    if (Number(value) < -100000000) value = '-100000000';
    return {
      value: [Number(dimValue), Number(value)],
      extra: trend.dimensions[index],
    };
  });
};
