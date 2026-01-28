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
import * as d3 from 'd3';

export function fetchPbTxt(filepath: string): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    fetch(filepath).then((res) => {
      // Fetch does not reject for 400+.
      if (res.ok) {
        res.arrayBuffer().then(resolve, reject);
      } else {
        res.text().then(reject, reject);
      }
    });
  });
}

const removePrototypePollution = (obj: any): void => {
  if (obj && typeof obj === 'object') {
    for (let key in obj) {
      if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
        delete obj[key];
      } else if (typeof obj[key] === 'object') {
        removePrototypePollution(obj[key]);
      }
    }
  }
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
  return input.replace(/[&<>"'/]/g, (match) => htmlEscapeMap[match as keyof typeof htmlEscapeMap]);
};

export const safeJSONParse = (str: string, defaultValue: any = null): any => {
  // 只接受 string 类型
  if (typeof str !== 'string') {
    return defaultValue;
  }

  // 长度限制（3GB
  const maxLength = 3 * 1024 * 1024 * 1024;
  if (str.length > maxLength) {
    return defaultValue;
  }
  try {
    const res = JSON.parse(str);
    removePrototypePollution(res);
    return res;
  } catch (error) {
    return defaultValue;
  }
};

/**
 * 根据文本内容、字体大小和最大宽度，判断是否需要截断文本
 * @param text 文本内容
 * @param fontSize
 * @param maxWidth
 * @returns
 */
export function maybeTruncateString(content: string, fontSize: number, containerWidth: number): string {
  if (!content) {
    return '';
  }

  // 提前处理无需截断的情况
  if (measureTextWidth(content, fontSize) <= containerWidth) {
    return content;
  }

  let leftBound = 1;
  let rightBound = content.length;
  let optimalIndex = 0;

  // 逆向二分查找定位截断点
  while (leftBound <= rightBound) {
    const currentIndex = Math.floor((leftBound + rightBound) / 2);
    const testString = `${content.slice(0, currentIndex)}…`;
    if (measureTextWidth(testString, fontSize) <= containerWidth) {
      optimalIndex = currentIndex;
      leftBound = currentIndex + 1;
    } else {
      rightBound = currentIndex - 1;
    }
  }
  // 边界条件处理
  return optimalIndex > 0 ? `${content.substring(0, optimalIndex)}…` : `${content[0]}…`; // 极端情况保留首字符
}

/**
 * 计算文本宽度
 * @param text 文本内容
 * @param fontSize  字体大小
 * @returns
 */
export function measureTextWidth(text: string, fontSize: number): number {
  const canvas = document.createElement('canvas');
  const measurerContext = canvas.getContext('2d');
  if (measurerContext) {
    measurerContext.font = `${fontSize}px Roboto, sans-serif`;
  }
  return measurerContext?.measureText(text).width as number;
}
/**
 * 从 transform 字符串中解析出 x, y 和 scale 值
 * @param transformStr 如 "translate(100,200) scale(1.5)" 或 "translate(50,60)"
 * @returns 包含 x, y, scale 的对象，scale 默认为 1
 */

export function getContainerTransform(container: HTMLElement | SVGGElement | null) {
  if (container) {
    const str = container.getAttribute('transform') || '';
    return parseTransform(str);
  }
  return { x: 36, y: 72, scale: 1.8 };
}
export function parseTransform(transformStr: string): { x: number; y: number; scale: number } {
  // 默认值
  const result = { x: 0, y: 0, scale: 1 };

  if (!transformStr) {
    return result;
  }

  // 匹配 translate(X,Y) 部分

  const translateMatch = transformStr.match(/translate\((?<x>[\w\s.-]{1,100}),(?<y>[\w\s.-]{1,100})\)/);
  if (translateMatch) {
    result.x = parseFloat(translateMatch.groups?.x.trim() ?? '0');
    result.y = parseFloat(translateMatch.groups?.y.trim() ?? '0');
  }

  // 匹配 scale(Z) 部分
  const scaleMatch = transformStr.match(/scale\((?<scaleValue>[\w\s.-]{0,100},?)/);
  if (scaleMatch) {
    result.scale = parseFloat(scaleMatch.groups?.scaleValue.trim() ?? '0');
  }
  return result;
}
/**
 * 更改图形的位置
 * @param element
 * @param x
 * @param y
 * @param scale
 * @param duration
 */
export function changeGraphPosition(element: HTMLElement, x: any, y: any, scale: any, duration = 16) {
  d3.select(element).transition().duration(duration).attr('transform', `translate(${x},${y}) scale(${scale})`);
}

export function darkenColor(color: string, amount: number): string {
  // 统一提取 RGB(A) 分量
  let r: number;
  let g: number;
  let b: number;
  let a: number = 1;

  // 处理十六进制格式
  if (color.startsWith('#')) {
    const hex = color.replace(/^#/, '');
    const hexParts = hex.match(/[0-9a-f]{2}|[0-9a-f]{1,2}/gi) || [];

    // 处理缩写格式 (#abc → #aabbcc)
    const normalizedHex = hexParts
      .map((p) => (p.length === 1 ? p + p : p))
      .join('')
      .padEnd(6, '0');

    r = parseInt(normalizedHex.substring(0, 2), 16);
    g = parseInt(normalizedHex.substring(2, 4), 16);
    b = parseInt(normalizedHex.substring(4, 6), 16);

    // 处理 Alpha 通道 (#aabbccdd)
    if (normalizedHex.length >= 8) {
      a = parseInt(normalizedHex.substring(6, 8), 16) / 255;
    }
  }
  // 处理 RGB/RGBA 格式
  else if (color.startsWith('rgb')) {
    const match = color.match(/(\d*\.?\d+%?)/g) || [];
    const components = match.map(parseComponent);
    [r, g, b] = components.slice(0, 3).map((v) => Math.min(255, v));
    a = components[3] !== undefined ? components[3] : 1;
  }
  // 无效格式直接返回
  else {
    return color;
  }

  // 应用变暗逻辑（考虑 Alpha 通道）
  const applyDarkening = (value: number) => Math.max(0, Math.floor(value * a - amount));

  // 转换为十六进制
  const toHex = (value: number) => Math.min(255, Math.max(0, value)).toString(16).padStart(2, '0');

  return `#${[applyDarkening(r), applyDarkening(g), applyDarkening(b)].map(toHex).join('')}`;

  // 辅助函数：解析 RGB 分量（支持百分比和数值）
  function parseComponent(comp: string): number {
    if (comp.includes('%')) {
      return Math.round(parseFloat(comp) * 2.55);
    }
    return parseInt(comp, 10);
  }
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) {
    return '0 Bytes';
  }
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))}  ${sizes[i]}`;
}

/**
 * 不区分大小写的字符串包含检查
 * @param source 源字符串
 * @param searchStr 查询字符串
 * @returns
 */
export function lowerCaseInclude(source: string, searchStr: string): boolean {
  return source.toLowerCase().includes(searchStr.toLowerCase());
}
