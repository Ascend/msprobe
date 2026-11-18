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

import { CONTINUOUS } from '../common/constant';
import { create } from 'zustand';
import { TreadDataType, DiffPairType } from '../common/type';
import { formatCurveLabel, formatDiffLabel } from '../utils';

export interface ContextStateType {
  heatMapData?: Array<any>;
  trendData: Array<TreadDataType>;
  diffPairs: Array<DiffPairType>;
  /** 图例中被单击隐藏的曲线名（持久化：加新曲线不重置） */
  hiddenLegendNames: string[];

  metric: string;
  stat: string;
  dimension: string;
  dimensionValue: string;
  heatMapType: string;
  viewMode: string;
  tags: string[];

  // 数据库文件相关
  dbFiles: string[];
  currentDbFile: string;

  dimX: string;
  dimY: string;
  loadingHeatMap: boolean;
  loadingLineChart: boolean;

  setHeatMapData: (heatMapData?: Array<any>) => void;
  setTrendData: (trendData: Array<TreadDataType>) => void;
  setDiffPairs: (diffPairs: Array<DiffPairType>) => void;
  addDiffPair: (curveA: string, curveB: string) => void;
  setHiddenLegendNames: (names: string[]) => void;
  /** 统一删除曲线：双击图例触发。diff 标签删差值曲线，其余按原始曲线删（连带引用它的 diff pair），并清理失效隐藏记录 */
  removeSeriesByName: (name: string) => void;
  setMetric: (metric: string) => void;
  setStat: (stat: string) => void;
  setDimension: (dimension: string) => void;
  setDimensionValue: (dimensionValue: string) => void;
  setHeatMapType: (heatMapType: string) => void;
  setViewMode: (viewMode: string) => void;
  setTags: (tags: string[]) => void;

  // 数据库文件相关方法
  setDbFiles: (dbFiles: string[]) => void;
  setCurrentDbFile: (currentDbFile: string) => void;

  setDimX: (dimX: string) => void;
  setDimY: (dimY: string) => void;
  setLoadingHeatMap: (loadingHeatMap: boolean) => void;
  setLoadingLineChart: (loadingLineChart: boolean) => void;
}

// 使用createWithEqualityFn而不是create
export const useGlobalStore = create<ContextStateType>((set) => ({
  heatMapData: [],
  trendData: [],
  diffPairs: [],
  hiddenLegendNames: [],
  metric: '',
  stat: '',
  dimension: '',
  dimensionValue: '',
  heatMapType: CONTINUOUS,
  viewMode: 'all',
  dimX: '',
  dimY: '',
  loadingHeatMap: false,
  loadingLineChart: false,
  tags: [],

  // 数据库文件相关初始值
  dbFiles: [],
  currentDbFile: '',

  setHeatMapData: (heatMapData?: Array<any>) => set({ heatMapData }),
  setTrendData: (trendData: Array<TreadDataType>) => set({ trendData }),
  setDiffPairs: (diffPairs: Array<DiffPairType>) => set({ diffPairs }),
  addDiffPair: (curveA: string, curveB: string) => {
    // crypto.randomUUID 仅在安全上下文(loopback/HTTPS)可用，非 loopback 访问时会抛异常，需回退
    const id =
      typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : `pair_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
    set((state) => ({ diffPairs: [...state.diffPairs, { id, curveA, curveB }] }));
  },
  setHiddenLegendNames: (names: string[]) => set({ hiddenLegendNames: names }),
  removeSeriesByName: (name: string) =>
    set((state) => {
      const { trendData, diffPairs, dimension, hiddenLegendNames } = state;
      const findByKey = (key: string) => trendData.find((t) => `${t.dimX}_${t.dimY}` === key);
      const labelOf = (p: DiffPairType) => {
        const a = findByKey(p.curveA);
        const b = findByKey(p.curveB);
        if (!a || !b) return '';
        return formatDiffLabel(formatCurveLabel(a, dimension), formatCurveLabel(b, dimension));
      };
      // name 是否为某条 diff 差值曲线的标签（"A - B"）
      const isDiff = diffPairs.some((p) => labelOf(p) === name);

      let nextTrend = trendData;
      let nextPairs = diffPairs;
      if (isDiff) {
        nextPairs = diffPairs.filter((p) => labelOf(p) !== name);
      } else {
        // 删除原始曲线，并连带删除引用它的 diff pair
        const target = trendData.find((t) => formatCurveLabel(t, dimension) === name);
        if (!target) return state; // 图例名找不到对应曲线，不动
        nextTrend = trendData.filter((t) => t !== target);
        const key = `${target.dimX}_${target.dimY}`;
        nextPairs = diffPairs.filter((p) => p.curveA !== key && p.curveB !== key);
      }

      // 清理隐藏记录：只保留仍然存在的曲线/diff 标签
      const validLabels = new Set<string>();
      nextTrend.forEach((t) => validLabels.add(formatCurveLabel(t, dimension)));
      nextPairs.forEach((p) => {
        const label = labelOf(p);
        if (label) validLabels.add(label);
      });
      return {
        trendData: nextTrend,
        diffPairs: nextPairs,
        hiddenLegendNames: hiddenLegendNames.filter((n) => validLabels.has(n)),
      };
    }),
  setMetric: (metric: string) => set({ metric }),
  setStat: (stat: string) => set({ stat }),
  setDimension: (dimension: string) => set({ dimension }),
  setDimensionValue: (dimensionValue: string) => set({ dimensionValue }),
  setHeatMapType: (heatMapType: string) => set({ heatMapType }),
  setViewMode: (viewMode: string) => set({ viewMode }),
  setTags: (tags: string[]) => set({ tags }),

  // 数据库文件相关方法
  setDbFiles: (dbFiles: string[]) => set({ dbFiles }),
  setCurrentDbFile: (currentDbFile: string) => set({ currentDbFile }),

  setDimX: (dimX: string) => set({ dimX }),
  setDimY: (dimY: string) => set({ dimY }),
  setLoadingHeatMap: (loadingHeatMap: boolean) => set({ loadingHeatMap }),
  setLoadingLineChart: (loadingLineChart: boolean) => set({ loadingLineChart }),
}));

// 调试/探针用：仅开发构建暴露 store 便于 E2E 断言，生产构建不注入
if (typeof window !== 'undefined' && process.env.NODE_ENV !== 'production') {
  (window as any).__trendStore = useGlobalStore;
}
