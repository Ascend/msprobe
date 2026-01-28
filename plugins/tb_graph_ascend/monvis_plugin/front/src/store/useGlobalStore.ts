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
import { TreadDataType } from '../common/type';
export interface ContextStateType {
  heatMapData?: Array<any>;
  trendData: Array<TreadDataType>;

  metric: string;
  stat: string;
  dimension: string;
  dimensionValue: string;
  heatMapType: string;
  tags: string[];

  dimX: string;
  dimY: string;
  loadingHeatMap: boolean;
  loadingLineChart: boolean;

  setHeatMapData: (heatMapData?: Array<any>) => void;
  setTrendData: (trendData: Array<TreadDataType>) => void;
  setMetric: (metric: string) => void;
  setStat: (stat: string) => void;
  setDimension: (dimension: string) => void;
  setDimensionValue: (dimensionValue: string) => void;
  setHeatMapType: (heatMapType: string) => void;
  setTags: (tags: string[]) => void;

  setDimX: (dimX: string) => void;
  setDimY: (dimY: string) => void;
  setLoadingHeatMap: (loadingHeatMap: boolean) => void;
  setLoadingLineChart: (loadingLineChart: boolean) => void;
}

// 使用createWithEqualityFn而不是create
export const useGlobalStore = create<ContextStateType>((set, get) => ({
  heatMapData: [],
  trendData: [],
  metric: '',
  stat: '',
  dimension: '',
  dimensionValue: '',
  heatMapType: CONTINUOUS,
  dimX: '',
  dimY: '',
  loadingHeatMap: false,
  loadingLineChart: false,
  tags: [],

  setHeatMapData: (heatMapData?: Array<any>) => set({ heatMapData }),
  setTrendData: (trendData: Array<TreadDataType>) => set({ trendData }),
  setMetric: (metric: string) => set({ metric }),
  setStat: (stat: string) => set({ stat }),
  setDimension: (dimension: string) => set({ dimension }),
  setDimensionValue: (dimensionValue: string) => set({ dimensionValue }),
  setHeatMapType: (heatMapType: string) => set({ heatMapType }),
  setTags: (tags: string[]) => set({ tags }),

  setDimX: (dimX: string) => set({ dimX }),
  setDimY: (dimY: string) => set({ dimY }),
  setLoadingHeatMap: (loadingHeatMap: boolean) => set({ loadingHeatMap }),
  setLoadingLineChart: (loadingLineChart: boolean) => set({ loadingLineChart }),
}));
