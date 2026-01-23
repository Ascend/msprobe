/* -------------------------------------------------------------------------
 Copyright (c) 2025, Huawei Technologies.
 All rights reserved.

 Licensed under the Apache License, Version 2.0  (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
--------------------------------------------------------------------------------------------*/

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
