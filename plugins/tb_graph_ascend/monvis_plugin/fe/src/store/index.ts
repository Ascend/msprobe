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

import { createWithEqualityFn } from 'zustand/traditional';
import { CONTINUOUS } from '../common/constant';
import type { ContextStateType } from '../common/type';

// 使用createWithEqualityFn而不是create
export const useGlobalStore = createWithEqualityFn<ContextStateType>(
  (set) => ({
    heatMaphData: [],
    trendData: { dimensions: [], values: [] },
    metric: '',
    stat: '',
    dimension: '',
    dimensionValue: '',
    heatMapType: CONTINUOUS,
    dimX: '',
    dimY: '',
    loadingHeatMap: false,
    loadingLineChart: false,
    setContextState: (newState) => set((state) => ({ ...state, ...newState })),
  }),
  Object.is, // 指定默认的相等性函数，可以是浅层比较
);
