/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import { create } from 'zustand';
import type { DefaultOptionType } from 'antd/es/select';
import { loadConvertedGraphData } from '../api/board';

import type { ConvertModelDataType, ConvertParamsType } from './types/useVisualizedStore';
import type { MessageInstance } from 'antd/es/message/interface';
import { BUILD_STEP, INIT_CONVERT_GRAPH_ARGS } from '../common/constant';

export interface VisualizedState {
  npuPathItems: DefaultOptionType[];
  benchPathItems: DefaultOptionType[];
  layerMappingItems: DefaultOptionType[];
  convertedGraphArgs: ConvertParamsType;
  currentBuildStep: BUILD_STEP;
  setConvertedGraphArgs: (convertedGraphArgs: ConvertParamsType) => void;
  setCurrentBuildStep: (currentBuildStep: BUILD_STEP) => void;
  fetchConvertedGraphData: (messageApi: MessageInstance) => void;
}

export const useVisualizedStore = create<VisualizedState>()((set) => ({
  npuPathItems: [],
  benchPathItems: [],
  layerMappingItems: [],
  currentBuildStep: BUILD_STEP.BUILD_CONFIG,
  convertedGraphArgs: INIT_CONVERT_GRAPH_ARGS as ConvertParamsType,
  setConvertedGraphArgs: (convertedGraphArgs: ConvertParamsType) => {
    set({ convertedGraphArgs });
  },
  setCurrentBuildStep: (currentBuildStep: BUILD_STEP) => {
    set({ currentBuildStep });
  },
  fetchConvertedGraphData: async (messageApi) => {
    const { success, data, error } = await loadConvertedGraphData<ConvertModelDataType>();
    if (success) {
      const pathOptions = data?.dirs.map((item) => ({
        label: item,
        value: item,
      }));
      const layerMappingItems = data?.yaml_files.map((item) => ({
        label: item,
        value: item,
      }));
      set({
        npuPathItems: pathOptions,
        benchPathItems: pathOptions,
        layerMappingItems: layerMappingItems,
      });
    } else {
      messageApi.error(error);
    }
  },
}));
