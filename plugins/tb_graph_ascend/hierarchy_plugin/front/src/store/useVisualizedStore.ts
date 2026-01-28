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
