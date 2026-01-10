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
import { loadGraphAllNodeList, loadGraphConfig, loadGraphFileInfoList } from '../api/board';

import type { ApiResponse } from '../common/type';
import type { GraphConfigType, FileInfoListType, FileErrorListType, GraphAllNodeType } from './types/useGraphStore';
import type { DefaultOptionType } from 'antd/es/select';
import type { CurrentMetaDataType } from '../common/type';
import { CURRENT_PAGE, CURRENT_TAB, GRAPH_TYPE, initTransform } from '../common/constant';
import type { GraphType } from '../APP/AppContent/Dashboard/type';
import useGlobalStore from './useGlobalStore';

export interface GraphStoreType {
  messageApi: any;

  currentMetaDir: string;
  currentMetaFile: string;
  currentMetaFileType: 'json' | 'db';
  currentMetaStep: number;
  currentMetaRank: number;
  currentMetaMicroStep: number;
  currentLang: 'zh' | 'en';
  // 当前是否为精度溢出模式
  isOverflowMode: boolean;

  selectedNode: string;
  hightLightMatchedNode: {
    [key in GRAPH_TYPE]?: string;
  };

  fileInfoList: ApiResponse<FileInfoListType>;
  fileErrorList: FileErrorListType;
  metaDirOptions: DefaultOptionType[];
  metaFileOptions: DefaultOptionType[];
  colors: GraphConfigType['colors'];
  tooltips: GraphConfigType['tooltips'];
  // 当前数据是否有精度溢出数据
  hasOverflow: GraphConfigType['hasOverflow'];
  isSingleGraph: GraphConfigType['isSingleGraph'];
  task: GraphConfigType['task'];

  matchedConfigFilesOptions: DefaultOptionType[];
  microStepOptions: DefaultOptionType[];
  stepOptions: DefaultOptionType[];
  rankOptions: DefaultOptionType[];

  graphNodeList: GraphAllNodeType;
  transform: { x: number; y: number; scale: number };

  isShowNpuMiniMap: boolean;
  isShowBenchMiniMap: boolean;
  isSyncExpand: boolean;

  setMessageApi: (messageApi: any) => void;

  setSelectedNode: (node: string) => void;
  setHightLightMatchedNode: (node: {
    [key in GRAPH_TYPE]?: string;
  }) => void;
  setOverflowMode: (isOverflow: boolean) => void;
  setCurrentMetaDir: (dir: string) => void;
  setCurrentMetaFile: (file: string) => void;
  setCurrentMetaFileType: (type: GraphStoreType['currentMetaFileType']) => void;
  setCurrentMetaStep: (step: number) => void;
  setCurrentMetaRank: (rank: number) => void;
  setCurrentMetaMicroStep: (microStep: number) => void;
  setCurrentLang: (lang: 'zh' | 'en') => void;

  setTransform: (transform: { x: number; y: number; scale: number }) => void;

  setIsShowNpuMiniMap: (isShow: boolean) => void;
  setIsShowBenchMiniMap: (isShow: boolean) => void;
  setIsSyncExpand: (isSync: boolean) => void;

  updateCurrentMetaFileByDir: (dir: string) => void;

  getCurrentMetaData: () => CurrentMetaDataType;

  fetchFileInfoList: (selectMetaDir?: string) => Promise<void>;

  fetchGraphConfig: () => Promise<void>;

  fetchAllNodeList: () => Promise<void>;
}

const useGraphStore = create<GraphStoreType>()((set, get) => ({
  messageApi: null,
  currentMetaDir: '',
  currentMetaFile: '',
  currentMetaFileType: 'db',
  currentMetaStep: 0,
  currentMetaRank: 0,
  currentMetaMicroStep: -1,
  currentLang: 'zh',
  isOverflowMode: false,

  selectedNode: '',
  hightLightMatchedNode: {},

  fileInfoList: {} as ApiResponse<FileInfoListType>,
  metaDirOptions: [] as DefaultOptionType[],
  metaFileOptions: [] as DefaultOptionType[],
  fileErrorList: [] as unknown as FileErrorListType,

  colors: {},
  tooltips: '',
  hasOverflow: false,
  isSingleGraph: false,
  task: '',
  matchedConfigFilesOptions: [],

  stepOptions: [],
  rankOptions: [],
  microStepOptions: [],
  transform: initTransform,

  isShowNpuMiniMap: true,
  isShowBenchMiniMap: true,
  isSyncExpand: true,

  graphNodeList: {
    npuNodeList: [],
    benchNodeList: [],
    npuUnMatchNodes: [],
    benchUnMatchNodes: [],
    npuMatchNodes: [],
    benchMatchNodes: [],
  },

  getCurrentMetaData: () => {
    const currentMetaData = {
      run: get().currentMetaDir,
      tag: get().currentMetaFile,
      type: get().currentMetaFileType,
      lang: get().currentLang,
      microStep: get().currentMetaMicroStep,
      step: get().currentMetaStep,
      rank: get().currentMetaRank,
    };
    return currentMetaData;
  },

  setMessageApi: (messageApi: any) => set({ messageApi: messageApi }),

  setCurrentMetaDir: (dir: string) => set({ currentMetaDir: dir }),
  setCurrentMetaFile: (file: string) => set({ currentMetaFile: file }),
  setCurrentMetaFileType: (type: GraphStoreType['currentMetaFileType']) => set({ currentMetaFileType: type }),
  setCurrentMetaStep: (step: number) => set({ currentMetaStep: step }),
  setCurrentMetaRank: (rank: number) => set({ currentMetaRank: rank }),
  setCurrentMetaMicroStep: (microStep: number) => set({ currentMetaMicroStep: microStep }),
  setCurrentLang: (lang: 'zh' | 'en') => set({ currentLang: lang }),

  setSelectedNode: (node: string) => set({ selectedNode: node }),
  setHightLightMatchedNode: (hightLightMatchedNode: {
    [key in GraphType]?: string;
  }) => set({ hightLightMatchedNode }),
  setOverflowMode: (isOverflow: boolean) => set({ isOverflowMode: isOverflow }),

  setTransform: (transform: { x: number; y: number; scale: number }) => set({ transform }),

  setIsShowNpuMiniMap: (isShow: boolean) => set({ isShowNpuMiniMap: isShow }),
  setIsShowBenchMiniMap: (isShow: boolean) => set({ isShowBenchMiniMap: isShow }),
  setIsSyncExpand: (isSync: boolean) => set({ isSyncExpand: isSync }),

  updateCurrentMetaFileByDir: (dir: string) => {
    const type = get().fileInfoList?.data?.[dir]?.type as GraphStoreType['currentMetaFileType'];
    const metaFileList = get().fileInfoList?.data?.[dir]?.tags || [];
    const metaFileOptions = metaFileList.map((item) => {
      return {
        value: item,
        label: item,
      };
    });
    const currentMetaFile = get().fileInfoList?.data?.[dir]?.tags[0];
    set({ currentMetaFileType: type });
    set({ currentMetaFile: currentMetaFile });
    set({ currentMetaMicroStep: -1 });
    set({ metaFileOptions: metaFileOptions });
  },

  fetchFileInfoList: async (selectMetaDir: string = '') => {
    const result = await loadGraphFileInfoList<FileInfoListType>();
    const metaDirectory = Object.keys(result.data || {});
    const metaDirOptions = metaDirectory.map((item) => {
      return {
        value: item,

        label: item,
      };
    });
    set({ fileInfoList: result });
    set({
      fileErrorList: [] as unknown as FileErrorListType,
    });
    set({ metaDirOptions: metaDirOptions });
    if (metaDirOptions.length > 0) {
      set({ currentMetaDir: selectMetaDir || metaDirOptions[0].value });
    } else {
      const setCurrentPage = useGlobalStore.getState().setCurrentPage;
      const setCurrentTab = useGlobalStore.getState().setCurrentTab;
      setCurrentPage(CURRENT_PAGE.VISUALIZATION);
      setCurrentTab(CURRENT_TAB.VISUALIZED_TAB);
    }
  },

  fetchGraphConfig: async () => {
    const messageApi = get().messageApi;
    const selection = get().getCurrentMetaData();
    const currentLang = get().currentLang;
    if (!selection.run || !selection.tag) {
      return;
    }
    const { success, data, error } = await loadGraphConfig<GraphConfigType>({ metaData: selection });
    if (success) {
      set({ colors: data?.colors });
      set({ tooltips: data?.tooltips });
      set({ hasOverflow: data?.overflowCheck });
      set({ isSingleGraph: data?.isSingleGraph });
      set({ task: data?.task });
      const matchedConfigFilesOptions = data?.matchedConfigFiles.map((item) => {
        return {
          value: item,
          label: item,
        };
      });
      matchedConfigFilesOptions?.unshift({
        value: '',
        label: currentLang === 'zh' ? '未选择' : 'Not selected',
      });
      set({ matchedConfigFilesOptions: matchedConfigFilesOptions });
      if (Number(data?.microSteps)) {
        const microStepOptions = Array.from({ length: Number(data?.microSteps) + 1 }, (_, index) => ({
          label: index === 0 ? 'ALL' : String(index - 1),
          value: index - 1,
        }));
        set({ microStepOptions: microStepOptions });
      }
      if (data?.steps && data?.steps?.length > 0) {
        const stepOptions = data?.steps.map((item) => {
          return {
            value: item,
            label: item,
          };
        });
        set({ stepOptions: stepOptions });
        set({ currentMetaStep: data?.steps[0] });
      }

      if (data?.ranks && data?.ranks?.length > 0) {
        const rankOptions = data?.ranks.map((item) => {
          return {
            value: item,
            label: item,
          };
        });
        set({ rankOptions: rankOptions });
        set({ currentMetaRank: data?.ranks[0] });
      }
    } else {
      messageApi.error(error);
    }
  },

  fetchAllNodeList: async () => {
    const messageApi = get().messageApi;
    const selection = get().getCurrentMetaData();
    const { success, data, error } = await loadGraphAllNodeList<GraphAllNodeType>({ metaData: selection });
    if (success) {
      if (data) {
        set({ graphNodeList: data });
      }
    } else {
      messageApi.error(error);
    }
  },
}));

export default useGraphStore;
