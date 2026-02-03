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
import { loadGraphConfig, loadGraphFileInfoList } from '../api/board';
import type { ApiResponse, GraphAllNodeType, GraphMatchedRelationsType } from '../common/type';
import type { GraphConfigType, FileInfoListType, FileErrorListType } from './types/useGraphStore';
import type { DefaultOptionType } from 'antd/es/select';
import type { CurrentMetaDataType } from '../common/type';
import { CURRENT_PAGE, CURRENT_TAB, GRAPH_TYPE, INIT_TRANSFORM } from '../common/constant';
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
  // 当前是否为溢出筛选模式
  isOverflowMode: boolean;

  isInitHierarchySwitch: boolean;

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
  // 当前数据是否执行了溢出检测
  hasOverflow: GraphConfigType['hasOverflow'];
  isSingleGraph: GraphConfigType['isSingleGraph'];
  task: GraphConfigType['task'];

  matchedConfigFilesOptions: string[];
  microStepOptions: DefaultOptionType[];
  stepOptions: DefaultOptionType[];
  rankOptions: DefaultOptionType[];

  graphNodeList: GraphAllNodeType;
  graphMatchedRelations: GraphMatchedRelationsType;
  transform: { x: number; y: number; scale: number };

  isShowNpuMiniMap: boolean;
  isShowBenchMiniMap: boolean;
  isSyncExpand: boolean;
  isMatchedStatusSwitch: boolean;
  metaDataCacheInSearch: string;
  metaDataCacheInMatch: string;

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
  setMatchedConfigFilesOptions: (options: string[]) => void;
  setGraphNodeList: (graphNodes: GraphAllNodeType) => void;
  setGraphMatchedRelations: (matchedRelations: GraphMatchedRelationsType) => void;
  switchMatchedStatus: () => void;
  updateMetaDataCacheInSearch: () => void;
  updateMetaDataCacheInMatch: () => void;

  updateCurrentMetaFileByDir: (dir: string) => void;

  getCurrentMetaData: () => CurrentMetaDataType;

  fetchFileInfoList: (selectMetaDir?: string) => Promise<void>;

  fetchGraphConfig: () => Promise<void>;

  isSearchCached: () => boolean;
  isMatchCached: () => boolean;
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
  metaDataCacheInSearch: '',
  metaDataCacheInMatch: '',

  isInitHierarchySwitch: false,

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
  transform: INIT_TRANSFORM,

  isShowNpuMiniMap: true,
  isShowBenchMiniMap: true,
  isSyncExpand: true,
  isMatchedStatusSwitch: true,

  graphNodeList: {
    npuNodeList: [],
    benchNodeList: [],
  },
  graphMatchedRelations: {
    npuUnMatchNodes: [],
    benchUnMatchNodes: [],
    npuMatchNodes: {},
    benchMatchNodes: {},
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
  setMatchedConfigFilesOptions: (options: string[]) => set({ matchedConfigFilesOptions: options }),
  setGraphNodeList: (graphNodes: GraphAllNodeType) => set({ graphNodeList: graphNodes }),
  setGraphMatchedRelations: (matchedRelations: GraphMatchedRelationsType) =>
    set({ graphMatchedRelations: matchedRelations }),
  switchMatchedStatus: () => set({ isMatchedStatusSwitch: !get().isMatchedStatusSwitch }),
  updateMetaDataCacheInSearch: () => set({ metaDataCacheInSearch: JSON.stringify(get().getCurrentMetaData()) }),
  updateMetaDataCacheInMatch: () => set({ metaDataCacheInMatch: JSON.stringify(get().getCurrentMetaData()) }),
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
      fileErrorList: (result.error || []) as unknown as FileErrorListType,
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
      set({ matchedConfigFilesOptions: data?.matchedConfigFiles ?? [] });
      if (Number(data?.microSteps)) {
        const microStepOptions: Array<{ label: number | string; value: number }> = Array.from(
          { length: Number(data?.microSteps) },
          (_, index) => ({
            label: index,
            value: index,
          }),
        );
        microStepOptions.unshift({
          label: 'ALL',
          value: -1,
        });
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
      //触发初始化图的状态刷新，解决单双图切换的图初始化问题
      set({ isInitHierarchySwitch: !get().isInitHierarchySwitch });
    } else {
      messageApi.error(error);
    }
  },

  isSearchCached: (): boolean => {
    const currentMetaDataStr = JSON.stringify(get().getCurrentMetaData());
    return get().metaDataCacheInSearch === currentMetaDataStr;
  },

  isMatchCached: (): boolean => {
    const currentMetaDataStr = JSON.stringify(get().getCurrentMetaData());
    return get().metaDataCacheInMatch === currentMetaDataStr;
  },
}));

export default useGraphStore;
