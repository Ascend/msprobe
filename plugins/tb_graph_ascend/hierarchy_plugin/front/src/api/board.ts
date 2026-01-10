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

import type { GRAPH_TYPE } from '../common/constant';
import request from '../common/request';
import type { CurrentMetaDataType } from '../common/type';
import type { ConvertParamsType } from '../store/types/useVisualizedStore';

export interface RequestGraphConfigParamsType {
  metaData: CurrentMetaDataType;
}

export interface FilterNodesRequestParams {
  metaData: CurrentMetaDataType;
  type: string;
  values: Array<string | number | number[]>;
}

export interface GetNodeInfoRequestParams {
  metaData: CurrentMetaDataType;
  nodeInfo: {
    nodeName: string;
    nodeType: GRAPH_TYPE;
  };
}

export const loadGraphFileInfoList = <T>() => request<T>({ url: 'load_meta_dir', method: 'POST' });

export const loadGraphConfig = <T>(params: RequestGraphConfigParamsType) =>
  request<T>({ url: 'loadGraphConfigInfo', method: 'POST', data: params });

export const requestChangeNodeExpandState = <T>(params: any) =>
  request<T>({ url: 'changeNodeExpandState', method: 'POST', data: params });

/**
 * 筛选节点列表
 * @param params
 * @returns
 */
export const filterNodes = <T>(params: FilterNodesRequestParams) =>
  request<T>({ url: 'filterNodes', method: 'POST', data: params });

/**
 * 获取全量节点列表
 * @param params
 * @returns
 */
export const loadGraphAllNodeList = <T>(params: RequestGraphConfigParamsType) =>
  request<T>({ url: 'loadGraphAllNodeList', method: 'POST', data: params });

/**
 * 获取选中节点信息
 * @param params
 * @returns
 */
export const getNodeInfo = <T>(params: GetNodeInfoRequestParams) =>
  request<T>({ url: 'getNodeInfo', method: 'POST', data: params });
export const loadConvertedGraphData = <T>() => request<T>({ url: 'loadConvertedGraphData', method: 'GET' });

export const requestConvertToGraph = <T>(params: ConvertParamsType) =>
  request<T>({ url: 'convertToGraph', method: 'POST', data: params });

export const loadGraphData = <T>(params: CurrentMetaDataType) =>
  request<T>({ url: 'loadGraphData', method: 'GET', params: params });
