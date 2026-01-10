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

import { SelectionType } from '../../graph_ascend/type';
export interface HierarchyNodeType {
  x: number;
  y: number;
  width: number;
  height: number;
  expand: boolean;
  parentNode: string;
  label: string;
  name: string;
  isRoot: boolean;
  children?: HierarchyNodeType[];
  nodeType: 0 | 1 | 8 | 9;
  matchedNodeLink: string[];
  matchedDistributed: { communications_type: 'send' | 'receive'; nodes_info: { [string]: Array<string> } };
  precisionIndex: string;
  overflowLevel: string;
}

export type HierarchyObjectType = { [key: string]: HierarchyNodeType };

export type GraphType = 'NPU' | 'Bench' | 'Single';

export type PrecisionColorType = Record<
  string,
  {
    value: number[] | string;
    color: string;
  }
>;

export interface PreProcessDataConfigType {
  colors: PrecisionColorType;
  isOverflowFilter: boolean;
  graphType: GraphType;
}

export type PreProcessDataType = (
  hierarchyObject: { [key: string]: HierarchyNodeType },
  selectedNode: string,
  config: PreProcessDataConfigType,
  transform: { x: number; y: number; scale: number },
) => Array<any>;

export interface ContextMenuItem {
  text?: string;
  rankId?: number;
  component?: any;
  nodeName?: string;
  type?: number;
  children?: ContextMenuItem[];
}

export interface UseGraphType {
  bindInnerRect: (container: any, data: any) => void;
  bindOuterRect: (container: any, data: any) => void;
  bindText: (container: any, data: any) => void;
  preProcessData: (
    hierarchyObject: { [key: string]: HierarchyNodeType },
    data: any[],
    selectedNode: string,
    config: PreProcessDataConfigType,
    transform: { x: number; y: number; scale: number },
  ) => Array<any>;
  changeNodeExpandState: (nodeInfo: any, metaData: SelectionType) => Promise<any>;
  createComponent: (text, precision, colors: PreProcessDataConfigType['colors']) => any;
  updateHierarchyData: (graphType: string, metaData: SelectionType) => Promise<any>;
}

export interface TransformType {
  x: number;
  y: number;
  scale: number;
}

export interface NodeWithStatus {
  name: string;
  status: string;
}

export interface NodeWithColor {
  name: string;
  color?: string;
}

interface NodeInfoDetail {
  id: string;
  input_data: Record<string, Record<string, unknown> | string>;
  output_data: Record<string, Record<string, unknown> | string>;
  stack_info: string[];
  parallel_merge_info: string[];
}

/**
 * 后端接口直接返回
 */
export interface NodeInfoResult {
  npu?: NodeInfoDetail;
  bench?: NodeInfoDetail;
}

export interface ConvertedNodeInfoDetail {
  name: string;
  inputData: Record<string, Record<string, unknown> | string>;
  outputData: Record<string, Record<string, unknown> | string>;
  stackData: string;
  parallelMergeInfo: string;
}

export interface StackInfo {
  npuName?: string;
  benchName?: string;
  npuStack?: string;
  benchStack?: string;
  npuParallelMergeInfo?: string;
  benchParallelMergeInfo?: string;
}
