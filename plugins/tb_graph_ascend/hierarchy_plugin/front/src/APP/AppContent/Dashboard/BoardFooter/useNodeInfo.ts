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
import { isEmpty, cloneDeep } from 'lodash';
import { safeJSONParse } from '../../../../common/utils';
import { BENCH_PREFIX, NPU_PREFIX } from '../../../../common/constant';
import type { ConvertedNodeInfoDetail, StackInfo } from '../type';

export interface UseNodeInfoType {
  getIoDataSet: (
    npuNode: any,
    benchNode: any,
    type: 'inputData' | 'outputData',
  ) => {
    matchedIoDataset: Array<Record<string, unknown>>;
    unMatchedNpuIoDataset: Array<Record<string, unknown>>;
    unMatchedBenchIoDataset: Array<Record<string, unknown>>;
  };
  getDetailDataSet: (npuNode: any, benchNode: any) => StackInfo;
}

const useNodeInfo = (): UseNodeInfoType => {
  /**
   * 获取匹配的输入输出数据
   * @param npuNode NPU节点信息
   * @param benchNode 匹配的节点信息
   * @param name 'inputData' | 'outputData'
   * @returns { matchedDataset: Array<{}>, unMatchedNpuDataset: Array<{}>, unMatchedBenchDataset: Array<{}> }
   */
  const getIoDataSet = (
    npuNode: ConvertedNodeInfoDetail | undefined,
    benchNode: ConvertedNodeInfoDetail | undefined,
    type: 'inputData' | 'outputData',
  ): {
    matchedIoDataset: Array<Record<string, unknown>>;
    unMatchedNpuIoDataset: Array<Record<string, unknown>>;
    unMatchedBenchIoDataset: Array<Record<string, unknown>>;
  } => {
    if (isEmpty(npuNode?.[type]) && isEmpty(benchNode?.[type])) {
      return {
        matchedIoDataset: [],
        unMatchedNpuIoDataset: [],
        unMatchedBenchIoDataset: [],
      };
    }
    const npuNodeName = npuNode?.name;
    const benchNodeName = benchNode?.name;
    const npuData = cloneDeep(npuNode?.[type]); // 获取当前节点的输入数据
    const benchData = cloneDeep(benchNode?.[type]); // 获取匹配节点的输入数据

    const matchedIoDataset: Array<Record<string, unknown>> = []; // 初始化输入数据集
    const unMatchedBenchIoDataset: Array<Record<string, unknown>> = [];
    const unMatchedNpuIoDataset: Array<Record<string, unknown>> = [];
    const npuKeys = Object.keys(npuData || {});
    const benchKeys = Object.keys(benchData || {});
    const minLength = Math.min(npuKeys.length, benchKeys.length);
    for (let i = 0; i < minLength; i++) {
      const npuKey = npuKeys[i];
      const benchKey = benchKeys[i];
      matchedIoDataset.push({
        name: npuKey.replace(`${npuNodeName}.`, ''),
        isMatched: true,
        ...(typeof npuData?.[npuKey] === 'object' ? npuData[npuKey] : {}),
      });
      matchedIoDataset.push({
        name: benchKey.replace(`${benchNodeName}.`, ''),
        isBench: true,
        isMatched: true,
        ...(typeof benchData?.[benchKey] === 'object' ? benchData[benchKey] : {}),
      });
      delete (npuData as Record<string, Record<string, unknown>>)[npuKey];
      delete (benchData as Record<string, Record<string, unknown>>)[benchKey];
    }
    Object.keys(npuData || {}).forEach((key) => {
      if (typeof npuData?.[key] === 'object') {
        unMatchedNpuIoDataset.push({
          name: key.replace(`${npuNodeName}.`, ''),
          ...npuData[key],
        });
      }
    });
    Object.keys(benchData || {}).forEach((key) => {
      if (typeof benchData?.[key] === 'object') {
        unMatchedBenchIoDataset.push({
          name: key.replace(`${benchNodeName}.`, ''),
          isBench: true,
          ...benchData[key],
        });
      }
    });
    return { matchedIoDataset, unMatchedNpuIoDataset, unMatchedBenchIoDataset };
  };

  const getDetailDataSet = (
    npuNode: ConvertedNodeInfoDetail | undefined,
    benchNode: ConvertedNodeInfoDetail | undefined,
  ): StackInfo => {
    if (isEmpty(npuNode) && isEmpty(benchNode)) {
      return {};
    }
    const npuName = npuNode?.name?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '');
    const benchName = benchNode?.name?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '');
    // 获取stackInfo
    const npuStack = npuNode?.stackData ? safeJSONParse(npuNode.stackData).join('\n') : undefined;
    const benchStack = benchNode?.stackData ? safeJSONParse(benchNode.stackData).join('\n') : undefined;
    // 获取parallel_merge_info
    const npuParallelMergeInfo = npuNode?.parallelMergeInfo
      ? safeJSONParse(npuNode.parallelMergeInfo).join('\n')
      : undefined;
    const benchParallelMergeInfo = benchNode?.parallelMergeInfo
      ? safeJSONParse(benchNode.parallelMergeInfo).join('\n')
      : undefined;
    return {
      npuName,
      benchName,
      npuStack,
      benchStack,
      npuParallelMergeInfo,
      benchParallelMergeInfo,
    };
  };

  return {
    getIoDataSet,
    getDetailDataSet,
  };
};

export default useNodeInfo;
