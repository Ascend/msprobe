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
import { Tabs } from 'antd';
import NodeDetailPanel from './NodeDetailPanel';
import NodeInfoPanel from './NodeInfoPanel';
import useGraphStore from '../../../../store/useGraphStore';
import { useEffect, useState } from 'react';
import { getNodeInfo, type GetNodeInfoRequestParams } from '../../../../api/board';
import { BENCH_PREFIX, GRAPH_TYPE, NPU_PREFIX } from '../../../../common/constant';
import type { StackInfo, ConvertedNodeInfoDetail, NodeInfoDetail, NodeInfoResult } from '../type';
import { isEmpty } from 'lodash';
import useNodeInfo, { type UseNodeInfoType } from './useNodeInfo';
import { useTranslation } from 'react-i18next';

const convertNodeInfo = (nodeInfo: NodeInfoDetail): ConvertedNodeInfoDetail => {
  return {
    name: nodeInfo.id.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), ''),
    inputData: nodeInfo.input_data,
    outputData: nodeInfo.output_data,
    stackData: !isEmpty(nodeInfo.stack_info) ? JSON.stringify(nodeInfo.stack_info) : '',
    parallelMergeInfo: !isEmpty(nodeInfo.parallel_merge_info) ? JSON.stringify(nodeInfo.parallel_merge_info) : '',
  };
};

const BorderFooter = (): React.JSX.Element => {
  const getCurrentMetaData = useGraphStore((state) => state.getCurrentMetaData);
  const isSingleGraph = useGraphStore((state) => state.isSingleGraph);
  const selectedNode = useGraphStore((state) => state.selectedNode);
  const messageApi = useGraphStore((state) => state.messageApi);
  const useNodeInfoHook: UseNodeInfoType = useNodeInfo();
  const { t } = useTranslation();

  const [npuNodeName, setNpuNodeName] = useState<string>('');
  const [benchNodeName, setBenchNodeName] = useState<string>('');
  const [ioDataset, setIoDataset] = useState<Array<Record<string, unknown>>>([]);
  const [stackInfo, setStackInfo] = useState<StackInfo>({});

  useEffect(() => {
    if (!selectedNode) {
      setIoDataset([]);
      setStackInfo({});
      return;
    }
    const metaData = getCurrentMetaData();
    const params: GetNodeInfoRequestParams = {
      metaData,
      nodeInfo: {
        nodeName: selectedNode.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), ''), // 去掉前缀
        nodeType: isSingleGraph
          ? GRAPH_TYPE.SINGLE
          : selectedNode.startsWith(NPU_PREFIX)
          ? GRAPH_TYPE.NPU
          : GRAPH_TYPE.BENCH,
      },
    };

    getNodeInfo<NodeInfoResult>(params)
      .then((res) => {
        const { success, data, error } = res;
        if (success) {
          if (data) {
            const npuNode = data.npu ? convertNodeInfo(data.npu) : undefined;
            const benchNode = data.bench ? convertNodeInfo(data.bench) : undefined;

            // 考虑选中的节点是匹配节点的情况
            setNpuNodeName(npuNode?.name ?? '');
            setBenchNodeName(benchNode?.name ?? '');
            const inputDataset = useNodeInfoHook.getIoDataSet(npuNode, benchNode, 'inputData');
            const outputDataSet = useNodeInfoHook.getIoDataSet(npuNode, benchNode, 'outputData');
            setIoDataset([
              ...inputDataset.matchedIoDataset,
              ...outputDataSet.matchedIoDataset,
              ...inputDataset.unMatchedNpuIoDataset,
              ...outputDataSet.unMatchedNpuIoDataset,
              ...inputDataset.unMatchedBenchIoDataset,
              ...outputDataSet.unMatchedBenchIoDataset,
            ]);
            const detailData = useNodeInfoHook.getDetailDataSet(npuNode, benchNode);
            setStackInfo(detailData);
          }
        } else {
          messageApi.error(error);
        }
      })
      .catch((err) => {
        messageApi.error(err);
      });
  }, [selectedNode]);

  return (
    <Tabs
      style={{ height: '100%', padding: '4px 16px' }}
      defaultActiveKey="1"
      items={[
        {
          label: t('comparisonDetails'),
          key: '1',
          children: <NodeDetailPanel npuName={npuNodeName} benchName={benchNodeName} data={ioDataset} />,
        },
        {
          label: t('nodeInfo'),
          key: '2',
          children: <NodeInfoPanel data={stackInfo} />,
        },
      ]}
    />
  );
};

export default BorderFooter;
