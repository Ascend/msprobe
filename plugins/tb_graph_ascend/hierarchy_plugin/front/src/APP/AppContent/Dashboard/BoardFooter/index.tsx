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
  const isMatchedStatusSwitch = useGraphStore((state) => state.isMatchedStatusSwitch);
  const useNodeInfoHook: UseNodeInfoType = useNodeInfo();
  const { t } = useTranslation();

  const [npuNodeName, setNpuNodeName] = useState<string>('');
  const [benchNodeName, setBenchNodeName] = useState<string>('');
  const [ioDataset, setIoDataset] = useState<Array<Record<string, unknown>>>([]);
  const [stackInfo, setStackInfo] = useState<StackInfo>({});

  useEffect(() => {
    if (!selectedNode) {
      setNpuNodeName('');
      setBenchNodeName('');
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
  }, [selectedNode, isMatchedStatusSwitch]);

  return (
    <Tabs
      style={{ height: '100%', padding: '4px 16px' }}
      defaultActiveKey="1"
      items={[
        {
          label: isSingleGraph ? t('singleDetails') : t('comparisonDetails'),
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
