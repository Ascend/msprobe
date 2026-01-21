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
import { Spin } from 'antd';
import NodeListPanel from './NodeListPanel';
import FilterPanel from './FilterPanel';
import { useState } from 'react';
import { filterNodes, type FilterNodesRequestParams } from '../../../../../api/board';
import useGraphStore from '../../../../../store/useGraphStore';
import type { NodeWithColor, NodeWithStatus } from '../../type';
import { OVERFLOW_COLOR, PRECISION_ERROR_COLOR } from '../../../../../common/constant';
import { useTranslation } from 'react-i18next';

const PrecisionSider = (): React.JSX.Element => {
  const getCurrentMetaData = useGraphStore((state) => state.getCurrentMetaData);
  const isOverflowMode = useGraphStore((state) => state.isOverflowMode);
  const messageApi = useGraphStore((state) => state.messageApi);
  const { t } = useTranslation();
  const [nodes, setNodes] = useState<NodeWithColor[]>([]);
  const [spinning, setSpin] = useState<boolean>(false);

  const changeFilteredNodes = async (values: Array<string | number>) => {
    if (values.length === 0) {
      setNodes([]);
      return;
    }
    const metaData = getCurrentMetaData();
    const params: FilterNodesRequestParams = {
      metaData,
      type: isOverflowMode ? 'overflow' : 'precision',
      values,
    };
    setSpin(true);
    const { success, data, error } = await filterNodes<NodeWithStatus[]>(params).finally(() => setSpin(false));
    if (success) {
      if (data) {
        setNodes(
          data.map((item) => {
            return {
              name: item.name,
              color: isOverflowMode
                ? OVERFLOW_COLOR[item.status as keyof typeof OVERFLOW_COLOR] ?? OVERFLOW_COLOR.default
                : PRECISION_ERROR_COLOR[item.status as keyof typeof PRECISION_ERROR_COLOR] ??
                  PRECISION_ERROR_COLOR.unmatched,
            };
          }),
        );
      }
    } else {
      messageApi.error(error);
    }
  };

  return (
    <div data-testid="precisionPanel">
      <Spin spinning={spinning} tip={t('loading')}>
        <FilterPanel onFilterNodes={changeFilteredNodes} />
        <NodeListPanel nodeList={nodes} />
      </Spin>
    </div>
  );
};

export default PrecisionSider;
