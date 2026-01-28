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
                ? (OVERFLOW_COLOR[item.status as keyof typeof OVERFLOW_COLOR] ?? OVERFLOW_COLOR.default)
                : (PRECISION_ERROR_COLOR[item.status as keyof typeof PRECISION_ERROR_COLOR] ??
                  PRECISION_ERROR_COLOR.unmatched),
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
