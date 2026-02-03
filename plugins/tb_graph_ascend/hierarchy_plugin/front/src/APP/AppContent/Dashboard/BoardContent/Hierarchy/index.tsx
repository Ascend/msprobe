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

import useGraphStore from '../../../../../store/useGraphStore';
import { GRAPH_TYPE } from '../../../../../common/constant';
import { useHierarchyGraph } from './useHierarchy';
import MiniMap from '../MiniMap';
import './index.less';
import { Dropdown, Spin } from 'antd';

interface HierarchyProps {
  graphType: GRAPH_TYPE;
  testid: string;
}

const Hierarchy = (params: HierarchyProps) => {
  const { graphType, testid } = params;

  const isShowNpuMiniMap = useGraphStore((state) => state.isShowNpuMiniMap);
  const isShowBenchMiniMap = useGraphStore((state) => state.isShowBenchMiniMap);
  // 常量
  const isShowMiniMap =
    (graphType === GRAPH_TYPE.NPU && isShowNpuMiniMap) ||
    (graphType === GRAPH_TYPE.BENCH && isShowBenchMiniMap) ||
    (graphType === GRAPH_TYPE.SINGLE && isShowNpuMiniMap);

  // 使用自定义 Hook
  const { expanding, transform, setTransform, contextMenuItems, containerRef, graphRef, hierarchyObjectRef } =
    useHierarchyGraph(graphType);

  return (
    <div style={{ position: 'relative', height: '100%', width: '100%' }} data-testid={testid}>
      {isShowMiniMap && (
        <div className="mini-map">
          <MiniMap
            transform={transform}
            setTransform={setTransform}
            graphType={graphType}
            graph={graphRef.current}
            container={containerRef.current}
            hierarchyObject={hierarchyObjectRef.current}
          />
        </div>
      )}
      <Spin spinning={expanding} wrapperClassName={'hierarchy-spin'}>
        <div className="board-content " style={{ height: '100%', width: '100%' }}>
          <Dropdown menu={{ items: contextMenuItems }} trigger={['contextMenu']} destroyOnHidden>
            <svg id="graph" ref={graphRef} style={{ height: '100%', width: '100%' }}>
              <g ref={containerRef} transform="translate(36,72) scale(1.8)"></g>
            </svg>
          </Dropdown>
        </div>
      </Spin>
    </div>
  );
};

export default Hierarchy;
