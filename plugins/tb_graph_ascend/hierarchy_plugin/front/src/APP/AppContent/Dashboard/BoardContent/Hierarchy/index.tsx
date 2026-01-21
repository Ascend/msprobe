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
              <g id="root" ref={containerRef} transform="translate(36,72) scale(1.8)"></g>
            </svg>
          </Dropdown>
        </div>
      </Spin>
    </div>
  );
};

export default Hierarchy;
