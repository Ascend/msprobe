/* -------------------------------------------------------------------------
 Copyright (c) 2025, Huawei Technologies.
 All rights reserved.

 Licensed under the Apache License, Version 2.0  (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
--------------------------------------------------------------------------------------------*/

import React, { memo } from 'react';
import { Spin, Splitter } from 'antd';
import { useShallow, shallow } from 'zustand/shallow';
import { useGlobalStore } from '../../store/useGlobalStore';
import HeatMap from './HeatMap';
import LineChart from './LineChart';

import './index.less';

const Graph = () => {
  const loadingHeatMap = useGlobalStore((state) => state.loadingHeatMap);
  const loadingLineChart = useGlobalStore((state) => state.loadingLineChart);

  return (
    <div className="graph-container">
      <div className="main-graph">
        <Splitter layout="vertical" style={{ height: '100%', boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)' }}>
          <Splitter.Panel style={{ height: '100%' }} defaultSize="70%">
            <Spin size="middle" spinning={loadingHeatMap}>
              <HeatMap />
            </Spin>
          </Splitter.Panel>
          <Splitter.Panel style={{ height: '100%' }} defaultSize="30%">
            <Spin size="middle" spinning={loadingLineChart}>
              <LineChart />
            </Spin>
          </Splitter.Panel>
        </Splitter>
      </div>
    </div>
  );
};

export default memo(Graph);
