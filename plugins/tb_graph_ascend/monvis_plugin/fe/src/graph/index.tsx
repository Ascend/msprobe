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
import { Spin } from 'antd';
import { useShallow, shallow } from 'zustand/shallow';
import { useGlobalStore } from '../store';
import HeatMap from './components/HeatMap';
import LineChart from './components/LineChart';

import './index.less';

const Graph = () => {
  const { loadingHeatMap, loadingLineChart } = useGlobalStore(
    useShallow((state) => ({
      loadingHeatMap: state.loadingHeatMap,
      loadingLineChart: state.loadingLineChart,
    })),
    shallow,
  );
  return (
    <div className="graph-container">
      <div className="graph-title">模型监控数据看板</div>
      <div className="main-graph">
        <Spin size="middle" spinning={loadingHeatMap}>
          <HeatMap />
        </Spin>
        <Spin size="middle" spinning={loadingLineChart}>
          <LineChart />
        </Spin>
      </div>
    </div>
  );
};

export default memo(Graph);
