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
