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

import React, { useRef, useEffect, useState, memo } from 'react';
import { useShallow } from 'zustand/shallow';
import ReactECharts from 'echarts-for-react';
import { message } from 'antd';
import { useGlobalStore } from '../../../store';
import useHeatMap from './useHeatMap';
import { isEmpty } from 'lodash';
import { TrendRequestParams, TrendResponseData } from '../type';
import { isEqual } from 'lodash';

const HeatMap = () => {
  const useHeatMapInstance = useHeatMap();
  const { metric, stat, dimension, dimensionValue, heatMaphData, heatMapType, setContextState } = useGlobalStore(
    useShallow((state) => ({
      metric: state.metric,
      stat: state.stat,
      dimension: state.dimension,
      dimensionValue: state.dimensionValue,
      heatMaphData: state.heatMaphData,
      heatMapType: state.heatMapType,
      setContextState: state.setContextState,
    })),
    (oldTreats, newTreats) => {
      return (
        isEqual(oldTreats.heatMaphData, newTreats.heatMaphData) && isEqual(oldTreats.heatMapType, newTreats.heatMapType)
      );
    },
  );

  const [option, setOption] = useState({});
  const yAxisMapRef = useRef(new Map());

  useEffect(() => {
    if (!isEmpty(heatMaphData)) {
      const { yAxisData, option } = useHeatMapInstance.updateHeatMap(heatMaphData, dimension, heatMapType);
      setOption(option);
      yAxisMapRef.current = yAxisData;
    }
  }, [heatMaphData, heatMapType]);

  const onChartClick = async (echartsParams) => {
    const dimX = echartsParams.data[0];
    const dimY = yAxisMapRef.current[echartsParams.data[1]];
    const params: TrendRequestParams = {
      metric: metric,
      stat: stat,
      dimension: dimension,
      value: dimensionValue,
      dimX,
      dimYIdx: dimY,
    } as TrendRequestParams;
    setContextState({ loadingLineChart: true });
    const result: TrendResponseData = await useHeatMapInstance.loadGraphData(params);
    setContextState({ loadingLineChart: false });
    if (result?.error) {
      message.error(result.error);
      return;
    }
    const trendData = result.data;
    setContextState({ trendData, dimX, dimY });
  };

  const onDataZoom = (params) => {
    console.log(params);
  };
  return (
    <div className="heatmap" style={{ height: '64vh', borderBottom: '2px solid #ccc' }}>
      <ReactECharts
        option={option}
        style={{ height: '100%' }}
        onEvents={{
          click: onChartClick,
          dataZoom: onDataZoom,
        }}
      />
    </div>
  );
};

export default memo(HeatMap);
