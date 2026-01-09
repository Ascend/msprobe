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
import ReactECharts from 'echarts-for-react';
import { useGlobalStore } from '../../../store';
import { useShallow, shallow } from 'zustand/shallow';
import { DIMENSIONS_AXIS_MAP, MODULE_NAME_DIMENSION } from '../../../common/constant';

const LineChart = () => {
  const { metric, stat, dimension, trendData, dimX, dimY } = useGlobalStore(
    useShallow((state) => ({
      metric: state.metric,
      stat: state.stat,
      dimension: state.dimension,
      trendData: state.trendData,
      dimX: state.dimX,
      dimY: state.dimY,
    })),
    shallow,
  );
  const heatMapXAxisName = DIMENSIONS_AXIS_MAP[dimension || '']?.x;
  const heatMapYAxisName = DIMENSIONS_AXIS_MAP[dimension || '']?.y;
  let yAxisMap = new Map<string, string>();
  let xAxisData = trendData?.dimensions;
  let xAxisName = dimension;
  if (dimension === MODULE_NAME_DIMENSION) {
    xAxisName = 'Target Id';
    trendData?.dimensions.forEach((item, index) => {
      yAxisMap.set(String(index), item);
    });
    xAxisData = Array.from(yAxisMap.keys());
  }

  const option = {
    title: {
      text: `${metric} 分布图 (${heatMapXAxisName}: ${dimX || ' '} / ${heatMapYAxisName}: ${dimY || ' '}) `,
      left: 'center',
      top: 20,
      textStyle: {
        fontSize: 14,
        color: '#666',
        fontWeight: 'bold',
      },
    },
    grid: {
      top: 80,
      bottom: 80,
      left: 140,
      right: 140,
    },
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const yAxisValue = params[0]?.data;
        const dimensionValue = params[0]?.axisValue;
        const xAxisLabel = dimension === MODULE_NAME_DIMENSION ? 'Module Name' : xAxisName;
        const xAxisValue = dimension === MODULE_NAME_DIMENSION ? yAxisMap.get(dimensionValue) : dimensionValue;
        return `
                    <div style="font-weight:bold;margin-bottom:5px">${xAxisLabel}: ${xAxisValue}</div>
                    <div style="font-weight:bold;margin-bottom:5px">${stat}: ${yAxisValue}</div>
                    `;
      },
    },
    xAxis: {
      type: 'category',
      data: xAxisData,
      name: xAxisName,
      axisLabel: {
        rotate: 45,
        fontSize: 12,
        fontWeight: 'bold',
      },
      nameTextStyle: {
        fontSize: 12,
        fontWeight: 'bold',
        color: '#444',
      },
    },
    yAxis: {
      type: 'value',
      name: stat,
      axisLabel: {
        fontSize: 12,
        fontWeight: 'bold',
        formatter: (value) => {
          if (value === 0) return '0';
          const absValue = Math.abs(value);
          if (absValue < 1e-4 || absValue >= 1e4) {
            return value.toExponential(4);
          } else {
            let str = value.toFixed(4);
            str = str.replace(/(\.\d*?[1-9])0+$/, '$1').replace(/\.$/, '');
            return str;
          }
        },
      },
      nameTextStyle: {
        fontSize: 12,
        fontWeight: 'bold',
        color: '#444',
      },
    },
    series: [
      {
        name: '趋势',
        type: 'line',
        data: trendData?.values,
        lineStyle: {
          width: 2,
        },
        progressive: 1000,
        animation: true,
      },
    ],
    animation: true,
    animationDuration: 1000,
    animationEasing: 'cubicInOut',
  };
  return (
    <div className="LineChart" style={{ height: '32vh' }}>
      <ReactECharts option={option} style={{ height: '100%' }} />
    </div>
  );
};

export default memo(LineChart);
