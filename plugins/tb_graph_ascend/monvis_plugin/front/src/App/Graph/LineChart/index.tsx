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
import { isEmpty } from 'lodash';
import { useGlobalStore } from '../../../store/useGlobalStore';
import { CLEAR_ICON, DIMENSIONS_AXIS_MAP, MODULE_NAME_DIMENSION } from '../../../common/constant';

const LineChart = () => {
  const trendData = useGlobalStore((state) => state.trendData);
  const stat = useGlobalStore((state) => state.stat); // 统计量
  const dimension = useGlobalStore((state) => state.dimension); // 维度
  const setTrendData = useGlobalStore((state) => state.setTrendData);

  const heatMapXAxisName = DIMENSIONS_AXIS_MAP[dimension || '']?.x;
  const heatMapYAxisName = DIMENSIONS_AXIS_MAP[dimension || '']?.y;
  let yAxisMap = new Map<string, string>();
  let xAxisData = trendData?.[0]?.dimensions;
  let xAxisName = dimension;
  if (dimension === MODULE_NAME_DIMENSION && !isEmpty(trendData)) {
    xAxisName = 'Target Id';
    trendData?.[0]?.dimensions.forEach((item, index) => {
      yAxisMap.set(String(index), item);
    });
    xAxisData = Array.from(yAxisMap.keys());
  }

  const option = {
    title: {
      left: 'center',
      top: 20,
      textStyle: {
        fontSize: 14,
        color: '#666',
        fontWeight: 'bold',
      },
    },
    legend: {
      data: trendData.map((trend) => {
        return `${heatMapXAxisName}: ${trend.dimX ?? ' '} / ${heatMapYAxisName}: ${trend.dimY ?? ' '}`;
      }),
      top: 10,
    },
    grid: {
      top: 80,
      bottom: 80,
      left: 120,
      right: 80,
    },
    tooltip: {
      trigger: 'axis',
    },
    toolbox: {
      right: 20,
      top: 10,
      feature: {
        myExport: {
          title: '清空',
          icon: CLEAR_ICON,
          onclick: () => {
            setTrendData([]);
          },
        },
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
      nameGap: 45,
      nameLocation: 'center',
    },
    yAxis: {
      type: 'value',
      name: stat,
      axisLabel: {
        fontSize: 12,
        fontWeight: 'bold',
        formatter: (value) => {
          if (value === 0) return '0';
          const abs = Math.abs(value);
          // 使用科学计数法的阈值
          if (abs < 1e-4 || abs >= 1e4) {
            let s = value.toExponential(6); // 高精度转字符串
            s = s.replace(/\.?0+e/, 'e'); // 移除 . 和末尾 0
            s = s.replace(/e\+?/, 'e'); // e+5 → e5
            return s;
          } else {
            // 普通数字：去尾零
            let str = value.toFixed(10); // 防止 0.1 + 0.2 问题
            str = str.replace(/\.?0+$/, ''); // 移除小数点及后面所有 0
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
    dataZoom: [
      {
        type: 'slider',
        show: true,
        startValue: 0,
        endValue: 500,
        bottom: 20,
        realtime: false,
      },
      {
        type: 'inside',
        xAxisIndex: 0,
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
      },
      {
        type: 'inside',
        yAxisIndex: 0,
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
      },
      {
        type: 'slider',
        show: true,
        yAxisIndex: 0,
        filterMode: 'empty',
        handleSize: 8,
        showDataShadow: false,
        left: '20',
        start: 0,
        end: 100,
      },
    ],
    series: trendData.map((trend) => {
      return {
        name: `${heatMapXAxisName}: ${trend.dimX ?? ' '} / ${heatMapYAxisName}: ${trend.dimY ?? ' '}`,
        type: 'line',
        data: trend?.values,
        lineStyle: {
          width: 2,
        },
        progressive: 1000,
        animation: true,
      };
    }),
    animation: true,
    animationDuration: 1000,
    animationEasing: 'cubicInOut',
  };

  return (
    <div className="LineChart" style={{ height: '100%' }}>
      <ReactECharts option={option} style={{ height: '100%' }} notMerge={true} />
    </div>
  );
};

export default memo(LineChart);
