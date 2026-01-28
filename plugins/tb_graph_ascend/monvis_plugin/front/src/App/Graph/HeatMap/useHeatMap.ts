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

import { DIMENSIONS_AXIS_MAP, CONTINUOUS } from '../../../common/constant';
import { formatSegmentLabel } from '../../../utils';
import request from '../../../utils/request';
import type { TrendRequestParams, TrendResponseData } from '../type';

const useHeatMap = () => {
  const loadGraphData = async (params: TrendRequestParams): Promise<TrendResponseData> => {
    try {
      const result = await request({
        url: 'trend',
        method: 'POST',
        data: params,
      });
      return result;
    } catch (error) {
      return {
        error: '网络异常：获取维度值列表失败',
      };
    }
  };
  const calculate3SigmaPieces = (data) => {
    const values = data.map((item) => item[2]).filter((v) => !isNaN(v) && isFinite(v));
    if (values.length === 0) return [];

    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const stdDev = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);

    const segments = [
      { threshold: mean - 3 * stdDev, color: '#313695' },
      { threshold: mean - 2 * stdDev, color: '#4575b4' },
      { threshold: mean - stdDev, color: '#74add1' },
      // { threshold: mean, color: '#abd9e9' },
      { threshold: mean + stdDev, color: '#ffffbf' },
      { threshold: mean + 2 * stdDev, color: '#fdae61' },
      { threshold: mean + 3 * stdDev, color: '#f46d43' },
      { threshold: Infinity, color: '#d73027' },
    ];

    const pieces: Array<{
      min: number;
      max: number;
      label: string;
      color: string;
    }> = [];
    let prevThreshold = -Infinity;

    segments.forEach((segment) => {
      if (segment.threshold > prevThreshold) {
        pieces.push({
          min: prevThreshold,
          max: segment.threshold,
          label: formatSegmentLabel(prevThreshold, segment.threshold),
          color: segment.color,
        });
        prevThreshold = segment.threshold;
      }
    });

    return pieces;
  };
  const createVisualMapConfig = (data, mode, minValue, maxValue) => {
    const values = data.map((item) => item[2]).filter((v) => !isNaN(v) && isFinite(v));
    if (values.length === 0) return {};

    if (mode === CONTINUOUS) {
      return {
        type: 'continuous',
        min: minValue,
        max: maxValue,
        range: null,
        inRange: {
          color: [
            '#4575b4',
            '#74add1',
            '#abd9e9',
            '#e0f3f8',
            '#ffffbf',
            '#fee090',
            '#fdae61',
            '#f46d43',
            '#d73027',
            '#a50026',
          ],
        },
        orient: 'horizontal',
        left: 'center',
        textStyle: {
          color: '#666',
        },
        calculable: true,
        itemWidth: 20,
        itemHeight: 800,
        precision: 4,
        top: 20,
      };
    } else {
      // 分段模式
      return {
        type: 'piecewise',
        pieces: calculate3SigmaPieces(data),
        orient: 'horizontal',
        left: 'center',
        itemGap: 7,
        itemSymbol: 'rect',
        textStyle: {
          fontSize: 10,
          color: '#666',
        },
        inRange: {
          color: [],
        },
        top: 20,
      };
    }
  };

  const updateHeatMap = (data, dimension, heatMapType) => {
    const ModuleNameMap = new Map(); // ID -> ModuleName
    const yAxisMap = new Map(); // ID -> Index
    let heatMapChartData = data.map((entry) => {
      ModuleNameMap.set(entry[1][0], entry[1][1]);
      return [entry[0], entry[1][0], entry[2]];
    });

    // x轴和y轴的刻度
    const xAxisSet = new Set<number>();
    const yAxisData = Array.from(ModuleNameMap.keys()).sort((a: number, b: number) => a - b);
    yAxisData.forEach((id, index) => {
      yAxisMap.set(id, index);
    });
    heatMapChartData = heatMapChartData.map((entry) => {
      return [entry[0], yAxisMap.get(entry[1]), entry[2]];
    });
    // x轴和y轴的标签
    const xAxisName = DIMENSIONS_AXIS_MAP[dimension]?.x;
    const yAxisName = DIMENSIONS_AXIS_MAP[dimension]?.y;

    let minValue = Number.MAX_VALUE;
    let maxValue = Number.MIN_VALUE;
    heatMapChartData.forEach((entry) => {
      xAxisSet.add(entry[0]);
      minValue = Math.min(minValue, entry[2]);
      maxValue = Math.max(maxValue, entry[2]);
    });
    const xAxisData = Array.from(xAxisSet).sort((a: number, b: number) => a - b);

    // 配置项
    const option = {
      backgroundColor: '#fff',
      tooltip: {
        position: function (point, params, dom, rect, size) {
          let [x, y] = point;
          const { viewSize } = size; // [容器宽, 容器高]

          if (y > viewSize[1] * 0.7) {
            y = y - 135;
          }
          if (y <= viewSize[1] * 0.7) {
            y = y + 10;
          }
          // 示例：根据鼠标靠近右边界，自动左对齐
          if (x > viewSize[0] * 0.7) {
            x = x - 200;
          }
          if (x <= viewSize[0] * 0.7) {
            x = x + 10;
          }

          return [x, y];
        },
        formatter: (params) => {
          const xLabel = params.data?.[0];
          const yLabel = ModuleNameMap.get(yAxisData[params.data?.[1]]);
          return `
                  <div style="z-index=1000;">
                    <div style="font-weight:bold;margin-bottom:5px">${yAxisName}: ${yLabel}</div>
                    <div style="margin-bottom:5px">${xAxisName}: ${xLabel}</div>
                    <div>Value: <b>${params.data?.[2]?.toFixed(12)}</b></div>
                  </div>
                    `;
        },
        backgroundColor: 'rgba(50,50,50,0.7)',
        borderColor: '#333',
        textStyle: {
          color: '#fff',
          fontSize: 12,
        },
        extraCssText:
          'box-shadow: 0 0 10px rgba(0,0,0,0.3);border-radius:4px;padding:8px;max-width: 200px; white-space: normal; word-break: break-all;',
      },
      grid: {
        left: 120,
        right: 80, // 设置大一些，防止tooltip超出页面宽度，触发视图重新渲染
        top: 100,
        bottom: 70,
      },

      xAxis: {
        type: 'category',
        name: xAxisName,
        data: xAxisData,
        splitArea: {
          show: true,
        },
        axisLabel: {
          formatter: (value) => (value.length > 20 ? value.slice(0, 8) + '...' : value),
          rotate: 45,
          fontSize: 11,
          color: '#666',
          fontWeight: 'bold',
        },
        show: true,
        nameLocation: 'center',
        nameGap: 35,
        axisTick: {
          alignWithLabel: true,
        },
        nameTextStyle: {
          fontSize: 12,
          fontWeight: 'bold',
          color: '#444',
        },
      },
      yAxis: {
        type: 'category',
        name: yAxisName,
        data: yAxisData,
        offset: 4,
        show: true,
        nameLocation: 'end',
        nameGap: 20,
        axisLabel: {
          fontSize: 11,
          fontWeight: 'bold',
          color: '#666',
          formatter: (value) => {
            return value.length > 15 ? value.slice(0, 8) + '...' : value;
          },
        },
        nameTextStyle: {
          fontSize: 12,
          fontWeight: 'bold',
          color: '#444',
        },
      },
      visualMap: createVisualMapConfig(heatMapChartData, heatMapType, minValue, maxValue),
      dataZoom: [
        {
          type: 'slider',
          show: true,
          startValue: 0,
          start: 0,
          end: 30,
          realtime: false,
          filterMode: 'filter',
        },
        {
          type: 'slider',
          show: true,
          yAxisIndex: 0,
          filterMode: 'filter',
          handleSize: 8,
          realtime: false,
          left: '20',
          start: 0,
          end: 30,
        },
        {
          type: 'inside',
          xAxisIndex: 0,
          filterMode: 'filter',
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
        },
        {
          type: 'inside',
          yAxisIndex: 0,
          filterMode: 'filter',
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
        },
      ],
      series: [
        {
          name: 'Heatmap',
          type: 'heatmap',
          data: heatMapChartData,
          label: {
            show: false,
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)',
            },
          },
          progressive: 300,
          animation: true,
        },
      ],
      animation: true,
      animationDuration: 300,
      animationEasing: 'cubicInOut',
    };

    return { yAxisData, option };
  };

  return { updateHeatMap, loadGraphData };
};
export default useHeatMap;
