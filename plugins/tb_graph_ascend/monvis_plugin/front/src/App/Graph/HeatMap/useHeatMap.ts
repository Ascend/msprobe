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
import { escapeHTML, formatSegmentLabel } from '../../../utils';
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
    let heatMapChartData = data.map((entry) => {
      // 数据格式校验
      if (entry?.length !== 3 || entry[1]?.length !== 2) {
        return [];
      }
      ModuleNameMap.set(String(entry[1][0]), entry[1][1]);
      // 转化为字符串，如果是值的话，echarts默认为索引
      return [String(entry[0]), String(entry[1][0]), entry[2]];
    });

    const xAxisName = DIMENSIONS_AXIS_MAP[dimension]?.x;
    const yAxisName = DIMENSIONS_AXIS_MAP[dimension]?.y;

    const xAxisSet = new Set<number>();
    let minValue = Number.MAX_VALUE;
    let maxValue = Number.MIN_VALUE;
    heatMapChartData.forEach((entry) => {
      xAxisSet.add(Number(entry[0]));
      minValue = Math.min(minValue, entry[2]);
      maxValue = Math.max(maxValue, entry[2]);
    });
    // x轴和y轴的刻度，转化为字符串，如果是值的话，echarts默认为索引
    const xAxisData = Array.from(xAxisSet)
      .sort((a: number, b: number) => a - b)
      .map((x) => x.toString());
    const yAxisData = Array.from(ModuleNameMap.keys())
      .sort((a: number, b: number) => a - b)
      .map((x) => x.toString());

    // 配置项
    const option = {
      backgroundColor: '#fff',
      tooltip: {
        formatter: (params) => {
          const xLabel = params.data?.[0];
          const yLabel = ModuleNameMap.get(params.data?.[1]);
          const value = params.data?.[2];
          return `
            <div style="font-size:12px; line-height:1.6;">
              <div style="font-weight:bold; margin-bottom:6px;">${escapeHTML(yAxisName)}：${escapeHTML(yLabel)}</div>
              <div style="margin-bottom:6px;">${escapeHTML(xAxisName)}：${escapeHTML(xLabel)}</div>
              <div>Value：
                  <span style="font-weight:bold">
                    <span style="display:inline-block;width:10px;height:10px;background-color:${params.color};border-radius:2px;margin-right:6px;"></span>
                    ${escapeHTML(String(value))}
                  </span>
              </div>
            </div>
          `;
        },
        backgroundColor: '#fff',
        borderColor: '#ccc',
        borderWidth: 1,
        textStyle: {
          color: '#333',
          fontSize: 12
        },
        extraCssText: 'border-radius:4px; padding:10px 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);max-width: 300px; white-space: normal; word-break: break-all;',
        confine: true
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

    return { option };
  };

  return { updateHeatMap, loadGraphData };
};
export default useHeatMap;
