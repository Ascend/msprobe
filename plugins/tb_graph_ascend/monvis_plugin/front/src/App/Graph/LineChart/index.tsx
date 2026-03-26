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
import ReactECharts from 'echarts-for-react';

import { useGlobalStore } from '../../../store/useGlobalStore';
import { CLEAR_ICON, DIMENSIONS_AXIS_MAP, MODULE_NAME_DIMENSION } from '../../../common/constant';
import { escapeHTML } from '../../../utils';

const LineChart = () => {
  const trendData = useGlobalStore((state) => state.trendData);
  const stat = useGlobalStore((state) => state.stat); // 统计量
  const dimension = useGlobalStore((state) => state.dimension); // 维度
  const setTrendData = useGlobalStore((state) => state.setTrendData);

  const heatMapXAxisName = DIMENSIONS_AXIS_MAP[dimension || '']?.x;
  const heatMapYAxisName = DIMENSIONS_AXIS_MAP[dimension || '']?.y;
  let xAxisName = dimension;

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
      backgroundColor: '#fff',
      borderColor: '#ccc',
      borderWidth: 1,
      textStyle: {
        color: '#333',
        fontSize: 12
      },
      padding: [8, 12],
      formatter: (params) => {
        if (!params || params.length === 0) return '';
        // 取第一个点的模块信息（只显示一次）
        const first = params[0];
        const extra = first.data?.extra;
        let html = `<div style="font-weight:bold;margin-bottom:6px">${escapeHTML(extra)}</div>`;
        // 官方样式列表：每条线一个颜色标记
        params.forEach(item => {
          const value = item.data?.value?.[1] || 0;
          html += `
            <div style="display:flex;align-items:center;margin-top:3px;">
              <span style="display:inline-block;width:10px;height:10px;background-color:${item.color};border-radius:2px;margin-right:6px;"></span>
              <span style="flex:1">${escapeHTML(item.seriesName)}</span>
              <span style="font-weight:bold;margin-left:8px">${escapeHTML(value)}</span>
            </div>
          `;
        });

        return html;
      }
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
      type: 'value',
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
        data: trend?.values.map((value, index) => {
          const dimValue = dimension === MODULE_NAME_DIMENSION ? String(trend.dimensions[index])?.split('_')?.[0] || '' : String(trend.dimensions[index]);
          if (Number(value) > 100000000) value = '100000000';
          if (Number(value) < -100000000) value = '-100000000';
          return {
            value: [Number(dimValue), Number(value)],
            extra: trend.dimensions[index], // 原始数据绑进去
          }

        }),
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
