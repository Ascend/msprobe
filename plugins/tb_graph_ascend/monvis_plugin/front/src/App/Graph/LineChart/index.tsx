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

import React, { memo, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { Empty } from 'antd';

import { useGlobalStore } from '../../../store/useGlobalStore';
import { DIFF_LEGEND_ICON } from '../../../common/constant';
import { buildLineSeriesData, escapeHTML, formatCurveLabel, formatDiffLabel } from '../../../utils';
import type { TreadDataType } from '../../../common/type';
import useLegendInteractions, { getLegendSelected } from '../useLegendInteractions';

import './index.less';

interface DiffCurvePoint {
  value: [number, number];
  extra: string;
}

interface DiffResult {
  id: string;
  labelA: string;
  labelB: string;
  data: DiffCurvePoint[];
}

/** 根据曲线标识 "dimX_dimY" 查找 trendData */
function findCurve(trendData: Array<TreadDataType>, key: string): TreadDataType | undefined {
  return trendData.find((t) => `${t.dimX}_${t.dimY}` === key);
}

/** 悬浮框 HTML：首行显示维度信息，下面每条线颜色标记 + 值 */
function buildTooltipHtml(params: any): string {
  if (!params || params.length === 0) return '';
  const first = params[0];
  const extra = first.data?.extra;
  let html = `<div style="font-weight:bold;margin-bottom:6px">${escapeHTML(extra)}</div>`;
  params.forEach((item: any) => {
    const value = item.data?.value?.[1] ?? 0;
    html += `
      <div style="display:flex;align-items:center;margin-top:3px;">
        <span style="display:inline-block;width:10px;height:10px;background-color:${item.color};border-radius:2px;margin-right:6px;"></span>
        <span style="flex:1">${escapeHTML(item.seriesName)}</span>
        <span style="font-weight:bold;margin-left:8px">${escapeHTML(String(value))}</span>
      </div>
    `;
  });
  return html;
}

/** y 轴刻度格式化：小值保留有效位，大值用科学计数法 */
function formatAxisLabel(value: number): string {
  if (value === 0) return '0';
  const abs = Math.abs(value);
  if (abs < 1e-4 || abs >= 1e4) {
    let s = value.toExponential(6);
    s = s.replace(/\.?0+e/, 'e');
    s = s.replace(/e\+?/, 'e');
    return s;
  } else {
    let str = value.toFixed(10);
    str = str.replace(/\.?0+$/, '');
    return str;
  }
}

/** 计算两条曲线的 diff */
function computeDiff(
  curveA: TreadDataType,
  curveB: TreadDataType,
  dimension: string,
): DiffCurvePoint[] {
  // 建立 B 曲线的 dimension→value 映射
  const bMap = new Map<string, number>();
  for (let i = 0; i < curveB.dimensions.length; i++) {
    bMap.set(curveB.dimensions[i], Number(curveB.values[i]));
  }

  const result: DiffCurvePoint[] = [];
  for (let i = 0; i < curveA.dimensions.length; i++) {
    const dim = curveA.dimensions[i];
    const valA = Number(curveA.values[i]);
    const valB = bMap.get(dim);
    if (valB === undefined) continue; // B 没有该 x 值则跳过

    const dimValue = parseFloat(dim);
    if (isNaN(dimValue)) continue;

    const diffVal = valA - valB;
    // clamp 和原折线图一致
    const clamped = Math.max(-100000000, Math.min(100000000, diffVal));

    result.push({
      value: [dimValue, clamped],
      extra: dim,
    });
  }
  return result;
}

/**
 * 趋势折线图（三种视图模式合一，替代原 LineChart + LineDiffChart 两份组件）：
 * - original 原始模式：只画原始曲线，单 y 轴
 * - all      叠加模式：原始曲线 + diff 差值同图（双 y 轴，右轴 Δ）
 * - diff     差值模式：只画差值曲线（单 y 轴 Δ）
 * 命名统一 formatCurveLabel / formatDiffLabel；图例单击隐藏/双击删除、A/B 下拉
 * 添加差值各模式共用。模式切换不再卸载/重挂 echarts，窗口/缩放跨模式保持。
 */
const LineChart = () => {
  const trendData = useGlobalStore((state) => state.trendData);
  const stat = useGlobalStore((state) => state.stat); // 统计量
  const dimension = useGlobalStore((state) => state.dimension); // 维度
  const viewMode = useGlobalStore((state) => state.viewMode); // 折线图模式：original/all/diff
  const diffPairs = useGlobalStore((state) => state.diffPairs);
  const { onLegendSelectChanged } = useLegendInteractions();

  // 计算所有 diff 结果
  const diffResults: DiffResult[] = useMemo(() => {
    return diffPairs
      .map((pair) => {
        const curveA = findCurve(trendData, pair.curveA);
        const curveB = findCurve(trendData, pair.curveB);
        if (!curveA || !curveB) return null;
        return {
          id: pair.id,
          labelA: formatCurveLabel(curveA, dimension),
          labelB: formatCurveLabel(curveB, dimension),
          data: computeDiff(curveA, curveB, dimension),
        };
      })
      .filter(Boolean) as DiffResult[];
  }, [diffPairs, trendData, dimension]);

  // ECharts option
  const option = useMemo(() => {
    // diff 差值曲线专用色板：中等明度的饱和色，与原始曲线的默认亮色板（#5470c6 #91cc75 #fac858…）
    // 明度/色相错开；配合 dashed 线型 + 图例虚线段图标，图上/图例双通道与原始曲线区分
    const colors = [
      '#d32f2f', '#7b1fa2', '#00838f', '#303f9f', '#ef6c00',
      '#8d6e63', '#0097a7', '#afb42b', '#c2185b', '#546e7a',
    ];
    // 差值曲线配色按 pair id 稳定哈希（非按下标取色）：删除/新增某条 pair 不影响
    // 其他差值曲线颜色，同一对 A/B 无论选择顺序调换都稳定同色
    const hashPairColor = (id: string) => {
      let h = 0;
      for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
      return colors[h % colors.length];
    };
    const axisLabel = {
      fontSize: 12,
      fontWeight: 'bold',
      formatter: (value: number) => formatAxisLabel(value),
    };
    const axisNameStyle = { fontSize: 11, fontWeight: 'bold', color: '#444' };

    const isDiffMode = viewMode === 'diff';
    const isAllMode = viewMode === 'all';

    // 原始曲线系列（所有模式共用构建，命名统一 formatCurveLabel）
    const rawSeries = trendData.map((trend) => ({
      name: formatCurveLabel(trend, dimension),
      type: 'line',
      data: buildLineSeriesData(trend, dimension),
      lineStyle: { width: 2 },
      progressive: 1000,
      animation: true,
    }));

    // diff 差值曲线系列（all 模式挂右 y 轴 Δ）
    const diffSeries = diffResults.map((r) => ({
      name: formatDiffLabel(r.labelA, r.labelB),
      type: 'line',
      data: r.data,
      yAxisIndex: isAllMode ? 1 : 0,
      lineStyle: {
        width: 2,
        color: hashPairColor(r.id),
        type: 'dashed', // 差值曲线一律虚线，与原始曲线(实线)视觉区分
      },
      itemStyle: {
        color: hashPairColor(r.id),
      },
      showSymbol: false, // 数据点密集时默认空心圆符号会互相重叠把虚线空隙填实，必须关掉
      animation: true,
    }));

    // 系列组装：按模式取 raw/diff（保持各模式原有绘制范围）
    const series = isDiffMode ? diffSeries : isAllMode ? [...rawSeries, ...diffSeries] : rawSeries;

    // 图例 data：原始项默认色块图标；diff 项自定义虚线段图标，图例上也能一眼区分
    const legendData = isDiffMode
      ? diffSeries.map((s) => ({ name: s.name, icon: DIFF_LEGEND_ICON }))
      : [
          ...rawSeries.map((s) => s.name),
          ...(isAllMode ? diffSeries.map((s) => ({ name: s.name, icon: DIFF_LEGEND_ICON })) : []),
        ];

    // y 轴：all 模式有 diff 时双轴（右轴 Δ stat），否则单轴
    const yAxis = (() => {
      if (isAllMode) {
        const axes: Array<any> = [
          {
            type: 'value',
            name: stat,
            axisLabel,
            nameTextStyle: axisNameStyle,
            splitLine: { lineStyle: { type: 'dashed', color: '#e8e8e8' } },
          },
        ];
        // 有 diff 才出右轴，避免空轴
        if (diffResults.length > 0) {
          axes.push({
            type: 'value',
            name: `Δ ${stat}`,
            axisLabel,
            nameTextStyle: axisNameStyle,
            splitLine: { show: false },
          });
        }
        return axes;
      }
      return {
        type: 'value',
        name: isDiffMode ? `Δ ${stat}` : stat,
        axisLabel,
        nameTextStyle: axisNameStyle,
        ...(isDiffMode ? { splitLine: { lineStyle: { type: 'dashed', color: '#e8e8e8' } } } : {}),
      };
    })();

    // 网格：三模式统一内边距（top 留出图例空间，bottom 给 x 轴滑杆）
    const grid = { top: 80, bottom: 60, left: 100, right: 60 };

    return {
      legend: {
        data: legendData,
        top: 10,
        selected: getLegendSelected(),
      },
      grid,
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#fff',
        borderColor: '#ccc',
        borderWidth: 1,
        textStyle: { color: '#333', fontSize: 12 },
        padding: [8, 12],
        formatter: (params: any) => buildTooltipHtml(params),
      },
      xAxis: {
        type: 'value',
        name: dimension,
        axisLabel: { rotate: 45, fontSize: 12, fontWeight: 'bold' },
        nameTextStyle: { fontSize: 12, fontWeight: 'bold', color: '#444' },
        nameGap: 45,
        nameLocation: 'center',
      },
      yAxis,
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
          yAxisIndex: isAllMode && diffResults.length > 0 ? [0, 1] : 0,
          filterMode: 'empty',
          handleSize: 8,
          showDataShadow: false,
          left: '20',
          start: 0,
          end: 100,
        },
      ],
      series,
      animation: true,
      animationDuration: 800,
      animationEasing: 'cubicInOut',
    };
  }, [viewMode, diffResults, trendData, dimension, stat]);

  const hasChart = viewMode === 'diff' ? diffResults.length > 0 : trendData.length > 0;

  return (
    <div className="LineChart" style={{ height: '100%' }}>
      {/* 图表区域 */}
      <div className="diff-chart-area" style={{ height: '100%', minHeight: 0 }}>
        {hasChart ? (
          <ReactECharts
            option={option}
            style={{ height: '100%' }}
            notMerge={true}
            onEvents={{ legendselectchanged: onLegendSelectChanged }}
          />
        ) : (
          <Empty
            description={viewMode === 'diff' ? '请选择两条曲线做差值对比' : '点击 heatmap 添加曲线'}
            style={{ paddingTop: 40 }}
          />
        )}
      </div>
    </div>
  );
};

export default memo(LineChart);
