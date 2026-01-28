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

import React, { useRef, useEffect, useState, memo, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { message } from 'antd';
import { useGlobalStore } from '../../../store/useGlobalStore';
import useHeatMap from './useHeatMap';
import { isEmpty } from 'lodash';
import { TrendRequestParams, TrendResponseData } from '../type';

const HeatMap = () => {
  const useHeatMapInstance = useHeatMap();
  // 1. 从 store 读取状态
  const trendData = useGlobalStore((state) => state.trendData);
  const heatMapData = useGlobalStore((state) => state.heatMapData);
  const metric = useGlobalStore((state) => state.metric); // 指标
  const stat = useGlobalStore((state) => state.stat); // 统计量
  const dimension = useGlobalStore((state) => state.dimension); // 维度
  const dimensionValue = useGlobalStore((state) => state.dimensionValue); // 维度值
  const heatMapType = useGlobalStore((state) => state.heatMapType); // 热力图类型
  const tags = useGlobalStore((state) => state.tags); // 标签

  const setDimX = useGlobalStore((state) => state.setDimX);
  const setDimY = useGlobalStore((state) => state.setDimY);

  const setLoadingLineChart = useGlobalStore((state) => state.setLoadingLineChart);
  const setTrendData = useGlobalStore((state) => state.setTrendData);
  const [option, setOption] = useState({});
  const yAxisMapRef = useRef(new Map());

  // 2. 创建 ref 存储最新状态
  const latestPropsRef = useRef({
    metric,
    stat,
    dimension,
    dimensionValue,
    heatMapData,
    setDimX,
    setDimY,
    setTrendData,
    setLoadingLineChart,
  });

  // 3. 每次 render 更新 ref（不会触发重渲染）
  useEffect(() => {
    latestPropsRef.current = {
      metric,
      stat,
      dimension,
      heatMapData,
      dimensionValue,
      trendData,
      setDimX,
      setDimY,
      setTrendData,
      setLoadingLineChart,
    };
  });

  useEffect(() => {
    if (metric && stat && dimension) {
      const { yAxisData, option } = useHeatMapInstance.updateHeatMap(heatMapData, dimension, heatMapType);
      setOption(option);
      yAxisMapRef.current = yAxisData;
    }
  }, [heatMapData, heatMapType]);

  // 4. 使用 useCallback 固定函数引用，从 ref 读取最新值
  const onChartClick = useCallback(async (echartsParams) => {
    const dimX = echartsParams.data[0];
    const dimY = yAxisMapRef.current[echartsParams.data[1]];

    // 从 ref 拿最新值，避免 stale closure
    const { metric, stat, dimension, trendData, dimensionValue, setDimX, setDimY, setTrendData, setLoadingLineChart } =
      latestPropsRef.current;

    if (!metric || !stat || !dimension) {
      console.warn('Missing required props in onChartClick');
      return;
    }
    // 判断是否已经存在
    if (!isEmpty(trendData)) {
      const isExist = trendData.some((item) => item.dimX === dimX && item.dimY === dimY);
      if (isExist) {
        return;
      }
    }

    const params: TrendRequestParams = {
      metric,
      stat,
      dimension,
      value: dimensionValue,
      dimX,
      dimYIdx: dimY,
      tags,
    } as TrendRequestParams;

    setLoadingLineChart(true);
    try {
      const result: TrendResponseData = await useHeatMapInstance.loadGraphData(params);
      setLoadingLineChart(false);

      if (result?.error) {
        message.error(result.error);
        return;
      }
      const newTrendData = [
        ...(trendData || []),
        {
          dimX,
          dimY,
          ...result.data,
        },
      ];
      setTrendData(newTrendData);
      setDimX(dimX);
      setDimY(dimY);
    } catch (err) {
      setLoadingLineChart(false);
      message.error('Failed to load trend data');
    }
  }, []);

  const onDataRangeSelected = useCallback((params) => {
    const selectedRange = params.selected; // { visualMapId: [min, max] }
    if (!selectedRange || !Array.isArray(selectedRange)) return;

    const [minVal, maxVal] = selectedRange;
    const heatMapData = latestPropsRef.current.heatMapData; // 从 ref 拿最新值，避免 stale closure

    // 过滤数据：只保留 value 在 [minVal, maxVal] 之间的项
    const filteredData = heatMapData.filter((item) => {
      const value = item[2]; // 假设第3维是映射维度
      return value >= minVal && value <= maxVal;
    });
    const { yAxisData, option } = useHeatMapInstance.updateHeatMap(filteredData, dimension, heatMapType);
    setOption({
      xAis: option.xAxis,
      yAxis: option.yAxis,
      series: option.series,
    });
    yAxisMapRef.current = yAxisData;
  }, []);

  return (
    <div style={{ height: '100%', borderBottom: '2px solid #ccc' }}>
      <ReactECharts
        option={option}
        style={{ height: '100%' }}
        onEvents={{
          click: onChartClick,
          datarangeselected: onDataRangeSelected,
        }}
      />
    </div>
  );
};

export default memo(HeatMap);
