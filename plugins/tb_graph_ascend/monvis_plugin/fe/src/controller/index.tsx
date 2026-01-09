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

import React, { useEffect, useState, memo } from 'react';
import { shallow, useShallow } from 'zustand/shallow';
import { message } from 'antd';
import { isEmpty } from 'lodash';
import './index.less';
import { useGlobalStore } from '../store';
import { DIMENSIONS_OPTIONS, HEATMAP_TYPE } from '../common/constant';
import SelectWithLabel from '../common/SelectWithLabel';
import useController from './useController';
import type { SelectOptionType } from '../common/type';
import type { ControllerProps, ValuesResponseType, ValuesRequestParamsType } from './type';

const Controller: React.FC = (props: ControllerProps) => {
  const { metrics } = props;
  const useControllerInstance = useController();
  const { metric, stat, dimension, dimensionValue, heatMapType, setContextState } = useGlobalStore(
    useShallow((state) => ({
      metric: state.metric,
      stat: state.stat,
      dimension: state.dimension,
      dimensionValue: state.dimensionValue,
      heatMapType: state.heatMapType,
      setContextState: state.setContextState,
    })),
    shallow,
  );
  const [metricsMapStats, setMetricsMapStats] = useState<Record<string, Array<SelectOptionType>>>({});
  const [metricsNameList, setMetricsNameList] = useState<string[]>([]);
  const [statNameList, setStatNameList] = useState<string[]>([]);
  const [dimensionValueList, setDimensionValueList] = useState<string[]>();

  useEffect(() => {
    if (metrics) {
      const metricsMapStats: Record<string, Array<SelectOptionType>> = {};
      const metricsNameList: Array<SelectOptionType> = [];
      metrics.forEach(({ name, stats }) => {
        metricsMapStats[name] = stats.map((stat) => {
          return {
            label: stat,
            value: stat,
          };
        });
        metricsNameList.push({ label: name, value: name });
      });

      const selecrMetric = metricsNameList?.[0]?.value;
      const statNameList: Array<SelectOptionType> = metricsMapStats[selecrMetric];
      const selectedStat = statNameList?.[0]?.value;
      setMetricsMapStats(metricsMapStats);
      // 初始化指标
      setMetricsNameList(metricsNameList);
      // 初始化统计量
      setStatNameList(statNameList);
      setContextState({ metric: selecrMetric, stat: selectedStat });
      // 初始化维度,默认选中step维度
      const params: ValuesRequestParamsType = {
        metric: selecrMetric,
        stat: selectedStat,
        dimension: DIMENSIONS_OPTIONS[0].value,
        value: 0,
      };
      updateDimensionValueList(params);
    }
  }, [metrics]);

  useEffect(() => {
    if (!metric || !stat || !dimension || !dimensionValue) {
      return;
    }
    const params = {
      metric,
      stat,
      dimension,
      value: dimensionValue,
    };
    const loadGraphData = async (params) => {
      setContextState({ loadingHeatMap: true });
      const result = await useControllerInstance.loadGraphData(params);
      setContextState({ loadingHeatMap: false });
      if (result.error) {
        message.error(result?.error);
        return;
      }
      setContextState({
        heatMaphData: result?.data,
        trendData: { dimensions: [], values: [] },
        dimX: ' ',
        dimY: ' ',
      });
    };
    loadGraphData(params);
  }, [metric, stat, dimension, dimensionValue]);

  const updateDimensionValueList = async (params: ValuesRequestParamsType) => {
    const result: ValuesResponseType = await useControllerInstance.loadDimensionValueList(params);
    if (result.error) {
      message.error(result.error);
      return;
    }
    if (!isEmpty(result)) {
      const dimensionValueList = Object.entries(result?.data || {}).map(([key, value]) => {
        return {
          value: key,
          label: value,
        };
      });
      setContextState({
        dimension: params.dimension,
        dimensionValue: dimensionValueList?.[0]?.value,
      });
      setDimensionValueList(dimensionValueList);
    }
  };

  // 指标选择
  const onSelectMetricChange = (value: string) => {
    const statNameList = metricsMapStats[value];
    const selectedStat = statNameList?.[0]?.value;
    setContextState({ metric: value, stat: selectedStat });
    setStatNameList(statNameList);
  };

  // 统计量选择
  const onSelectStatChange = (value: string) => {
    setContextState({ stat: value });
  };

  // 维度选择
  const onSelectDimensionChange = (value: string) => {
    const params: ValuesRequestParamsType = {
      metric,
      stat,
      dimension: value,
      value: 0,
    } as ValuesRequestParamsType;
    updateDimensionValueList(params);
  };
  // 维度值选择
  const onSelectDimensionValueChange = (value: string) => {
    setContextState({ dimensionValue: value });
  };

  return (
    <div className="warpper">
      <div className="title">设置</div>
      <div className="controller">
        <SelectWithLabel
          className="select-with-label"
          value={metric}
          label="选择指标"
          text="选择要分析的模型指标"
          onChange={onSelectMetricChange}
          options={metricsNameList}
        />
        <SelectWithLabel
          className="select-with-label"
          label="选择统计量"
          text="选择要计算的统计量"
          value={stat}
          onChange={onSelectStatChange}
          options={statNameList}
        />
        <SelectWithLabel
          className="select-with-label"
          placeholder="请选择维度"
          label="选择维度"
          text="选择分析的维度"
          value={dimension}
          onChange={onSelectDimensionChange}
          options={DIMENSIONS_OPTIONS}
        />
        <SelectWithLabel
          className="select-with-label"
          label="选择值"
          text="选择维度的具体值"
          value={dimensionValue}
          placeholder="请选择值"
          onChange={onSelectDimensionValueChange}
          options={dimensionValueList}
        />
        <SelectWithLabel
          className="select-with-label"
          label="热力图模式"
          text="选择热力图的渲染模式"
          value={heatMapType}
          placeholder="请选择值"
          onChange={(value) =>
            setContextState({
              heatMapType: value,
            })
          }
          options={HEATMAP_TYPE}
        />
      </div>
    </div>
  );
};

export default memo(Controller);
