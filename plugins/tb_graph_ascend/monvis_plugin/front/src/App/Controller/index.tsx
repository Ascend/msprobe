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

import React, { useEffect, useState, memo } from 'react';
import { message, Tooltip } from 'antd';
import { isEmpty } from 'lodash';
import './index.less';
import { useGlobalStore } from '../../store/useGlobalStore';
import { DIMENSIONS_OPTIONS, HEATMAP_TYPE } from '../../common/constant';
import SelectWithLabel from '../../components/SelectWithLabel';
import useController from './useController';
import type { SelectOptionType } from '../../common/type';
import type { ControllerProps, ValuesResponseType, ValuesRequestParamsType, TagsResponseType } from './type';
import { Typography } from 'antd';
import { color } from 'echarts';
const { Text } = Typography;

const Controller: React.FC = (props: ControllerProps) => {
  const { metrics } = props;
  const useControllerInstance = useController();

  const metric = useGlobalStore((state) => state.metric); // 指标
  const stat = useGlobalStore((state) => state.stat); // 统计量
  const dimension = useGlobalStore((state) => state.dimension); // 维度
  const dimensionValue = useGlobalStore((state) => state.dimensionValue); // 维度值
  const heatMapType = useGlobalStore((state) => state.heatMapType); // 热力图类型
  const tags = useGlobalStore((state) => state.tags); // 标签

  const setDimX = useGlobalStore((state) => state.setDimX);
  const setDimY = useGlobalStore((state) => state.setDimY);
  const setStat = useGlobalStore((state) => state.setStat);
  const setTags = useGlobalStore((state) => state.setTags);
  const setMetric = useGlobalStore((state) => state.setMetric);
  const setDimension = useGlobalStore((state) => state.setDimension);
  const setDimensionValue = useGlobalStore((state) => state.setDimensionValue);
  const setHeatMapData = useGlobalStore((state) => state.setHeatMapData);
  const setTrendData = useGlobalStore((state) => state.setTrendData);
  const setLoadingHeatMap = useGlobalStore((state) => state.setLoadingHeatMap);
  const setHeatMapType = useGlobalStore((state) => state.setHeatMapType);

  const [metricsMapStats, setMetricsMapStats] = useState<Record<string, Array<SelectOptionType>>>({});
  const [metricsNameList, setMetricsNameList] = useState<string[]>([]);
  const [statNameList, setStatNameList] = useState<string[]>([]);
  const [dimensionValueList, setDimensionValueList] = useState<Array<SelectOptionType>>();
  const [tagsValueList, setTagsValueList] = useState<Array<SelectOptionType>>();

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

      const selectMetric = metricsNameList?.[0]?.value;
      const statNameList: Array<SelectOptionType> = metricsMapStats[selectMetric];
      const selectedStat = statNameList?.[0]?.value;
      setMetricsMapStats(metricsMapStats);
      // 初始化指标
      setMetricsNameList(metricsNameList);
      // 初始化统计量
      setStatNameList(statNameList);
      // 初始化指标和统计量
      setMetric(selectMetric);
      setStat(selectedStat);
      // 初始化维度,默认选中step维度
      const params: ValuesRequestParamsType = {
        metric: selectMetric,
        stat: selectedStat,
        dimension: DIMENSIONS_OPTIONS[0].value,
        tags,
      };
      updateDimensionValueList(params);
      updateTagsValueList({ metric: selectMetric });
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
      tags,
    };
    const loadGraphData = async (params) => {
      setLoadingHeatMap(true);
      const result = await useControllerInstance.loadGraphData(params);
      setLoadingHeatMap(false);
      if (result.error) {
        message.error(result?.error);
        return;
      }
      setHeatMapData(result?.data);
      setTrendData([]);
      setDimX(' ');
      setDimY(' ');
    };
    loadGraphData(params);
  }, [metric, stat, dimension, dimensionValue, tags]);

  const updateDimensionValueList = async (params: ValuesRequestParamsType) => {
    const { metric, stat, dimension } = params;
    if (!metric || !stat || !dimension) {
      return;
    }
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
      setDimension(params.dimension);
      setDimensionValue(dimensionValueList?.[0]?.value);
      setDimensionValueList(dimensionValueList);
    }
  };

  const updateTagsValueList = async (params: { metric: string }) => {
    const { metric } = params;
    if (!metric) {
      return;
    }
    const result: TagsResponseType = await useControllerInstance.loadTagsValueList(params);
    if (result.error) {
      message.error(result.error);
      return;
    }
    if (!isEmpty(result)) {
      const tagsValueList = (result?.data || []).map(({ category, id, text }) => {
        return {
          value: id,
          label: (
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Tooltip title={text}>
                <Text ellipsis> {text}</Text>
              </Tooltip>
              <span style={{ color: '#a5a5a5' }}> {category}</span>
            </div>
          ),
        };
      });
      setTagsValueList(tagsValueList);
    }
  };

  // 指标选择
  const onSelectMetricChange = (value: string) => {
    const statNameList = metricsMapStats[value];
    const selectedStat = statNameList?.[0]?.value;
    setMetric(value);
    setStat(selectedStat);
    setStatNameList(statNameList);
  };

  // 统计量选择
  const onSelectStatChange = (value: string) => {
    setStat(value);
  };

  // 维度选择
  const onSelectDimensionChange = (value: string) => {
    const params: ValuesRequestParamsType = {
      metric,
      stat,
      dimension: value,
    } as ValuesRequestParamsType;
    updateDimensionValueList(params);
  };
  // 标签选择
  const onSelectTagsChange = (value: string[]) => {
    const params: ValuesRequestParamsType = {
      metric,
      stat,
      dimension,
      tags: value,
    };
    updateDimensionValueList(params);
    setTags(value);
  };
  return (
    <div className="wrapper">
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
          showSearch
          filterOption={(input, option) => (option?.label ?? '').toLowerCase().includes(input.toLowerCase())}
          onChange={(value: string) => {
            setDimensionValue(value);
          }}
          options={dimensionValueList}
        />
        <SelectWithLabel
          className="select-with-label"
          label="选择标签"
          text="选择标签"
          mode="multiple"
          value={tags}
          placeholder="请选择标签"
          showSearch
          filterOption={(input, option) => (option?.label ?? '').toLowerCase().includes(input.toLowerCase())}
          onChange={onSelectTagsChange}
          options={tagsValueList}
        />
        <SelectWithLabel
          className="select-with-label"
          label="热力图模式"
          text="选择热力图的渲染模式"
          value={heatMapType}
          placeholder="请选择值"
          onChange={(value) => setHeatMapType(value)}
          options={HEATMAP_TYPE}
        />
      </div>
    </div>
  );
};

export default memo(Controller);
