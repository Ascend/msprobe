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

import request from '../../utils/request';
import { ValuesResponseType, ValuesRequestParamsType, HeatmapDataResponseType, TagsResponseType } from './type';

const useController = () => {
  const loadGraphData = async (params: ValuesRequestParamsType) => {
    try {
      const result = (await request({
        url: 'heatmap_data',
        method: 'POST',
        data: params,
      })) as unknown as HeatmapDataResponseType;
      return result;
    } catch (error) {
      return {
        error: '网络异常：获取维度值列表失败',
      };
    }
  };
  const loadDimensionValueList = async (params: ValuesRequestParamsType) => {
    if (!params.dimension || !params.metric || !params.stat) {
      return {};
    }
    try {
      const result: ValuesResponseType = (await request({
        url: 'values',
        method: 'POST',
        data: params,
      })) as unknown as ValuesResponseType;
      return result;
    } catch (error) {
      return {
        error: '网络异常：获取维度值列表失败',
      };
    }
  };

  const loadTagsValueList = async (params: { metric: string }) => {
    if (!params.metric) {
      return {};
    }
    try {
      const result: TagsResponseType = (await request({
        url: 'tags',
        method: 'GET',
        params,
      })) as unknown as TagsResponseType;
      return result;
    } catch (error) {
      return {
        error: '网络异常：获取标签值列表失败',
      };
    }
  };

  return {
    loadGraphData,
    loadTagsValueList,
    loadDimensionValueList,
  };
};

export default useController;
