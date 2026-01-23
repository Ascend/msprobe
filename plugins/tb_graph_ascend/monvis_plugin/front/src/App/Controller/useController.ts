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
