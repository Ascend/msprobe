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
import { Layout } from 'antd';
import Controller from '../controller';
import Graph from '../graph';
import { message } from 'antd';
import './index.less';
import request from '../utils/request';
import { isEmpty } from 'lodash';
import type { MetricsResponseType } from './type';
const { Sider, Content } = Layout;

const MonVis = () => {
  const [metrics, setMetrics] = useState<MetricsResponseType>([]);

  // API获取指标信息
  const loadIndicatorsInfo = async () => {
    try {
      const { data } = (await request({ url: 'metrics', method: 'GET' })) as unknown as MetricsResponseType;
      if (!isEmpty(data)) {
        setMetrics(data);
      }
    } catch (error) {
      message.error('网络异常：获取指标信息失败');
    }
  };

  useEffect(() => {
    loadIndicatorsInfo();
  }, []);

  return (
    <Layout style={{ height: '100vh' }}>
      <Sider width="360" className="sider">
        <Controller metrics={metrics} />
      </Sider>
      <Content className="content">
        <Graph />
      </Content>
    </Layout>
  );
};

export default memo(MonVis);
