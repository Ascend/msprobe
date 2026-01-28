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
import { Layout, Splitter } from 'antd';
import Controller from '../Controller';
import Graph from '../Graph';
import { message } from 'antd';
import './index.less';
import request from '../../utils/request';
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
      <Splitter layout="horizontal" style={{ height: '100%', boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)' }}>
        <Splitter.Panel style={{ height: '100%', width: '100%' }} defaultSize="20%" max="30%" min="10%">
          <Sider className="sider">
            <Controller metrics={metrics} />
          </Sider>
        </Splitter.Panel>
        <Splitter.Panel style={{ height: '100%' }} defaultSize="80%">
          <Content className="content">
            <Graph />
          </Content>
        </Splitter.Panel>
      </Splitter>
    </Layout>
  );
};

export default memo(MonVis);
