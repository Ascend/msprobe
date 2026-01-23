/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { Layout, Spin, Splitter, message } from 'antd';
import BoardSider from './BoardSider';
import BoardHeader from './BoardHeader';
import BoardContent from './BoardContent';
import useGraphStore from '../../../store/useGraphStore';
import { useEffect, useRef, useState } from 'react';
import styles from './index.module.less';
import { loadGraphData } from '../../../api/board';
import { formatBytes, safeJSONParse } from '../../../common/utils';
import BorderFooter from './BoardFooter';

import { useTranslation } from 'react-i18next';
const { Header, Sider, Content, Footer } = Layout;

const Dashboard = () => {
  const { t } = useTranslation();

  const currentMetaFile = useGraphStore((state) => state.currentMetaFile);
  const currentMetaFileType = useGraphStore((state) => state.currentMetaFileType);
  const currentMetaStep = useGraphStore((state) => state.currentMetaStep);
  const currentMetaRank = useGraphStore((state) => state.currentMetaRank);
  const currentMetaData = useGraphStore((state) => state.getCurrentMetaData)();
  const fetchGraphConfig = useGraphStore((state) => state.fetchGraphConfig);
  const fetchAllNodeList = useGraphStore((state) => state.fetchAllNodeList);
  //局部变量
  // const [contextModalHolder] = Modal.useModal();
  const [messageApi, contextHolder] = message.useMessage();
  const eventSourceRef = useRef<EventSource | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingTip, setLoadingTip] = useState(t('dashboard.loading.default')); // 默认提示

  // 加载 DB 图数据
  const loadDBGraphData = async (isLoadConfig = true) => {
    setLoading(true);
    setLoadingTip(t('dashboard.loading.graphData'));
    await loadGraphData(currentMetaData);
    if (isLoadConfig) {
      setLoadingTip(t('dashboard.loading.graphConfig'));
      await fetchGraphConfig();
    }
    setLoading(false);
  };

  // 加载 JSON 图数据（带进度）
  const loadJSONGraphData = async () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    eventSourceRef.current = new EventSource(
      `/data/plugin/graph_ascend/loadGraphData?run=${currentMetaData.run}&tag=${currentMetaData.tag}&type=${currentMetaData.type}&lang=${currentMetaData.lang}`,
    );

    eventSourceRef.current.onmessage = (e: MessageEvent) => {
      const data = safeJSONParse(e.data);
      if (data?.error) {
        messageApi.error({
          content: data.error,
          key: 'load-error',
        });
        setLoading(false);
        eventSourceRef.current?.close();
        eventSourceRef.current = null;
        return;
      }

      if (data?.status === 'reading') {
        const progressValue = data.done ? 1 : data.progress / 100.0;
        const size = formatBytes(data.size);
        const read = formatBytes(data.read);
        const info = t('dashboard.loading.fileProgress', {
          size,
          read,
          percent: (progressValue * 100).toFixed(1),
        });
        setLoadingTip(info);
      }

      if (data?.status === 'loading') {
        if (data.done) {
          eventSourceRef.current?.close();
          eventSourceRef.current = null;
          setLoading(false);
          fetchGraphConfig().catch(() => {
            messageApi.error({
              content: t('dashboard.error.loadGraphConfigFailed'),
              key: 'config-error',
            });
          });
        } else {
          setLoadingTip(t('dashboard.loading.graphData'));
        }
      }
    };

    eventSourceRef.current.onerror = () => {
      messageApi.error({
        content: t('dashboard.error.loadGraphDataFailed'),
        key: 'sse-error',
      });
      setLoading(false);
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
    };
  };

  // 监听文件变化
  useEffect(() => {
    if (!currentMetaFile || !currentMetaFileType) return;
    switch (currentMetaFileType) {
      case 'json':
        loadJSONGraphData();
        break;
      case 'db':
        loadDBGraphData(true);
        break;
      default:
        break;
    }
  }, [currentMetaFile, currentMetaFileType]);

  // 监听 step/rank 变化（用于 DB 切换）
  useEffect(() => {
    if (!currentMetaFile || currentMetaStep == null || currentMetaRank == null) return;
    loadDBGraphData(false);
  }, [currentMetaStep, currentMetaRank]);

  useEffect(() => {
    if (currentMetaData.tag && currentMetaData.run) {
      fetchAllNodeList();
    }
  }, [JSON.stringify(currentMetaData)]);

  return (
    <Spin spinning={loading} tip={loadingTip}>
      {contextHolder}
      <Layout className={styles.dashboardLayout}>
        <Splitter>
          <Splitter.Panel
            collapsible={{ start: true, end: true, showCollapsibleIcon: true }}
            defaultSize="15%"
            min="10%"
            max="50%"
            className={styles.controlSiderPanel}
          >
            <Sider width="100%" className={styles.controlSider}>
              <BoardSider />
            </Sider>
          </Splitter.Panel>
          <Splitter.Panel className={styles.contentPanel}>
            <Layout className={styles.contentLayout}>
              <Header className={styles.header}>
                <BoardHeader />
              </Header>
              <Splitter layout="vertical">
                <Splitter.Panel defaultSize="70%" min="5%" max="90%">
                  <Content className={styles.content}>
                    <BoardContent />
                  </Content>
                </Splitter.Panel>
                <Splitter.Panel collapsible={{ start: true, end: true, showCollapsibleIcon: true }}>
                  <Footer className={styles.footer}>
                    <BorderFooter />
                  </Footer>
                </Splitter.Panel>
              </Splitter>
            </Layout>
          </Splitter.Panel>
        </Splitter>
      </Layout>
    </Spin>
  );
};

export default Dashboard;
