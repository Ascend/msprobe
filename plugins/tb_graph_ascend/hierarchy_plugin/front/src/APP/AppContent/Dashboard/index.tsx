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

import { Layout, Spin, Splitter } from 'antd';
import BoardSider from './BoardSider';
import BoardHeader from './BoardHeader';
import BoardContent from './BoardContent';
import useGraphStore from '../../../store/useGraphStore';
import { useEffect, useState } from 'react';
import styles from './index.module.less';
import { loadGraphData } from '../../../api/board';
import BorderFooter from './BoardFooter';

import { useTranslation } from 'react-i18next';
const { Header, Sider, Content, Footer } = Layout;

const Dashboard = () => {
  const { t } = useTranslation();
  const currentMetaDir = useGraphStore((state) => state.currentMetaDir);
  const currentMetaFile = useGraphStore((state) => state.currentMetaFile);
  const currentMetaFileType = useGraphStore((state) => state.currentMetaFileType);
  const currentMetaStep = useGraphStore((state) => state.currentMetaStep);
  const currentMetaRank = useGraphStore((state) => state.currentMetaRank);
  const currentMetaData = useGraphStore((state) => state.getCurrentMetaData)();
  const fetchGraphConfig = useGraphStore((state) => state.fetchGraphConfig);
  //局部变量
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

  // 监听文件变化
  useEffect(() => {
    if (!currentMetaDir || !currentMetaFile || !currentMetaFileType) return;
    switch (currentMetaFileType) {
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

  return (
    <Spin spinning={loading} tip={loadingTip}>
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
