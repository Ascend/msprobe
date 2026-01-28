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

// AppSider.tsx
import { Button, Popover, Tooltip } from 'antd';
import {
  FileOutlined,
  AppstoreOutlined,
  NodeIndexOutlined,
  SearchOutlined,
  ApartmentOutlined,
  SunOutlined,
  TranslationOutlined,
} from '@ant-design/icons';

import MetaContent from './MetaContent';
import styles from './index.module.less';
import useGlobalStore from '../../store/useGlobalStore';
import { CURRENT_PAGE, CURRENT_TAB } from '../../common/constant';
import { useTranslation } from 'react-i18next';
import useGraphStore from '../../store/useGraphStore';
import { useEffect } from 'react';

interface AppSiderProps {
  toggleTheme: () => void;
  toggleLanguage: () => void;
}

const AppSider = ({ toggleTheme, toggleLanguage }: AppSiderProps) => {
  const { t } = useTranslation(); // 👈 使用翻译函数
  const { currentTab, setCurrentTab, setCurrentPage } = useGlobalStore();
  const isSingleGraph = useGraphStore((state) => state.isSingleGraph);
  useEffect(() => {
    if (isSingleGraph) {
      setCurrentTab(CURRENT_TAB.PRECISION_TAB);
      setCurrentPage(CURRENT_PAGE.DASHBOARD);
    }
  }, [isSingleGraph]);

  return (
    <div className={styles.siderContainer}>
      <Tooltip placement="right" title={t('sider.dataSelection')}>
        <Popover placement="right" content={MetaContent} trigger="click">
          <Button
            className={`${styles.siderButton} ${currentTab === CURRENT_TAB.FILE_TAB ? styles.activeTab : ''}`}
            icon={<FileOutlined />}
            variant="text"
          />
        </Popover>
      </Tooltip>

      <Tooltip placement="right" title={t('sider.precisionFiltering')}>
        <Button
          className={`${styles.siderButton} ${currentTab === CURRENT_TAB.PRECISION_TAB ? styles.activeTab : ''}`}
          icon={<AppstoreOutlined />}
          data-testid="precisionSiderButton"
          variant="text"
          onClick={() => {
            setCurrentTab(CURRENT_TAB.PRECISION_TAB);
            setCurrentPage(CURRENT_PAGE.DASHBOARD);
          }}
        />
      </Tooltip>
      {!isSingleGraph && (
        <Tooltip placement="right" title={t('sider.nodeMatching')}>
          <Button
            className={styles.siderButton + ' ' + (currentTab === CURRENT_TAB.MATCH_TAB ? styles.activeTab : '')}
            icon={<NodeIndexOutlined />}
            data-testid="matchSiderButton"
            variant="text"
            onClick={() => {
              setCurrentTab(CURRENT_TAB.MATCH_TAB);
              setCurrentPage(CURRENT_PAGE.DASHBOARD);
            }}
          />
        </Tooltip>
      )}

      <Tooltip placement="right" title={t('sider.nodeSearch')}>
        <Button
          className={`${styles.siderButton} ${currentTab === CURRENT_TAB.SEARCH_TAB ? styles.activeTab : ''}`}
          icon={<SearchOutlined />}
          data-testid="searchSiderButton"
          variant="text"
          onClick={() => {
            setCurrentTab(CURRENT_TAB.SEARCH_TAB);
            setCurrentPage(CURRENT_PAGE.DASHBOARD);
          }}
        />
      </Tooltip>

      <Tooltip placement="right" title={t('sider.dumpVisualization')}>
        <Button
          className={`${styles.siderButton} ${currentTab === CURRENT_TAB.VISUALIZED_TAB ? styles.activeTab : ''}`}
          icon={<ApartmentOutlined />}
          data-testid="conversionSiderButton"
          variant="text"
          onClick={() => {
            setCurrentTab(CURRENT_TAB.VISUALIZED_TAB);
            setCurrentPage(CURRENT_PAGE.VISUALIZATION);
          }}
        />
      </Tooltip>
      <Button
        className={styles.siderButton}
        data-testid="themeSiderButton"
        shape="circle"
        onClick={toggleTheme}
        variant="text"
      >
        <SunOutlined />
      </Button>

      <Tooltip placement="right" title={t('sider.switchLanguage')}>
        <Button className={styles.siderButton} shape="circle" onClick={toggleLanguage} variant="text">
          <TranslationOutlined />
        </Button>
      </Tooltip>
    </div>
  );
};

export default AppSider;
