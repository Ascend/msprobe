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

interface AppSiderProps {
  toggleTheme: () => void;
  toggleLanguage: () => void;
}

const AppSider = ({ toggleTheme, toggleLanguage }: AppSiderProps) => {
  const { t } = useTranslation(); // 👈 使用翻译函数
  const { currentTab, setCurrentTab, setCurrentPage } = useGlobalStore();

  return (
    <div className={styles.siderContainer}>
      <Tooltip placement="left" title={t('sider.dataSelection')}>
        <Popover placement="left" content={MetaContent} trigger="click">
          <Button
            className={`${styles.siderButton} ${currentTab === CURRENT_TAB.FILE_TAB ? styles.activeTab : ''}`}
            icon={<FileOutlined />}
            variant="text"
          />
        </Popover>
      </Tooltip>

      <Tooltip placement="left" title={t('sider.precisionFiltering')}>
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

      <Tooltip placement="left" title={t('sider.nodeMatching')}>
        <Button
          className={`${styles.siderButton} ${currentTab === CURRENT_TAB.MATCH_TAB ? styles.activeTab : ''}`}
          icon={<NodeIndexOutlined />}
          data-testid="matchSiderButton"
          variant="text"
          onClick={() => {
            setCurrentTab(CURRENT_TAB.MATCH_TAB);
            setCurrentPage(CURRENT_PAGE.DASHBOARD);
          }}
        />
      </Tooltip>

      <Tooltip placement="left" title={t('sider.nodeSearch')}>
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

      <Tooltip placement="left" title={t('sider.dumpVisualization')}>
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

      <Tooltip placement="left" title={t('sider.switchLanguage')}>
        <Button className={styles.siderButton} shape="circle" onClick={toggleLanguage} variant="text">
          <TranslationOutlined />
        </Button>
      </Tooltip>
    </div>
  );
};

export default AppSider;
