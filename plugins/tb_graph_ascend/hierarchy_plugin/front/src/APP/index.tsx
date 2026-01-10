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

import { Layout, ConfigProvider, theme, Modal } from 'antd';
import useTheme from './useTheme';
import AppContent from './AppContent';
import AppSider from './AppSider';
import { useEffect } from 'react';
import useGraphStore from '../store/useGraphStore';
import styles from './index.module.less';
import BorderSafeModal from '../components/BorderSafeModal';
import enUS from 'antd/locale/en_US';
import zhCN from 'antd/locale/zh_CN';
import dayjs from 'dayjs';
import 'dayjs/locale/zh-cn'; // 导入中文 locale

import '../common/i18n';
import { useTranslation } from 'react-i18next';

const { defaultAlgorithm, darkAlgorithm } = theme;
const { Sider, Content } = Layout;

const App = () => {
  const { i18n, t } = useTranslation();
  const [modal, contextHolder] = Modal.useModal();
  const { themeType, themeToken, toggleTheme } = useTheme();

  const fetchFileInfoList = useGraphStore((store) => store.fetchFileInfoList);
  const fileErrorList = useGraphStore((store) => store.fileErrorList);
  const setCurrentLang = useGraphStore((store) => store.setCurrentLang);

  // 同步 dayjs 语言（用于 DatePicker 等组件）
  useEffect(() => {
    if (i18n.language === 'zh') {
      dayjs.locale('zh-cn');
      setCurrentLang('zh');
    } else {
      dayjs.locale('en');
      setCurrentLang('en');
    }
  }, [i18n.language]);

  // 根据当前语言选择 Antd 的 locale
  const antdLocale = i18n.language === 'zh' ? zhCN : enUS;

  const toggleLanguage = () => {
    const newLang = i18n.language === 'zh' ? 'en' : 'zh';
    i18n.changeLanguage(newLang);
    setCurrentLang(newLang);
  };
  useEffect(() => {
    fetchFileInfoList(); // fetch metadata when the app is mounted
  }, []);

  useEffect(() => {
    if (fileErrorList.length > 0) {
      modal.warning({
        title: t('risk_warning'),
        content: <BorderSafeModal fileErrorList={fileErrorList} />,
        style: {
          width: 600,
        },
        width: 600,
        okText: t('risk_confirm'),
      });
    }
  }, [fileErrorList]);

  return (
    <ConfigProvider
      locale={antdLocale}
      theme={{
        algorithm: themeType === 'dark' ? darkAlgorithm : defaultAlgorithm,
        token: themeToken,
        cssVar: true,
        components: {
          Segmented: { algorithm: true },
        },
      }}
    >
      {contextHolder}
      <Layout>
        <Sider width={44} className={styles.sider}>
          <AppSider toggleTheme={toggleTheme} toggleLanguage={toggleLanguage} />
        </Sider>
        <Content className={styles.content}>
          <AppContent />
        </Content>
      </Layout>
    </ConfigProvider>
  );
};

export default App;
