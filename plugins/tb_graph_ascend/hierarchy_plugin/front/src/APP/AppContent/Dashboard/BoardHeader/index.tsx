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
import { Button, Dropdown, type MenuProps, Typography } from 'antd';
import styles from './index.module.less';
import Legend from './Legend';
import { PicLeftOutlined, PicRightOutlined, SubnodeOutlined, ReadOutlined, CompressOutlined } from '@ant-design/icons';
import { ControlButton } from '../../../../common/constant';
import { useState } from 'react';
import useGraphStore from '../../../../store/useGraphStore';
import { useTranslation } from 'react-i18next';

const { Text } = Typography;
const BoardHeader = () => {
  const { t } = useTranslation();
  const isSingleGraph = useGraphStore((state) => state.isSingleGraph);
  const setIsShowNpuMiniMap = useGraphStore((state) => state.setIsShowNpuMiniMap);
  const setIsShowBenchMiniMap = useGraphStore((state) => state.setIsShowBenchMiniMap);
  const setIsSyncExpand = useGraphStore((state) => state.setIsSyncExpand);
  // 使用枚举作为 key 的状态对象
  const [activeButtons, setActiveButtons] = useState<Record<ControlButton, boolean>>({
    [ControlButton.NPU_MAP]: true,
    [ControlButton.BENCH_MAP]: true,
    [ControlButton.MATCH_SYNC]: true,
    [ControlButton.PROFILE]: false,
    [ControlButton.EXPAND]: false,
  });

  const items: MenuProps['items'] = [
    {
      label: (
        <div className={styles.shortcut}>
          <Text>{t('zoomIn')}</Text>
          <Text className={styles.value}>W</Text>
        </div>
      ),
      key: '0',
    },
    {
      label: (
        <div className={styles.shortcut}>
          <Text>{t('zoomOut')}</Text>
          <Text className={styles.value}>S</Text>
        </div>
      ),
      key: '1',
    },
    {
      label: (
        <div className={styles.shortcut}>
          <Text>{t('moveLeft')}</Text>
          <Text className={styles.value}>A</Text>
        </div>
      ),
      key: '2',
    },
    {
      label: (
        <div className={styles.shortcut}>
          <Text>{t('moveRight')}</Text> {/* 修正：原代码写成了“左移” */}
          <Text className={styles.value}>D</Text>
        </div>
      ),
      key: '3',
    },
    {
      label: (
        <div className={styles.shortcut}>
          <Text>{t('scrollUpDown')}</Text>
          <Text className={styles.value}>{t('scroll')}</Text>
        </div>
      ),
      key: '4',
    },
  ];

  const toggleActive = (type: ControlButton) => {
    const newActive = !activeButtons[type];
    switch (type) {
      case ControlButton.NPU_MAP:
        setIsShowNpuMiniMap(newActive);
        break;

      case ControlButton.BENCH_MAP:
        setIsShowBenchMiniMap(newActive);
        break;

      case ControlButton.MATCH_SYNC:
        setIsSyncExpand(newActive);
        break;

      case ControlButton.PROFILE:
        return;

      case ControlButton.EXPAND:
        const changeMatchNodeExpandState = new CustomEvent('fitScreen', {
          detail: {},
          bubbles: true, // 允许事件冒泡
          composed: true, // 允许跨 Shadow DOM 边界
        });
        document.dispatchEvent(changeMatchNodeExpandState);
        return;

      default:
        break;
    }
    setActiveButtons((prev) => ({
      ...prev,
      [type]: newActive,
    }));
  };

  return (
    <div className={styles.boardHeader}>
      <div className={styles.legend}>
        <Legend />
      </div>
      <div className={styles.controlMenu}>
        <Button
          key={ControlButton.NPU_MAP}
          type="text"
          className={`${styles.controlItem} ${activeButtons[ControlButton.NPU_MAP] ? styles.activeControlItem : ''}`}
          icon={<PicLeftOutlined />}
          onClick={() => toggleActive(ControlButton.NPU_MAP)}
        />
        {!isSingleGraph && (
          <Button
            key={ControlButton.BENCH_MAP}
            type="text"
            className={`${styles.controlItem} ${
              activeButtons[ControlButton.BENCH_MAP] ? styles.activeControlItem : ''
            }`}
            icon={<PicRightOutlined />}
            onClick={() => toggleActive(ControlButton.BENCH_MAP)}
          />
        )}
        {!isSingleGraph && (
          <Button
            key={ControlButton.MATCH_SYNC}
            type="text"
            className={`${styles.controlItem} ${
              activeButtons[ControlButton.MATCH_SYNC] ? styles.activeControlItem : ''
            }`}
            icon={<SubnodeOutlined />}
            onClick={() => toggleActive(ControlButton.MATCH_SYNC)}
          />
        )}
        <Dropdown menu={{ items }} trigger={['click']}>
          <Button
            key={ControlButton.PROFILE}
            type="text"
            className={`${styles.controlItem} ${activeButtons[ControlButton.PROFILE] ? styles.activeControlItem : ''}`}
            icon={<ReadOutlined />}
            onClick={() => toggleActive(ControlButton.PROFILE)}
          />
        </Dropdown>
        <Button
          key={ControlButton.EXPAND}
          type="text"
          style={{ fontSize: 14 }}
          className={`${styles.controlItem} ${activeButtons[ControlButton.EXPAND] ? styles.activeControlItem : ''}`}
          icon={<CompressOutlined />}
          onClick={() => toggleActive(ControlButton.EXPAND)}
        />
      </div>
    </div>
  );
};
export default BoardHeader;
