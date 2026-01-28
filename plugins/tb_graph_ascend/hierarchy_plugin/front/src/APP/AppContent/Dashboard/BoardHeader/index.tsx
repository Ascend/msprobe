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
import { Button, Dropdown, type MenuProps, Tooltip, Typography } from 'antd';
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
          <Text>{t('boardHeader.shortcuts.zoomIn')}</Text>
          <Text className={styles.value}>W</Text>
        </div>
      ),
      key: '0',
    },
    {
      label: (
        <div className={styles.shortcut}>
          <Text>{t('boardHeader.shortcuts.zoomOut')}</Text>
          <Text className={styles.value}>S</Text>
        </div>
      ),
      key: '1',
    },
    {
      label: (
        <div className={styles.shortcut}>
          <Text>{t('boardHeader.shortcuts.moveLeft')}</Text>
          <Text className={styles.value}>A</Text>
        </div>
      ),
      key: '2',
    },
    {
      label: (
        <div className={styles.shortcut}>
          <Text>{t('boardHeader.shortcuts.moveRight')}</Text>
          <Text className={styles.value}>D</Text>
        </div>
      ),
      key: '3',
    },
    {
      label: (
        <div className={styles.shortcut}>
          <Text>{t('boardHeader.shortcuts.scrollUpDown')}</Text>
          <Text className={styles.value}>{t('boardHeader.shortcuts.scroll')}</Text>
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
          bubbles: true,
          composed: true,
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
        <Tooltip placement="bottom" title={t('boardHeader.tooltips.npuMiniMap')}>
          <Button
            key={ControlButton.NPU_MAP}
            type="text"
            className={`${styles.controlItem} ${activeButtons[ControlButton.NPU_MAP] ? styles.activeControlItem : ''}`}
            icon={<PicLeftOutlined />}
            onClick={() => toggleActive(ControlButton.NPU_MAP)}
          />
        </Tooltip>

        {!isSingleGraph && (
          <Tooltip placement="bottom" title={t('boardHeader.tooltips.benchMiniMap')}>
            <Button
              key={ControlButton.BENCH_MAP}
              type="text"
              className={`${styles.controlItem} ${
                activeButtons[ControlButton.BENCH_MAP] ? styles.activeControlItem : ''
              }`}
              icon={<PicRightOutlined />}
              onClick={() => toggleActive(ControlButton.BENCH_MAP)}
            />
          </Tooltip>
        )}

        {!isSingleGraph && (
          <Tooltip placement="bottom" title={t('boardHeader.tooltips.syncExpand')}>
            <Button
              key={ControlButton.MATCH_SYNC}
              type="text"
              className={`${styles.controlItem} ${
                activeButtons[ControlButton.MATCH_SYNC] ? styles.activeControlItem : ''
              }`}
              icon={<SubnodeOutlined />}
              onClick={() => toggleActive(ControlButton.MATCH_SYNC)}
            />
          </Tooltip>
        )}

        <Tooltip placement="bottom" title={t('boardHeader.tooltips.shortcuts')}>
          <Dropdown menu={{ items }} trigger={['click']}>
            <Button
              key={ControlButton.PROFILE}
              type="text"
              className={`${styles.controlItem} ${
                activeButtons[ControlButton.PROFILE] ? styles.activeControlItem : ''
              }`}
              icon={<ReadOutlined />}
              onClick={() => toggleActive(ControlButton.PROFILE)}
            />
          </Dropdown>
        </Tooltip>

        <Tooltip placement="bottom" title={t('boardHeader.tooltips.fitScreen')}>
          <Button
            key={ControlButton.EXPAND}
            type="text"
            style={{ fontSize: 14 }}
            className={`${styles.controlItem} ${activeButtons[ControlButton.EXPAND] ? styles.activeControlItem : ''}`}
            icon={<CompressOutlined />}
            onClick={() => toggleActive(ControlButton.EXPAND)}
          />
        </Tooltip>
      </div>
    </div>
  );
};

export default BoardHeader;
