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
import { Typography } from 'antd';
import { DownOutlined, UpOutlined } from '@ant-design/icons';
import styles from './index.module.less';
import useGraphStore from '../../../../../../store/useGraphStore';
import { useTranslation } from 'react-i18next';

interface IProps {
  nodes: string[];
  prefix: string;
}

const Text = Typography.Text;

const PanelHeader = (props: IProps): React.JSX.Element => {
  const { nodes, prefix } = props;
  const { t } = useTranslation();
  // 当前选中节点
  const selectedNode = useGraphStore((state) => state.selectedNode);
  const setSelectedNode = useGraphStore((state) => state.setSelectedNode);
  const messageApi = useGraphStore((state) => state.messageApi);

  const selectedUp = (): void => {
    const selectedIndex = nodes.indexOf(selectedNode.replace(prefix, ''));
    if (selectedIndex < 0) {
      return;
    }
    if (selectedIndex === 0) {
      messageApi.info(t('topOfList'));
      return;
    }
    setSelectedNode(`${prefix}${nodes[selectedIndex - 1]}`);
  };

  const selectedDown = (): void => {
    const selectedIndex = nodes.indexOf(selectedNode.replace(prefix, ''));
    if (selectedIndex < 0) {
      return;
    }
    if (selectedIndex === nodes.length - 1) {
      messageApi.info(t('bottomOfList'));
      return;
    }
    setSelectedNode(`${prefix}${nodes[selectedIndex + 1]}`);
  };

  return (
    <div className={styles.panelHeader}>
      <Text className={styles.title} data-testid="nodeCountLabel">
        {t('nodeList', { count: nodes.length })}
      </Text>
      <Text className={styles.icons}>
        <UpOutlined className={styles.icon} onClick={selectedUp} />
        <DownOutlined className={styles.icon} onClick={selectedDown} />
      </Text>
    </div>
  );
};

export default PanelHeader;
