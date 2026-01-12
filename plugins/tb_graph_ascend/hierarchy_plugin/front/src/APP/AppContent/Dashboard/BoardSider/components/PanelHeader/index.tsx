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

  const selectedUp = (): void => {
    const selectedIndex = nodes.indexOf(selectedNode.replace(prefix, ''));
    if (selectedIndex <= 0) {
      return;
    }
    setSelectedNode(`${prefix}${nodes[selectedIndex - 1]}`);
  };

  const selectedDown = (): void => {
    const selectedIndex = nodes.indexOf(selectedNode.replace(prefix, ''));
    if (selectedIndex < 0 || selectedIndex === nodes.length - 1) {
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
