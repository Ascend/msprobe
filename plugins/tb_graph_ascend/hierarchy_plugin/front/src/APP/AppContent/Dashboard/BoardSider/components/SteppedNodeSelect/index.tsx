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

import { Select, Typography } from 'antd';
import { ArrowDownOutlined, ArrowUpOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import styles from './index.module.less';
import useGraphStore from '../../../../../../store/useGraphStore';

interface IProps {
  prefix: string;
  label: string;
  nodeList: string[];
  selectedValue: string | null;
  testPrefix: string;
}

const Text = Typography.Text;

const SteppedNodeSelect = (props: IProps): React.JSX.Element => {
  const { prefix, label, nodeList, selectedValue, testPrefix } = props;
  const { t } = useTranslation();
  const setSelectedNode = useGraphStore((state) => state.setSelectedNode);
  const messageApi = useGraphStore((state) => state.messageApi);

  const onSelect = (value: string): void => {
    setSelectedNode(`${prefix}${value}`);
  };
  const onUpClick = (): void => {
    if (nodeList.length === 0) {
      messageApi.warning(t('emptyNodeList'));
      return;
    }
    if (selectedValue === null) {
      setSelectedNode(`${prefix}${nodeList[nodeList.length - 1]}`);
      return;
    }
    const index = nodeList.indexOf(selectedValue);
    if (index < 0) {
      return;
    }
    if (index === 0) {
      messageApi.info(t('topOfList'));
      return;
    }
    setSelectedNode(`${prefix}${nodeList[index - 1]}`);
  };
  const onDownClick = (): void => {
    if (nodeList.length === 0) {
      messageApi.warning(t('emptyNodeList'));
      return;
    }
    if (selectedValue === null) {
      setSelectedNode(`${prefix}${nodeList[0]}`);
      return;
    }
    const index = nodeList.indexOf(selectedValue);
    if (index < 0) {
      return;
    }
    if (index === nodeList.length - 1) {
      messageApi.info(t('bottomOfList'));
      return;
    }
    setSelectedNode(`${prefix}${nodeList[index + 1]}`);
  };

  return (
    <>
      <Text className={styles.textLabel} data-testid={`${testPrefix}Count`}>
        {label}
      </Text>
      <div className={styles.selectDiv}>
        <Select
          value={selectedValue}
          options={nodeList.map((item) => ({ label: item, value: item }))}
          className={styles.nodeSelect}
          onSelect={onSelect}
          placeholder={t('notSelected')}
          data-testid={`${testPrefix}Select`}
          showSearch
        />
        <ArrowUpOutlined className={styles.iconButton} onClick={onUpClick} data-testid={`${testPrefix}Up`} />
        <ArrowDownOutlined className={styles.iconButtonRight} onClick={onDownClick} data-testid={`${testPrefix}Down`} />
      </div>
    </>
  );
};

export default SteppedNodeSelect;
