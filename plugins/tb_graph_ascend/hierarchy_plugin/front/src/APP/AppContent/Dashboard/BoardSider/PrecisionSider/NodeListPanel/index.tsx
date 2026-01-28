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
import { useEffect, useState } from 'react';
import { Input } from 'antd';
import { useTranslation } from 'react-i18next';
import styles from './index.module.less';
import VirtualNodeList from '../../components/VirtualNodeList';
import PanelHeader from '../../components/PanelHeader';
import { NPU_PREFIX } from '../../../../../../common/constant';
import type { NodeWithColor } from '../../../type';
import useGraphStore from '../../../../../../store/useGraphStore';
import { lowerCaseInclude } from '../../../../../../common/utils';

const { Search } = Input;

interface IProps {
  nodeList: NodeWithColor[];
}

const NodeListPanel = (props: IProps): React.JSX.Element => {
  // 勾选框筛选传入的节点列表
  const { nodeList } = props;
  const isSingleGraph = useGraphStore((state) => state.isSingleGraph);
  const isOverflowMode = useGraphStore((state) => state.isOverflowMode);
  const { t } = useTranslation();

  // 节点名称筛选之后的列表
  const [searchedNodes, setSearchedNodes] = useState<NodeWithColor[]>(nodeList);
  // 节点搜索条件
  const [serachName, setSearchName] = useState<string>('');

  const onChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setSearchName(e.target.value);
    setSearchedNodes(nodeList.filter((node) => lowerCaseInclude(node.name, e.target.value)));
  };

  const onSearch = (value: string): void => {
    setSearchName(value);
    setSearchedNodes(nodeList.filter((node) => lowerCaseInclude(node.name, value)));
  };

  useEffect(() => {
    setSearchedNodes(nodeList.filter((node) => lowerCaseInclude(node.name, serachName)));
  }, [nodeList]);

  return (
    <div className={styles.nodeListPanel}>
      <PanelHeader nodes={searchedNodes.map((node) => node.name)} prefix={isSingleGraph ? '' : NPU_PREFIX} />
      <Search
        className={styles.search}
        placeholder={t('searchName')}
        onSearch={onSearch}
        onChange={onChange}
        maxLength={200}
        allowClear
      />
      <VirtualNodeList
        nodes={searchedNodes}
        query={serachName}
        height={`calc(100vh - ${isOverflowMode ? 310 : 340}px)`}
        prefix={isSingleGraph ? '' : NPU_PREFIX}
      />
    </div>
  );
};

export default NodeListPanel;
