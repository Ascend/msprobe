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
import { useEffect, useState } from 'react';
import styles from './index.module.less';
import { Input } from 'antd';
import VirtualNodeList from '../../components/VirtualNodeList';
import PanelHeader from '../../components/PanelHeader';
import { NPU_PREFIX } from '../../../../../../common/constant';
import type { NodeWithColor } from '../../../type';
import useGraphStore from '../../../../../../store/useGraphStore';
import { lowerCaseInclude } from '../../../../../../common/utils';
import { useTranslation } from 'react-i18next';

const { Search } = Input;

interface IProps {
  nodeList: NodeWithColor[];
}

const NodeListPanel = (props: IProps): React.JSX.Element => {
  // 勾选框筛选传入的节点列表
  const { nodeList } = props;
  const isSingleGraph = useGraphStore((state) => state.isSingleGraph);
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
        height={'calc(100vh - 348px)'}
        prefix={isSingleGraph ? '' : NPU_PREFIX}
      />
    </div>
  );
};

export default NodeListPanel;
