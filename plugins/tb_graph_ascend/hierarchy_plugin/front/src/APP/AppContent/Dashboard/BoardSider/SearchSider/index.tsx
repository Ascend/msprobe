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
import { Radio, Input, type RadioChangeEvent } from 'antd';
import useGraphStore from '../../../../../store/useGraphStore';
import type { CheckboxGroupProps } from 'antd/es/checkbox';
import { useEffect, useState } from 'react';
import VirtualNodeList from '../components/VirtualNodeList';
import PanelHeader from '../components/PanelHeader';
import styles from './index.module.less';
import { BENCH_PREFIX, NPU_PREFIX } from '../../../../../common/constant';
import { lowerCaseInclude } from '../../../../../common/utils';
import { useTranslation } from 'react-i18next';

const { Search } = Input;

const SearchSider = () => {
  const isSingleGraph = useGraphStore((state) => state.isSingleGraph);
  const { npuNodeList, benchNodeList } = useGraphStore((state) => state.graphNodeList);
  const { t } = useTranslation();

  const options: CheckboxGroupProps<string>['options'] = [
    { label: t('debug'), value: NPU_PREFIX },
    { label: t('bench'), value: BENCH_PREFIX, disabled: isSingleGraph },
  ];

  const [searchName, setSearchName] = useState<string>('');
  const [nodeList, setNodeList] = useState<string[]>(npuNodeList);
  // 节点名称筛选之后的列表
  const [searchedNodes, setSearchedNodes] = useState<string[]>(nodeList);
  // 查询哪一侧节点
  const [currentSide, setCurrentSide] = useState<string>(NPU_PREFIX);

  const onRadioChange = (e: RadioChangeEvent): void => {
    setCurrentSide(e.target.value);
    setNodeList(e.target.value === NPU_PREFIX ? npuNodeList : benchNodeList);
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setSearchName(e.target.value);
  };

  const onSearch = (value: string): void => {
    setSearchName(value);
  };

  useEffect(() => {
    setSearchedNodes(nodeList.filter((node) => lowerCaseInclude(node, searchName)));
  }, [nodeList, searchName]);

  useEffect(() => {
    setCurrentSide(NPU_PREFIX);
    setNodeList(npuNodeList);
    setSearchName('');
  }, [npuNodeList, benchNodeList]);

  return (
    <div className={styles.searchSider} data-testid="searchPanel">
      <Radio.Group options={options} onChange={onRadioChange} value={currentSide} />
      <Search
        value={searchName}
        className={styles.search}
        placeholder={t('searchName')}
        onSearch={onSearch}
        onChange={onInputChange}
        maxLength={200}
        allowClear
      />
      <PanelHeader nodes={searchedNodes} prefix={isSingleGraph ? '' : currentSide} />
      <VirtualNodeList
        nodes={searchedNodes.map((node) => {
          return { name: node };
        })}
        query={searchName}
        height={'calc(100vh - 150px)'}
        prefix={isSingleGraph ? '' : currentSide}
        visibleItems={40}
      />
    </div>
  );
};

export default SearchSider;
