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
import { Radio, Input, type RadioChangeEvent, Spin } from 'antd';
import type { CheckboxGroupProps } from 'antd/es/checkbox';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import VirtualNodeList from '../components/VirtualNodeList';
import PanelHeader from '../components/PanelHeader';
import styles from './index.module.less';
import { BENCH_PREFIX, NPU_PREFIX } from '../../../../../common/constant';
import { lowerCaseInclude } from '../../../../../common/utils';
import useGraphStore from '../../../../../store/useGraphStore';
import type { GraphAllNodeType } from '../../../../../common/type';
import { loadGraphAllNodeList } from '../../../../../api/board';

const { Search } = Input;

const SearchSider = () => {
  const isSingleGraph = useGraphStore((state) => state.isSingleGraph);
  const currentMetaData = useGraphStore((state) => state.getCurrentMetaData)();
  const messageApi = useGraphStore((state) => state.messageApi);
  const { npuNodeList, benchNodeList } = useGraphStore((state) => state.graphNodeList);
  const setGraphNodeList = useGraphStore((state) => state.setGraphNodeList);
  const isSearchCached = useGraphStore((state) => state.isSearchCached);
  const updateMetaDataCacheInSearch = useGraphStore((state) => state.updateMetaDataCacheInSearch);
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
  const [spinning, setSpinning] = useState<boolean>(false);

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

  const fetchAllNodeList = async (): Promise<void> => {
    setSpinning(true);
    const { success, data, error } = await loadGraphAllNodeList<GraphAllNodeType>({
      metaData: currentMetaData,
    });
    if (success) {
      if (data) {
        setGraphNodeList(data);
        updateMetaDataCacheInSearch();
      }
    } else {
      messageApi.error(error);
    }
    setSpinning(false);
  };

  useEffect(() => {
    // 已经加载过一次后不需要再次调接口查询数据，此为防止切换侧边栏看板导致重新请求
    if (isSearchCached()) {
      return;
    }
    if (currentMetaData.tag && currentMetaData.run) {
      fetchAllNodeList();
    }
  }, [JSON.stringify(currentMetaData)]);

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
      <Spin spinning={spinning} tip={t('loading')}>
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
      </Spin>
    </div>
  );
};

export default SearchSider;
