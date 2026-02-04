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
import { Button, Checkbox, Spin, Typography, type CheckboxChangeEvent } from 'antd';
import { useEffect, useState, type JSX } from 'react';
import { useTranslation } from 'react-i18next';
import type { TFunction } from 'i18next';
import SteppedNodeSelect from '../components/SteppedNodeSelect';
import useGraphStore from '../../../../../store/useGraphStore';
import { BENCH_PREFIX, NPU_PREFIX } from '../../../../../common/constant';
import {
  addMatchNodes,
  deleteMatchNodes,
  loadGraphMatchedRelations,
  type AddMatchNodesRequestParams,
  type DeleteMatchNodesRequestParams,
} from '../../../../../api/board';
import type { GraphMatchedRelationsType } from '../../../../../common/type';
import ConfigFilePanel from './ConfigFilePanel';
import type { MatchResultType } from '../../type';
import styles from './index.module.less';

interface MatchBuildProps {
  t: TFunction<'translation', undefined>;
  debugUnmatchNodes: string[];
  benchUnMatchNodes: string[];
  debugUnmatchSelected: string | null;
  benchUnmatchSelected: string | null;
}

interface MatchCancelProps {
  t: TFunction<'translation', undefined>;
  debugMatchedNodes: string[];
  benchMatchedNodes: string[];
  debugMatchSelected: string | null;
  benchMatchSelected: string | null;
}

const Text = Typography.Text;

const MatchBuildPanel = (props: MatchBuildProps): JSX.Element => {
  const { t, debugUnmatchNodes, benchUnMatchNodes, debugUnmatchSelected, benchUnmatchSelected } = props;
  const [desChecked, setDesChecked] = useState<boolean>(true);
  const messageApi = useGraphStore((state) => state.messageApi);
  const getCurrentMetaData = useGraphStore((state) => state.getCurrentMetaData);
  const setGraphMatchedRelations = useGraphStore((state) => state.setGraphMatchedRelations);
  const switchMatchedStatus = useGraphStore((state) => state.switchMatchedStatus);

  const onCheck = (e: CheckboxChangeEvent): void => {
    setDesChecked(e.target.checked);
  };

  const onMatchBuild = async (): Promise<void> => {
    if (debugUnmatchSelected === null) {
      messageApi.warning(t('selectDebugMatchNode'));
      return;
    }
    if (benchUnmatchSelected === null) {
      messageApi.warning(t('selectBenchMatchNode'));
      return;
    }
    const params: AddMatchNodesRequestParams = {
      npuNodeName: debugUnmatchSelected,
      benchNodeName: benchUnmatchSelected,
      metaData: getCurrentMetaData(),
      isMatchChildren: desChecked,
    };
    const { success, data, error } = await addMatchNodes<MatchResultType>(params);
    if (success) {
      if (data) {
        setGraphMatchedRelations({
          npuMatchNodes: data.npuMatchNodes,
          benchMatchNodes: data.benchMatchNodes,
          npuUnMatchNodes: data.npuUnMatchNodes,
          benchUnMatchNodes: data.benchUnMatchNodes,
        });
        // 更新节点之间的匹配关系,更新匹配精度,节点重新上色
        document.dispatchEvent(new CustomEvent('updateHierarchyData', { bubbles: true, composed: true }));
        // 刷新选中节点匹配状态
        switchMatchedStatus();
        messageApi.success(t('matchSuccess'));
      }
    } else {
      messageApi.error(error);
    }
  };

  return (
    <div className={styles.matchPanel}>
      <Text className={styles.panelTitle}>{t('matchNodes')}</Text>
      <SteppedNodeSelect
        prefix={NPU_PREFIX}
        label={`${t('debug')}-${t('unmatched')}(${debugUnmatchNodes.length})`}
        nodeList={debugUnmatchNodes}
        selectedValue={debugUnmatchSelected}
        testPrefix="debugUnmatched"
      />
      <SteppedNodeSelect
        prefix={BENCH_PREFIX}
        label={`${t('bench')}-${t('unmatched')}(${benchUnMatchNodes.length})`}
        nodeList={benchUnMatchNodes}
        selectedValue={benchUnmatchSelected}
        testPrefix="benchUnmatched"
      />
      <Checkbox
        className={styles.desCheckbox}
        checked={desChecked}
        onChange={onCheck}
        data-testid="matchDesendantCheckbox"
      >
        {t('manageDesendant')}
      </Checkbox>
      <Button
        color="primary"
        variant="filled"
        className={styles.matchButton}
        iconPosition="end"
        onClick={onMatchBuild}
        data-testid="matchButton"
      >
        {t('matchNodes')}
      </Button>
    </div>
  );
};

const MatchCancelPanel = (props: MatchCancelProps): JSX.Element => {
  const { t, debugMatchedNodes, benchMatchedNodes, debugMatchSelected, benchMatchSelected } = props;
  const [desChecked, setDesChecked] = useState<boolean>(true);
  const messageApi = useGraphStore((state) => state.messageApi);
  const getCurrentMetaData = useGraphStore((state) => state.getCurrentMetaData);
  const setGraphMatchedRelations = useGraphStore((state) => state.setGraphMatchedRelations);
  const switchMatchedStatus = useGraphStore((state) => state.switchMatchedStatus);

  const onCheck = (e: CheckboxChangeEvent): void => {
    setDesChecked(e.target.checked);
  };
  const onUnmatchBuild = async (): Promise<void> => {
    if (debugMatchSelected === null || benchMatchSelected === null) {
      messageApi.warning(t('selectMatchedNodes'));
      return;
    }
    const params: DeleteMatchNodesRequestParams = {
      npuNodeName: debugMatchSelected,
      benchNodeName: benchMatchSelected,
      metaData: getCurrentMetaData(),
      isUnMatchChildren: desChecked,
    };
    const { success, data, error } = await deleteMatchNodes<MatchResultType>(params);
    if (success) {
      if (data) {
        setGraphMatchedRelations({
          npuMatchNodes: data.npuMatchNodes,
          benchMatchNodes: data.benchMatchNodes,
          npuUnMatchNodes: data.npuUnMatchNodes,
          benchUnMatchNodes: data.benchUnMatchNodes,
        });
        // 更新节点之间的匹配关系,更新匹配精度,节点重新上色
        document.dispatchEvent(new CustomEvent('updateHierarchyData', { bubbles: true, composed: true }));
        // 刷新选中节点匹配状态
        switchMatchedStatus();
        messageApi.success(t('unmatchSuccess'));
      }
    } else {
      messageApi.error(error);
    }
  };
  return (
    <div className={styles.matchPanel}>
      <Text className={styles.panelTitle}>{t('unmatchNodes')}</Text>
      <SteppedNodeSelect
        prefix={NPU_PREFIX}
        label={`${t('debug')}-${t('matched')}(${debugMatchedNodes.length})`}
        nodeList={debugMatchedNodes}
        selectedValue={debugMatchSelected}
        testPrefix="debugMatched"
      />
      <SteppedNodeSelect
        prefix={BENCH_PREFIX}
        label={`${t('bench')}-${t('matched')}(${benchMatchedNodes.length})`}
        nodeList={benchMatchedNodes}
        selectedValue={benchMatchSelected}
        testPrefix="benchMatched"
      />
      <Checkbox
        className={styles.desCheckbox}
        checked={desChecked}
        onChange={onCheck}
        data-testid="unmatchDesendantCheckbox"
      >
        {t('manageDesendant')}
      </Checkbox>
      <Button
        color="primary"
        variant="filled"
        className={styles.matchButton}
        iconPosition="end"
        onClick={onUnmatchBuild}
        data-testid="unmatchButton"
      >
        {t('unmatchNodes')}
      </Button>
    </div>
  );
};

const MatchSider = (): JSX.Element => {
  const { t } = useTranslation();
  const selectedNode = useGraphStore((state) => state.selectedNode);
  const graphMatchedRelations = useGraphStore((state) => state.graphMatchedRelations);
  const messageApi = useGraphStore((state) => state.messageApi);
  const currentMetaData = useGraphStore((state) => state.getCurrentMetaData)();
  const setGraphMatchedRelations = useGraphStore((state) => state.setGraphMatchedRelations);
  const isMatchCached = useGraphStore((state) => state.isMatchCached);
  const updateMetaDataCacheInMatch = useGraphStore((state) => state.updateMetaDataCacheInMatch);

  const [debugUnmatchedValue, setDebugUnmatchedValue] = useState<string | null>(null);
  const [debugMatchedValue, setDebugMatchedValue] = useState<string | null>(null);
  const [benchUnmatchedValue, setBenchUnmatchedValue] = useState<string | null>(null);
  const [benchMatchedValue, setBenchMatchedValue] = useState<string | null>(null);
  const [spinning, setSpinning] = useState<boolean>(false);
  const fetchGraphMatchedRelations = async (): Promise<void> => {
    setSpinning(true);
    const { success, data, error } = await loadGraphMatchedRelations<GraphMatchedRelationsType>({
      metaData: currentMetaData,
    });
    if (success) {
      if (data) {
        setGraphMatchedRelations(data);
        updateMetaDataCacheInMatch();
      }
    } else {
      messageApi.error(error);
    }
    setSpinning(false);
  };

  useEffect(() => {
    // 已经加载过一次后不需要再次调接口查询数据，此为防止切换侧边栏看板导致重新请求
    if (isMatchCached()) {
      return;
    }
    if (currentMetaData.tag && currentMetaData.run) {
      fetchGraphMatchedRelations();
    }
  }, [JSON.stringify(currentMetaData)]);

  useEffect(() => {
    const selectedDebugNode = selectedNode.replace(NPU_PREFIX, '');
    const selectedBenchNode = selectedNode.replace(BENCH_PREFIX, '');
    const { npuMatchNodes, benchMatchNodes, npuUnMatchNodes, benchUnMatchNodes } = graphMatchedRelations;
    if (npuUnMatchNodes.includes(selectedDebugNode)) {
      setDebugUnmatchedValue(selectedDebugNode);
      setDebugMatchedValue(null);
      setBenchMatchedValue(null);
    } else if (benchUnMatchNodes.includes(selectedBenchNode)) {
      setBenchUnmatchedValue(selectedBenchNode);
      setDebugMatchedValue(null);
      setBenchMatchedValue(null);
    } else if (selectedDebugNode in npuMatchNodes) {
      setDebugUnmatchedValue(null);
      setBenchUnmatchedValue(null);
      setDebugMatchedValue(selectedDebugNode);
      setBenchMatchedValue(npuMatchNodes[selectedDebugNode]);
    } else if (selectedBenchNode in benchMatchNodes) {
      setDebugUnmatchedValue(null);
      setBenchUnmatchedValue(null);
      setDebugMatchedValue(benchMatchNodes[selectedBenchNode]);
      setBenchMatchedValue(selectedBenchNode);
    } else {
      setDebugMatchedValue(null);
      setBenchMatchedValue(null);
      setDebugUnmatchedValue(null);
      setBenchUnmatchedValue(null);
    }
  }, [graphMatchedRelations, selectedNode]);

  return (
    <div data-testid="matchPanel">
      <Spin spinning={spinning} tip={t('loading')}>
        <ConfigFilePanel />
        <MatchBuildPanel
          t={t}
          debugUnmatchNodes={graphMatchedRelations.npuUnMatchNodes}
          benchUnMatchNodes={graphMatchedRelations.benchUnMatchNodes}
          debugUnmatchSelected={debugUnmatchedValue}
          benchUnmatchSelected={benchUnmatchedValue}
        />
        <MatchCancelPanel
          t={t}
          debugMatchedNodes={Object.keys(graphMatchedRelations.npuMatchNodes)}
          benchMatchedNodes={Object.keys(graphMatchedRelations.benchMatchNodes)}
          debugMatchSelected={debugMatchedValue}
          benchMatchSelected={benchMatchedValue}
        />
      </Spin>
    </div>
  );
};

export default MatchSider;
