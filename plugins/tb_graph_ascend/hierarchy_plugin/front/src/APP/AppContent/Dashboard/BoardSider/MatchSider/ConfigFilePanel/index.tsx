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
import { useState } from 'react';
import { Button, Select, Spin, Tooltip, Typography } from 'antd';
import { useTranslation } from 'react-i18next';
import { QuestionCircleOutlined } from '@ant-design/icons';
import useGraphStore from '../../../../../../store/useGraphStore';
import {
  addMatchNodesByConfig,
  generateConfigFile,
  type AddMatchNodesByConfigRequestParams,
} from '../../../../../../api/board';
import type { MatchResultType } from '../../../type';
import styles from './index.module.less';

const Text = Typography.Text;

const ConfigFilePanel = (): React.JSX.Element => {
  const fileName = useGraphStore((state) => state.currentMetaFile);
  const currentStep = useGraphStore((state) => state.currentMetaStep);
  const currentRank = useGraphStore((state) => state.currentMetaRank);
  const matchedConfigFilesOptions = useGraphStore((state) => state.matchedConfigFilesOptions);
  const setMatchedConfigFilesOptions = useGraphStore((state) => state.setMatchedConfigFilesOptions);
  const setGraphMatchedRelations = useGraphStore((state) => state.setGraphMatchedRelations);
  const messageApi = useGraphStore((state) => state.messageApi);
  const getCurrentMetaData = useGraphStore((state) => state.getCurrentMetaData);
  const { t } = useTranslation();
  const [spin, setSpin] = useState<boolean>(false);

  const addMatchNodes = async (configFile: string) => {
    const params: AddMatchNodesByConfigRequestParams = {
      configFile,
      metaData: getCurrentMetaData(),
    };
    setSpin(true);
    const { success, data, error } = await addMatchNodesByConfig<MatchResultType>(params).finally(() => setSpin(false));
    if (success) {
      if (data) {
        setGraphMatchedRelations({
          npuMatchNodes: data.npuMatchNodes,
          benchMatchNodes: data.benchMatchNodes,
          npuUnMatchNodes: data.npuUnMatchNodes,
          benchUnMatchNodes: data.benchUnMatchNodes,
        });
        messageApi.success(
          t('matchSuccessByConfig', {
            total: data.matchedTotal,
            success: data.matchedSuccess,
            failed: data.matchedTotal - data.matchedSuccess,
          }),
        );
        // 更新节点之间的匹配关系,更新匹配精度,节点重新上色
        document.dispatchEvent(new CustomEvent('updateHierarchyData', { bubbles: true, composed: true }));
      }
    } else {
      messageApi.error(error);
    }
  };

  const onChange = (value: string): void => {
    if (value === '-1' || value === undefined) {
      return;
    }
    addMatchNodes(value);
  };

  const onGenerateConfig = async (): Promise<void> => {
    setSpin(true);
    const params = { metaData: getCurrentMetaData() };
    const { success, data, error } = await generateConfigFile<string>(params).finally(() => setSpin(false));
    if (success) {
      if (data) {
        if (!matchedConfigFilesOptions.includes(data)) {
          setMatchedConfigFilesOptions([data, ...matchedConfigFilesOptions]);
        }
        messageApi.success(`${t('file_generate_success')}${data}`);
      }
    } else {
      messageApi.error(error);
    }
  };
  return (
    <div className={styles.configFilePanel}>
      <Spin spinning={spin} tip={t('loading')}>
        <Text className={styles.panelTitle}>{t('configFile')}</Text>
        <Text className={styles.selectLabel}>
          {t('selectConfigFile')}
          <Tooltip title={t('selectConfigFileTooltip')} placement="right">
            <QuestionCircleOutlined className={styles.icon} />
          </Tooltip>
        </Text>
        <Select
          options={[
            {
              value: '-1',
              label: t('notSelected'),
            },
            ...matchedConfigFilesOptions.map((item) => ({ value: item, label: item })),
          ]}
          data-testid="configurationSelect"
          className={styles.fileSelect}
          defaultValue={'-1'}
          onChange={onChange}
        />
      </Spin>
      <Button
        color="primary"
        variant="filled"
        className={styles.generateButton}
        data-testid="configurationGenerateButton"
        icon={
          <Tooltip
            title={
              <Text className={styles.tooltipContent}>
                {t('generateConfigFileTooltipBefore')}
                <Text className={styles.fileNameTooltip}>{`${fileName}_${currentStep}_${currentRank}.vis.config`}</Text>
                {t('generateConfigFileTooltipAfter')}
              </Text>
            }
            placement="right"
          >
            <QuestionCircleOutlined className={styles.icon} data-testid="configurationGenerateTooltip" />
          </Tooltip>
        }
        iconPosition="end"
        onClick={onGenerateConfig}
      >
        {t('generateConfigFile')}
      </Button>
    </div>
  );
};

export default ConfigFilePanel;
