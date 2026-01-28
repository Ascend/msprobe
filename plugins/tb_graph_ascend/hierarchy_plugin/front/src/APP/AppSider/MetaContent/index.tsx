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

import styles from './index.module.less';

import CustomSelect from '../../../components/CustomSelect';
import useGraphStore from '../../../store/useGraphStore';
import { useTranslation } from 'react-i18next';
import { useEffect } from 'react';

const MetaContent = () => {
  const {
    currentMetaDir,
    currentMetaStep,
    currentMetaRank,
    currentMetaMicroStep,
    currentMetaFileType,
    metaDirOptions,
    currentMetaFile,
    metaFileOptions,
    stepOptions,
    rankOptions,
    microStepOptions,
    setCurrentMetaDir,
    setCurrentMetaFile,
    setCurrentMetaStep,
    setCurrentMetaRank,
    setCurrentMetaMicroStep,
    updateCurrentMetaFileByDir,
  } = useGraphStore();
  const { t } = useTranslation();
  useEffect(() => {
    updateCurrentMetaFileByDir(currentMetaDir);
  }, [currentMetaDir]);

  return (
    <div className={styles.metaContent} data-testid="metaContentPanel">
      <div className={styles.metaItem}>
        <CustomSelect
          label={t('dir')}
          testId="runSelect"
          value={currentMetaDir}
          style={{ width: 368, marginBottom: 16 }}
          onChange={(value) => {
            setCurrentMetaDir(value);
          }}
          options={metaDirOptions}
        />
        <CustomSelect
          label={t('file')}
          testId="tagSelect"
          value={currentMetaFile}
          style={{ width: 368, marginBottom: 16 }}
          onChange={(value) => {
            setCurrentMetaFile(value);
          }}
          options={metaFileOptions}
        />
        {currentMetaFileType == 'db' && (
          <CustomSelect
            label="Step"
            testId="stepSelect"
            value={currentMetaStep}
            style={{ width: 368, marginBottom: 16 }}
            onChange={(value) => {
              setCurrentMetaStep(value);
            }}
            options={stepOptions}
          />
        )}
        {currentMetaFileType == 'db' && (
          <CustomSelect
            label="Rank"
            testId="rankSelect"
            value={currentMetaRank}
            style={{ width: 368, marginBottom: 16 }}
            onChange={(value) => {
              setCurrentMetaRank(value);
            }}
            options={rankOptions}
          />
        )}
        <CustomSelect
          label="MicroStep"
          testId="microStepSelect"
          value={currentMetaMicroStep}
          style={{ width: 368, marginBottom: 16 }}
          onChange={(value) => {
            setCurrentMetaMicroStep(value);
          }}
          options={microStepOptions}
        />
      </div>
    </div>
  );
};

export default MetaContent;
