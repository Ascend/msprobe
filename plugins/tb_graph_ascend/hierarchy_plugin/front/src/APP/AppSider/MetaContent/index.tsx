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
    <div className={styles.metaContent}>
      <div className={styles.metaItem}>
        <CustomSelect
          label={t('run')}
          value={currentMetaDir}
          style={{ width: 368, marginBottom: 16 }}
          onChange={(value) => {
            setCurrentMetaDir(value);
          }}
          options={metaDirOptions}
        />
        <CustomSelect
          label={t('tag')}
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
