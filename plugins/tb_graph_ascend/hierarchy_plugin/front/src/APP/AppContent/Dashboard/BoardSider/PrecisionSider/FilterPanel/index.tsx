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
import { Checkbox, Segmented, Tooltip, Typography } from 'antd';
import styles from './index.module.less';
import { QuestionCircleOutlined } from '@ant-design/icons';
import useGraphStore from '../../../../../../store/useGraphStore';
import { useEffect, useState } from 'react';
import { OVERFLOW_COLOR, PRECISION_ERROR_COLOR } from '../../../../../../common/constant';
import { useTranslation } from 'react-i18next';

interface IProps {
  onFilterNodes: (value: Array<string | number>) => void;
}

interface PrecisionPanelProps {
  precisionRange: PrecisionRange[];
  title: string;
  onFilterNodes: (value: Array<string | number>) => void;
}

interface PrecisionRange {
  label: string;
  color: string;
  value: string | number;
  disabled?: boolean;
}

const Text = Typography.Text;

const PrecisionPanel = (props: PrecisionPanelProps): React.JSX.Element => {
  const [checkItems, setCheckItems] = useState<Array<string | number>>([]);
  const currentDir = useGraphStore((state) => state.currentMetaDir);
  const currentFile = useGraphStore((state) => state.currentMetaFile);
  const currentFileType = useGraphStore((state) => state.currentMetaFileType);
  const currentStep = useGraphStore((state) => state.currentMetaStep);
  const currentRank = useGraphStore((state) => state.currentMetaRank);
  const currentMicroStep = useGraphStore((state) => state.currentMetaMicroStep);
  const isOverflowMode = useGraphStore((state) => state.isOverflowMode);
  const isSingleGraph = useGraphStore((state) => state.isSingleGraph);
  const task = useGraphStore((state) => state.task);
  const { t } = useTranslation();
  const onChange = (value: Array<string | number>): void => {
    setCheckItems(value);
    props.onFilterNodes(value);
  };

  // 改变数据展示模式和展示模型清空勾选条件
  useEffect(() => {
    setCheckItems([]);
    props.onFilterNodes([]);
  }, [isOverflowMode, currentDir, currentFile, currentFileType, currentStep, currentRank, currentMicroStep]);

  return (
    <div className={styles.tabItem}>
      <Text className={styles.title}>
        {props.title}
        {!isOverflowMode && !isSingleGraph && (
          <Tooltip title={t(`precision_desc.${task}`)} placement="right">
            <QuestionCircleOutlined className={styles.icon} data-testid="precisionErrorTooltip" />
          </Tooltip>
        )}
      </Text>
      <Checkbox.Group value={checkItems} className={styles.checkBoxItemGroup} onChange={onChange}>
        {props.precisionRange.map((item) => {
          return (
            <Checkbox key={item.value} className={styles.checkBoxItem} value={item.value} disabled={item.disabled}>
              <div className={styles.colorBlock} style={{ backgroundColor: item.color }}></div>
              <Text>{item.label}</Text>
            </Checkbox>
          );
        })}
      </Checkbox.Group>
    </div>
  );
};

const FilterPanel = (props: IProps): React.JSX.Element => {
  const hasOverflow = useGraphStore((state) => state.hasOverflow);
  const isOverflowMode = useGraphStore((state) => state.isOverflowMode);
  const isSingleGraph = useGraphStore((state) => state.isSingleGraph);
  const setOverflowMode = useGraphStore((state) => state.setOverflowMode);
  const { t } = useTranslation();

  const errorRanges: PrecisionRange[] = [
    { label: 'Pass', color: PRECISION_ERROR_COLOR.pass, value: 'pass', disabled: isSingleGraph },
    { label: 'Warning', color: PRECISION_ERROR_COLOR.warning, value: 'warning', disabled: isSingleGraph },
    { label: 'Error', color: PRECISION_ERROR_COLOR.error, value: 'error', disabled: isSingleGraph },
    { label: t('Unmatched'), color: PRECISION_ERROR_COLOR.unmatched, value: -1 },
  ];

  const overflowRanges: PrecisionRange[] = [
    { label: 'Medium', color: OVERFLOW_COLOR.medium, value: 'medium' },
    { label: 'High', color: OVERFLOW_COLOR.high, value: 'high' },
    { label: 'Critical', color: OVERFLOW_COLOR.critical, value: 'critical' },
  ];
  const onChangeMode = (value: string): void => {
    setOverflowMode(value === '2');
  };

  return (
    <div className={styles.filterPanel}>
      <Segmented
        options={[
          { label: t('accuracy_error'), value: '1', className: styles.segmentedLabel },
          {
            label: t('overflow'),
            value: '2',
            className: styles.segmentedLabel,
            disabled: !hasOverflow,
            title: hasOverflow ? t('overflow') : t('noOverflowData'),
          },
        ]}
        block
        onChange={onChangeMode}
      />
      <PrecisionPanel
        precisionRange={isOverflowMode ? overflowRanges : errorRanges}
        title={isOverflowMode ? t('overflow') : t('accuracy_error')}
        onFilterNodes={props.onFilterNodes}
      />
    </div>
  );
};

export default FilterPanel;
