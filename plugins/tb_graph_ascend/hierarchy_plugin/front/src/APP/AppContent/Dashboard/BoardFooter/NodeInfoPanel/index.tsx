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

import { Empty, Typography, Input, Button, Card, Row, Col } from 'antd';
import styles from './index.module.less';
import { useState } from 'react';
import useGraphStore from '../../../../../store/useGraphStore';
import type { StackInfo } from '../../type';
import { useTranslation } from 'react-i18next';
import { t } from 'i18next';

const Text = Typography.Text;
const TextArea = Input.TextArea;

interface IProps {
  data: StackInfo;
}

interface NodeDetailInfo {
  name: string;
  stackTrace?: string;
  parallelInfo?: string;
}

const StackInfoComponent = (props: NodeDetailInfo) => {
  const { name, stackTrace, parallelInfo } = props;
  const messageApi = useGraphStore((state) => state.messageApi);
  const { t } = useTranslation();
  const [stackBtnVis, setStackBtnVis] = useState<boolean>(false);
  const [parallelBtnVis, setParallelBtnVis] = useState<boolean>(false);

  const handleCopy = (text: string): void => {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        messageApi.success(t('nodeInfoPanel.copySuccessful'));
      })
      .catch((err) => {
        messageApi.error(`${t('nodeInfoPanel.copyFailed')}${err}`);
      });
  };
  return (
    <div className={styles.content}>
      <Text className={styles.nameLabel}>{name}</Text>
      {stackTrace && (
        <div className={styles.stackInfo}>
          <Card title={t('nodeInfoPanel.stackInfo')} size="small">
            <TextArea
              value={stackTrace}
              readOnly
              variant="borderless"
              autoSize={{ minRows: 6, maxRows: 10 }}
              style={{ resize: 'none' }}
              onMouseEnter={() => setStackBtnVis(true)}
              onMouseLeave={() => setStackBtnVis(false)}
            />
            <Button
              className={styles.copyBtn}
              style={{ display: stackBtnVis ? 'unset' : 'none' }}
              onMouseMove={() => setStackBtnVis(true)}
              onClick={() => handleCopy(stackTrace)}
            >
              copy
            </Button>
          </Card>
        </div>
      )}
      {parallelInfo && (
        <div className={styles.parallelInfo}>
          <Card title={t('nodeInfoPanel.parallelMergedInfo')} size="small">
            <TextArea
              value={parallelInfo}
              readOnly
              variant="borderless"
              autoSize={{ minRows: 6, maxRows: 10 }}
              style={{ resize: 'none' }}
              onMouseEnter={() => setParallelBtnVis(true)}
              onMouseLeave={() => setParallelBtnVis(false)}
            />
            <Button
              className={styles.copyBtn}
              style={{ display: parallelBtnVis ? 'unset' : 'none' }}
              onMouseMove={() => setParallelBtnVis(true)}
              onClick={() => handleCopy(parallelInfo)}
            >
              copy
            </Button>
          </Card>
        </div>
      )}
    </div>
  );
};

const NodeInfoPanel = (props: IProps): React.JSX.Element => {
  const { npuName, benchName, npuStack, benchStack, npuParallelMergeInfo, benchParallelMergeInfo } = props.data;
  const { t } = useTranslation();
  const hasNpu = Boolean(npuName && (npuParallelMergeInfo || npuStack));
  const hasBench = Boolean(benchName && (benchParallelMergeInfo || benchStack));

  return (
    <div className={styles.nodeInfoPanel}>
      {hasNpu || hasBench ? (
        <Row gutter={16} style={{ width: '100%' }}>
          {hasNpu && (
            <Col span={hasBench ? 12 : 24}>
              <StackInfoComponent
                name={`${t('nodeInfoPanel.debug')}${npuName}`}
                stackTrace={npuStack}
                parallelInfo={npuParallelMergeInfo}
              />
            </Col>
          )}
          {hasBench && (
            <Col span={hasNpu ? 12 : 24}>
              <StackInfoComponent
                name={`${t('nodeInfoPanel.bench')}${benchName}`}
                stackTrace={benchStack}
                parallelInfo={benchParallelMergeInfo}
              />
            </Col>
          )}
        </Row>
      ) : (
        <Empty style={{ marginTop: '36px' }} description={t('noData')} />
      )}
    </div>
  );
};

export default NodeInfoPanel;
