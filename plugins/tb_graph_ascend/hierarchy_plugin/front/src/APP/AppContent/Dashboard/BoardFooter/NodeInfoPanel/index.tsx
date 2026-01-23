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

import { Empty, Typography, Input, Button, Row, Col, Collapse, type CollapseProps } from 'antd';
import styles from './index.module.less';
import type { MouseEvent } from 'react';
import useGraphStore from '../../../../../store/useGraphStore';
import type { StackInfo } from '../../type';
import { useTranslation } from 'react-i18next';

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

const StackInfoComponent = (props: NodeDetailInfo): React.JSX.Element => {
  const { name, stackTrace, parallelInfo } = props;
  const messageApi = useGraphStore((state) => state.messageApi);
  const { t } = useTranslation();
  const items: CollapseProps['items'] = [];

  if (stackTrace) {
    items.push({
      key: '1',
      label: t('nodeInfoPanel.stackInfo'),
      children: (
        <TextArea
          value={stackTrace}
          readOnly
          variant="borderless"
          autoSize={{ minRows: 6, maxRows: 10 }}
          style={{ resize: 'none' }}
        />
      ),
      extra: (
        <Button className={styles.copyBtn} onClick={(e) => handleCopy(e, stackTrace)} size="small">
          {t('copy')}
        </Button>
      ),
    });
  }
  if (parallelInfo) {
    items.push({
      key: '2',
      label: t('nodeInfoPanel.parallelMergedInfo'),
      children: (
        <TextArea
          value={parallelInfo}
          readOnly
          variant="borderless"
          autoSize={{ minRows: 6, maxRows: 10 }}
          style={{ resize: 'none' }}
        />
      ),
      extra: (
        <Button className={styles.copyBtn} onClick={(e) => handleCopy(e, parallelInfo)} size="small">
          {t('copy')}
        </Button>
      ),
    });
  }

  const handleCopy = (e: MouseEvent, text: string): void => {
    e.stopPropagation();
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
      <Text className={styles.nameLabel} title={name}>
        {name}
      </Text>
      <Collapse defaultActiveKey={['1']} items={items} />
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
