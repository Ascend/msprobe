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
