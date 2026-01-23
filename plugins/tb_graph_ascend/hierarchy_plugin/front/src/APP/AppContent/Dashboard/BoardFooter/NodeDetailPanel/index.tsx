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

import { Empty, Table, Typography } from 'antd';
import { useMemo } from 'react';
import type { ColumnsType } from 'antd/es/table';
import styles from './index.module.less';
import { useTranslation } from 'react-i18next';

interface IProps {
  npuName: string;
  benchName: string;
  data: Array<Record<string, unknown>>;
}

const Text = Typography.Text;

const ignoreDataIndex = ['data_name', 'isBench', 'isMatched', 'value'];
const NodeDetailPanel = (props: IProps): React.JSX.Element => {
  const { npuName, benchName, data } = props;
  const { t } = useTranslation();
  const getClassName = (record: any, item: string): string => {
    const benchClass = ` ${record.isBench ? styles.benchCell : styles.debugCell}`;
    if (item === 'name') {
      const nameClass = record.isMatched ? styles.matchedName : styles.unMatchedName;
      return `${nameClass}${benchClass}`;
    }
    return benchClass.trim();
  };

  const columns: ColumnsType<Record<string, unknown>> = useMemo(() => {
    if (!Array.isArray(data) || data.length === 0) {
      return [];
    }
    return Array.from(
      data.reduce((keys, item) => {
        Object.keys(item).forEach((key) => {
          if (!ignoreDataIndex.includes(key)) {
            keys.add(String(key));
          }
        });
        return keys;
      }, new Set<string>()),
    ).map((item) => ({
      title: item,
      width: 'auto',
      dataIndex: item,
      render: (text: string) => text || '-',
      onCell: (record: any) => {
        return {
          className: getClassName(record, item),
        };
      },
    }));
  }, [data]);

  return (
    <div className={styles.nodeDetailPanel}>
      {(!!npuName || !!benchName) && (
        <div className={styles.header}>
          {!!npuName && <Text className={styles.nodeNameLabel} title={npuName}>{`${t('debugNode')}${npuName}`}</Text>}
          {!!benchName && (
            <Text className={styles.nodeNameLabel} title={benchName}>{`${t('benchNode')}${benchName}`}</Text>
          )}
          <div className={styles.legends}>
            <div className={styles.debugColorBlock} />
            <Text>{t('debug')}</Text>
            <div className={styles.benchColorBlock} />
            <Text>{t('bench')}</Text>
            <Text className={styles.matchedName}>{t('nodeInfoPanel.matchedParams')}</Text>
            <Text className={styles.unMatchedName}>{t('nodeInfoPanel.unmatchedParams')}</Text>
          </div>
        </div>
      )}
      {data && data.length > 0 ? (
        <div className={styles.tableContainer}>
          <Table
            size="small"
            columns={columns}
            dataSource={data}
            style={{
              whiteSpace: 'nowrap', //防止文本换行影响宽度计算
            }}
            pagination={false}
          />
        </div>
      ) : (
        <Empty style={{ marginTop: '36px' }} description={t('noData')} />
      )}
    </div>
  );
};

export default NodeDetailPanel;
