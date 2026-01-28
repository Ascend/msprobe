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
