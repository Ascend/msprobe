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

import { Select, Typography } from 'antd';
import styles from './index.module.less';
import type { SelectProps } from 'antd';

const { Text } = Typography;

interface CustomSelectProps extends SelectProps {
  label: React.ReactNode;
  testId?: string;
}
const CustomSelect = (props: CustomSelectProps) => {
  const { label, testId, ...rest } = props;
  return (
    <div className={styles.customSelectWrapper}>
      <Text className={styles.label}>{label}</Text>
      <Select data-testid={testId} {...rest} className={styles.select} />
    </div>
  );
};
export default CustomSelect;
