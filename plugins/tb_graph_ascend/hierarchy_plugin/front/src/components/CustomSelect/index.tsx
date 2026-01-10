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

import { Select, Typography } from 'antd';
import styles from './index.module.less';
import type { SelectProps } from 'antd';

const { Text } = Typography;

interface CustomSelectProps extends SelectProps {
  label: React.ReactNode;
}
const CustomSelect = (props: CustomSelectProps) => {
  const { label, ...rest } = props;
  return (
    <div className={styles.customSelectWrapper}>
      <Text className={styles.label}>{label}</Text>
      <Select {...rest} className={styles.select} />
    </div>
  );
};
export default CustomSelect;
