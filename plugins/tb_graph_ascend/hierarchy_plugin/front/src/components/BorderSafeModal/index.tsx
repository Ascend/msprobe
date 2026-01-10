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

import { Card } from 'antd';
import i18next from 'i18next';

import type { FileErrorListType } from '../../store/types/useGraphStore';

const BorderSafeModal = ({ fileErrorList }: { fileErrorList: FileErrorListType }) => {
  return (
    <div>
      <p>{i18next.t('risk_info')}</p>
      <p>{i18next.t('risk_warning_info')}</p>
      <Card
        size="small"
        title={i18next.t('error_title')}
        style={{
          maxHeight: 260,
          padding: 6,
          overflow: 'auto',
          borderRadius: 4,
          marginTop: 10,
        }}
      >
        {(fileErrorList || []).map((item: FileErrorListType[number]) => {
          return (
            <div>
              <span style={{ color: 'red' }}>
                {item.run}/{item.tag}：
              </span>
              {item.info}
            </div>
          );
        })}
      </Card>
    </div>
  );
};

export default BorderSafeModal;
