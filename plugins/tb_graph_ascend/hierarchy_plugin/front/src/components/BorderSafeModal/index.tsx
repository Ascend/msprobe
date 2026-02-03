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
        {(Array.isArray(fileErrorList) ? fileErrorList : []).map((item) => {
          return (
            <div key={` $ {item.run}/ $ {item.tag}`}>
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
