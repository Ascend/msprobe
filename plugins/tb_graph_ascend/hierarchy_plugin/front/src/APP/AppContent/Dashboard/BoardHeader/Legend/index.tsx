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
import './index.less';
import { Tooltip, Typography } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
const { Text } = Typography;

const Legend = () => {
  const { t } = useTranslation();
  return (
    <div className="legend">
      <div className="legend-item">
        <svg className="module-rect"></svg>
        <Text className="legend-item-value">{t('legend.moduleOrOperators')}</Text>
      </div>
      <div className="legend-item">
        <svg className="unexpanded-nodes"></svg>
        <Text className="legend-item-value">{t('legend.unexpandedNodes')}</Text>
        <Tooltip placement="right" title={t('tooltip.unexpandedNodes')}>
          <QuestionCircleOutlined />
        </Tooltip>
      </div>
      <div className="legend-item">
        <svg className="api-list "></svg>
        <Text className="legend-item-value">{t('legend.apiList')}</Text>
        <Tooltip placement="right" title={t('tooltip.apiList')}>
          <QuestionCircleOutlined />
        </Tooltip>
      </div>
      <div className="legend-item">
        <svg className="fusion-node">
          <rect width="30" height="15" rx="8" ry="8" x="0" y="0" />
        </svg>
        <Text className="legend-item-value">{t('legend.multiCollection')}</Text>
        <Tooltip placement="right" title={t('tooltip.fusionNode')}>
          <QuestionCircleOutlined />
        </Tooltip>
      </div>
    </div>
  );
};

export default Legend;
