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
