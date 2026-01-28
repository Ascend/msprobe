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
import { Button, Typography } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useVisualizedStore } from '../../../../store/useVisualizedStore';
import useGlobalStore from '../../../../store/useGlobalStore';
import { BUILD_STEP, CURRENT_PAGE, CURRENT_TAB, INIT_CONVERT_GRAPH_ARGS } from '../../../../common/constant';
import useGraphStore from '../../../../store/useGraphStore';
import type { ConvertParamsType } from '../../../../store/types/useVisualizedStore';
const { Paragraph } = Typography;

interface BuildProcessProps {
  resultStatus: boolean;
  resultLog: string;
}

const BuildResult = (props: BuildProcessProps) => {
  const { t } = useTranslation();
  const { resultStatus, resultLog } = props;
  const convertedGraphArgs = useVisualizedStore((state) => state.convertedGraphArgs);
  const setCurrentBuildStep = useVisualizedStore((state) => state.setCurrentBuildStep);
  const setConvertedGraphArgs = useVisualizedStore((state) => state.setConvertedGraphArgs);
  const setCurrentPage = useGlobalStore((state) => state.setCurrentPage);
  const setCurrentTab = useGlobalStore((state) => state.setCurrentTab);
  const fetchFileInfoList = useGraphStore((state) => state.fetchFileInfoList);
  const loadBuildResult = async () => {
    await fetchFileInfoList(convertedGraphArgs.output_path);
    setCurrentPage(CURRENT_PAGE.DASHBOARD);
    setCurrentTab(CURRENT_TAB.PRECISION_TAB);
    setCurrentBuildStep(BUILD_STEP.BUILD_CONFIG);
    setConvertedGraphArgs(INIT_CONVERT_GRAPH_ARGS as ConvertParamsType);
  };
  const rebuild = () => {
    setCurrentBuildStep(BUILD_STEP.BUILD_CONFIG);
    setConvertedGraphArgs({
      ...INIT_CONVERT_GRAPH_ARGS,
      ...convertedGraphArgs,
    });
  };

  const backToBuildConfig = () => {
    setCurrentBuildStep(BUILD_STEP.BUILD_CONFIG);
    setConvertedGraphArgs(INIT_CONVERT_GRAPH_ARGS as ConvertParamsType);
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', marginTop: 300 }}>
      {resultStatus && (
        <div style={{ width: 800 }}>
          <div style={{ fontSize: 20, fontWeight: 700 }}>{t('buildResult.success.title')}</div>
          <div style={{ fontSize: 14, marginTop: 20, fontWeight: 700 }}>
            <CheckCircleOutlined style={{ color: 'green', marginRight: 6, fontSize: 16, fontWeight: 700 }} />
            {t('buildResult.success.message', { outputPath: convertedGraphArgs.output_path })}
          </div>
          <div style={{ marginTop: 32 }}>
            <Button type="primary" style={{ width: 268 }} onClick={loadBuildResult}>
              {t('buildResult.button.loadFile')}
            </Button>
            <Button style={{ marginLeft: 20, width: 268 }} onClick={backToBuildConfig}>
              {t('buildResult.button.back')}
            </Button>
          </div>
        </div>
      )}
      {!resultStatus && (
        <div style={{ width: 552 }}>
          <div style={{ fontSize: 20, fontWeight: 700 }}>{t('buildResult.failure.title')}</div>
          <div style={{ fontSize: 14, marginTop: 20, fontWeight: 700 }}>
            <CloseCircleOutlined style={{ color: 'red', marginRight: 6, fontSize: 16, fontWeight: 700 }} />
            {t('buildResult.failure.message')}
          </div>
          <div style={{ fontSize: 14, marginTop: 20, fontWeight: 500 }}>{t('buildResult.failure.logTitle')}</div>
          <Paragraph
            copyable
            style={{
              background: 'rgba(25,59,103,0.05)',
              marginTop: 12,
              padding: 16,
              fontSize: 14,
              borderRadius: 8,
              maxHeight: 500,
              overflow: 'auto',
            }}
          >
            {resultLog}
          </Paragraph>
          <div style={{ marginTop: 32, display: 'flex', justifyContent: 'space-between' }}>
            <Button type="primary" style={{ width: 268 }} onClick={rebuild}>
              {t('buildResult.button.rebuild')}
            </Button>
            <Button style={{ marginLeft: 20, width: 268 }} onClick={backToBuildConfig}>
              {t('buildResult.button.back')}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default BuildResult;
