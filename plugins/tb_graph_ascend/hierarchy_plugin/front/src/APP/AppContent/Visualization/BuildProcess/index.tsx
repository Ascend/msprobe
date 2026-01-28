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

import { Button, Modal, Progress } from 'antd';
import styles from './index.module.less';
import { useVisualizedStore } from '../../../../store/useVisualizedStore';
import { safeJSONParse } from '../../../../common/utils';
import { useEffect, useRef, useState } from 'react';
import { BUILD_STEP } from '../../../../common/constant';
import { useTranslation } from 'react-i18next';

interface BuildProcessProps {
  setResultStatus: (buildStatus: boolean) => void;
  setResultLog: (resultLog: string) => void;
}

const BuildProcess = (props: BuildProcessProps) => {
  const { t } = useTranslation();
  const { setResultStatus, setResultLog } = props;
  const eventSourceRef = useRef<EventSource | null>(null);
  const [modal, contextHolder] = Modal.useModal();
  const convertedGraphArgs = useVisualizedStore((state) => state.convertedGraphArgs);
  const setCurrentBuildStep = useVisualizedStore((state) => state.setCurrentBuildStep);
  const [progressValue, setProgressValue] = useState(0);
  const handleCancel = () => {
    modal.info({
      title: t('cancel_build_title'),
      content: t('cancel_build_content'),
      okText: t('confirm'),
    });
  };
  useEffect(() => {
    requestGetConvertProgress();
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const requestGetConvertProgress = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    eventSourceRef.current = new EventSource(`getConvertProgress`);
    eventSourceRef.current.onmessage = (e: MessageEvent) => {
      const data = safeJSONParse(e.data);
      if (data?.status === 'building') {
        setProgressValue(data.progress);
      }
      if (data?.status === 'done') {
        eventSourceRef.current?.close();
        eventSourceRef.current = null;
        setResultStatus(true);
        setCurrentBuildStep(BUILD_STEP.BUILD_RESULT);
      }
      if (data?.status === 'error') {
        eventSourceRef.current?.close();
        eventSourceRef.current = null;
        setResultStatus(false);
        setResultLog(data.error);
        setCurrentBuildStep(BUILD_STEP.BUILD_RESULT);
      }
    };
    eventSourceRef.current.onerror = () => {
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
      setResultStatus(false);
      setResultLog(t('build_error'));
      setCurrentBuildStep(BUILD_STEP.BUILD_RESULT);
    };
  };
  return (
    <div className={styles.buildInfoContainer}>
      {contextHolder}
      <div className={styles.buildWrapper}>
        <p className={styles.progressTitle}>{t('building_graph_files')}</p>
        <Progress percent={progressValue} />
        <div className={styles.processItem} style={{ fontWeight: 700 }}>
          {t('config_info')}
        </div>
        <div className={styles.processItem}>
          {t('debug_side_path')}：<span>{convertedGraphArgs.npu_path}</span>
        </div>
        <div className={styles.processItem}>
          {t('benchmark_side_path')}：<span>{convertedGraphArgs.bench_path}</span>
        </div>
        <div className={styles.processItem}>
          {t('output_path')}：<span>{convertedGraphArgs.output_path}</span>
        </div>
        <div className={styles.processItem}>
          {t('operator_log_printing')}：
          <span>{convertedGraphArgs.is_print_compare_log ? t('enabled') : t('disabled')}</span>
        </div>
        <div className={styles.processItem}>
          {t('graph_merge_strategy')}：<span>{convertedGraphArgs.parallel_merge ? t('enabled') : t('disabled')}</span>
        </div>
        <div className={styles.processItem}>
          {t('cross_framework_mapping')}：<span>{convertedGraphArgs.layer_mapping ?? t('not_enabled')}</span>
        </div>
        <div className={styles.processItem}>
          {t('overflow_detection')}：<span>{convertedGraphArgs.overflow_check ? t('enabled') : t('disabled')}</span>
        </div>
        <div className={styles.processItem}>
          {t('fuzzy_matching')}：<span>{convertedGraphArgs.fuzzy_match ? t('enabled') : t('disabled')}</span>
        </div>
        <Button block type="primary" onClick={handleCancel}>
          {t('cancel_conversion')}
        </Button>
      </div>
    </div>
  );
};
export default BuildProcess;
