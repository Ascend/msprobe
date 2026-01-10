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

import { Button, Modal, Progress } from "antd";
import styles from "./index.module.less";
import { useVisualizedStore } from "../../../../store/useVisualizedStore";
import { safeJSONParse } from "../../../../common/utils";
import { useEffect, useRef, useState } from "react";
import { BUILD_STEP } from "../../../../common/constant";
import { useTranslation } from "react-i18next";

interface BuildProcessProps {
  setResultStatus: (buildStatus: boolean) => void;
  setResultLog: (resultLog: string) => void;
}

const BuildProcess = (props: BuildProcessProps) => {
  const { t } = useTranslation();
  const { setResultStatus, setResultLog } = props;
  const eventSourceRef = useRef<EventSource | null>(null);
  const [modal, contextHolder] = Modal.useModal();
  const convertedGraphArgs = useVisualizedStore(
    (state) => state.convertedGraphArgs
  );
  const setCurrentBuildStep = useVisualizedStore(
    (state) => state.setCurrentBuildStep
  );
  const [progressValue, setProgressValue] = useState(0);
  const handleCancel = () => {
    modal.info({
      title: t("cancel_build_title"),
      content: t("cancel_build_content"),
      okText: t("confirm"),
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
      if (data?.status === "building") {
        setProgressValue(data.progress);
      }
      if (data?.status === "done") {
        eventSourceRef.current?.close();
        eventSourceRef.current = null;
        setResultStatus(true);
        setCurrentBuildStep(BUILD_STEP.BUILD_RESULT);
      }
      if (data?.status === "error") {
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
      setResultLog(t("build_error"));
      setCurrentBuildStep(BUILD_STEP.BUILD_RESULT);
    };
  };
  return (
    <div className={styles.buildInfoContainer}>
      {contextHolder}
      <div className={styles.buildWrapper}>
        <p className={styles.progressTitle}>{t("building_graph_files")}</p>
        <Progress percent={progressValue} />
        <div className={styles.processItem} style={{ fontWeight: 700 }}>
          {t("config_info")}
        </div>
        <div className={styles.processItem}>
          {t("debug_side_path")}：<span>{convertedGraphArgs.npu_path}</span>
        </div>
        <div className={styles.processItem}>
          {t("benchmark_side_path")}：
          <span>{convertedGraphArgs.bench_path}</span>
        </div>
        <div className={styles.processItem}>
          {t("output_path")}：<span>{convertedGraphArgs.output_path}</span>
        </div>
        <div className={styles.processItem}>
          {t("operator_log_printing")}：
          <span>
            {convertedGraphArgs.is_print_compare_log
              ? t("enabled")
              : t("disabled")}
          </span>
        </div>
        <div className={styles.processItem}>
          {t("graph_merge_strategy")}：
          <span>
            {convertedGraphArgs.parallel_merge ? t("enabled") : t("disabled")}
          </span>
        </div>
        <div className={styles.processItem}>
          {t("cross_framework_mapping")}：
          <span>{convertedGraphArgs.layer_mapping ?? t("not_enabled")}</span>
        </div>
        <div className={styles.processItem}>
          {t("overflow_detection")}：
          <span>
            {convertedGraphArgs.overflow_check ? t("enabled") : t("disabled")}
          </span>
        </div>
        <div className={styles.processItem}>
          {t("fuzzy_matching")}：
          <span>
            {convertedGraphArgs.fuzzy_match ? t("enabled") : t("disabled")}
          </span>
        </div>
        <Button block type="primary" onClick={handleCancel}>
          {t("cancel_conversion")}
        </Button>
      </div>
    </div>
  );
};
export default BuildProcess;
