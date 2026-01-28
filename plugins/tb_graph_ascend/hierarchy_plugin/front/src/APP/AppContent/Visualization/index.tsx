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
import { useState } from 'react';
import { BUILD_STEP } from '../../../common/constant';
import { useVisualizedStore } from '../../../store/useVisualizedStore';
import BuildInfo from './BuildInfo';
import BuildProcess from './BuildProcess';
import BuildResult from './BuildResult';

const Visualization = () => {
  const currentBuildStep = useVisualizedStore((state) => state.currentBuildStep);
  const [resultStatus, setResultStatus] = useState(false);
  const [resultLog, setResultLog] = useState('');
  return (
    <div style={{ height: '100vh' }}>
      {currentBuildStep === BUILD_STEP.BUILD_CONFIG && <BuildInfo />}
      {currentBuildStep === BUILD_STEP.BUILD_PROGRESS && (
        <BuildProcess setResultStatus={setResultStatus} setResultLog={setResultLog} />
      )}
      {currentBuildStep === BUILD_STEP.BUILD_RESULT && (
        <BuildResult resultStatus={resultStatus} resultLog={resultLog} />
      )}
    </div>
  );
};
export default Visualization;
