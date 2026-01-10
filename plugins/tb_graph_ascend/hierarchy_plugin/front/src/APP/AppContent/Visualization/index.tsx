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
