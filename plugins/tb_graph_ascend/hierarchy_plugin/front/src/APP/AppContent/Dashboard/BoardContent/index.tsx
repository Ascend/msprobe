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
import Hierarchy from './Hierarchy';
import { GRAPH_TYPE } from '../../../../common/constant';
import { message, Splitter, type SplitterProps } from 'antd';
import useGraphStore from '../../../../store/useGraphStore';
import { useEffect } from 'react';
import styles from './index.module.less';

const BoardContent = () => {
  const setMessageApi = useGraphStore((state) => state.setMessageApi);
  const [messageApi, contextHolder] = message.useMessage();
  const isSingleGraph = useGraphStore((state) => state.isSingleGraph);
  useEffect(() => {
    setMessageApi(messageApi);
  }, []);
  const stylesObject: SplitterProps['style'] = {
    height: '100%',
    boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)',
  };

  return (
    <div className={styles.boardContentWrapper}>
      {contextHolder}
      <Splitter style={stylesObject}>
        <Splitter.Panel style={{ overflow: 'hidden' }} min="20%" max="100%">
          <Hierarchy graphType={isSingleGraph ? GRAPH_TYPE.SINGLE : GRAPH_TYPE.NPU} />
        </Splitter.Panel>
        {!isSingleGraph && (
          <Splitter.Panel collapsible style={{ overflow: 'hidden' }}>
            <Hierarchy graphType={GRAPH_TYPE.BENCH} />
          </Splitter.Panel>
        )}
      </Splitter>
    </div>
  );
};

export default BoardContent;
