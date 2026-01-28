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
          <Hierarchy testid="debugGraph" graphType={isSingleGraph ? GRAPH_TYPE.SINGLE : GRAPH_TYPE.NPU} />
        </Splitter.Panel>
        {!isSingleGraph && (
          <Splitter.Panel collapsible style={{ overflow: 'hidden' }}>
            <Hierarchy testid="benchGraph" graphType={GRAPH_TYPE.BENCH} />
          </Splitter.Panel>
        )}
      </Splitter>
    </div>
  );
};

export default BoardContent;
