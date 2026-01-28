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
import { useState, useRef, useEffect } from 'react';
import { Empty, Typography } from 'antd';
import styles from './index.module.less';
import useGraphStore from '../../../../../../store/useGraphStore';
import type { NodeWithColor } from '../../../type';
import { useTranslation } from 'react-i18next';

interface VirtualNodeListProps {
  nodes: NodeWithColor[];
  query: string;
  // 考虑到动态计算属性calc()，因此可传入string
  height: number | string;
  // 节点前缀，标注当前是单图模式/调试侧/标杆侧
  prefix: string;
  itemHeight?: number;
  visibleItems?: number;
}

const Text = Typography.Text;

const VirtualNodeList = (props: VirtualNodeListProps): React.JSX.Element => {
  const { nodes, query, height, prefix, itemHeight = 40, visibleItems = 30 } = props;
  const selectedNode = useGraphStore((state) => state.selectedNode);
  const { t } = useTranslation();
  const setSelectedNode = useGraphStore((state) => state.setSelectedNode);
  const [scrollTop, setScrollTop] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);

  const totalHeight = nodes.length * itemHeight;

  // 计算渲染视口内元素索引
  const startIndex = Math.floor(scrollTop / itemHeight);
  const endIndex = Math.min(startIndex + visibleItems, nodes.length - 1);

  const visibleNodes = nodes.slice(startIndex, endIndex + 1);
  const offsetY = startIndex * itemHeight;

  const handleScroll = () => {
    if (listRef.current) {
      setScrollTop(listRef.current.scrollTop);
    }
  };

  // 高亮匹配的部分，需考虑文字超出时省略的样式，尤其是需要高亮文本需展示
  const highlightMatch = (node: NodeWithColor): React.JSX.Element => {
    const colorBlock = node.color ? (
      <div className={styles.colorBlock} style={{ backgroundColor: node.color }}></div>
    ) : (
      <></>
    );
    if (!query) {
      return (
        <>
          {colorBlock}
          <Text className={styles.nodeName} title={node.name}>
            {node.name}
          </Text>
        </>
      );
    }

    const lowerNode = node.name.toLowerCase();
    const lowerQueryValue = query.toLowerCase();
    const matchStartIdx = lowerNode.indexOf(lowerQueryValue);

    if (matchStartIdx === -1) {
      return <></>;
    }

    const matchEndIdx = matchStartIdx + query.length;

    return (
      <>
        {colorBlock}
        <Text className={styles.nodeName} title={node.name}>
          {node.name.substring(0, matchStartIdx)}
          <Text className={styles.matchedStr}>{node.name.substring(matchStartIdx, matchEndIdx)}</Text>
          {node.name.substring(matchEndIdx)}
        </Text>
      </>
    );
  };

  const onClick = (node: string): void => {
    // 根据选中的节点来改变背景色，需要拼接前缀
    setSelectedNode(`${prefix}${node}`);
  };

  useEffect(() => {
    // 节点列表本身没有前缀标识，因此要去除节点前缀后找索引
    const selectedIndex = nodes.findIndex((node) => node.name === selectedNode.replace(prefix, ''));
    if (selectedIndex < 0 || !listRef.current) {
      return;
    }
    const listElement = listRef.current;
    const selectedElementTop = selectedIndex * itemHeight;
    const selectedElementBottom = (selectedIndex + 1) * itemHeight;
    const visibleAreaTop = listElement.scrollTop;
    const visibleAreaBottom = visibleAreaTop + listElement.clientHeight;
    if (selectedElementTop < visibleAreaTop) {
      // 选中项在可视区域上方，滚动到选中项顶部
      listElement.scrollTo({
        top: selectedElementTop,
        behavior: 'smooth',
      });
    } else if (selectedElementBottom > visibleAreaBottom) {
      // 选中项在可视区域下方，滚动到选中项底部
      listElement.scrollTo({
        top: selectedElementBottom - listElement.clientHeight,
        behavior: 'smooth',
      });
    }
  }, [selectedNode, nodes]);

  return (
    <div ref={listRef} className={styles.virtualNodeList} style={{ height: height }} onScroll={handleScroll}>
      <div style={{ height: `${totalHeight}px` }}>
        {nodes.length <= 0 ? (
          <Empty style={{ marginTop: '36px' }} description={t('noData')} />
        ) : (
          <div className={styles.container} style={{ top: `${offsetY}px` }}>
            {visibleNodes.map((node, index) => (
              <div
                key={`${startIndex + index}-${prefix}-${node.name}`}
                className={`${prefix}${node.name}` === selectedNode ? styles.selectedItem : styles.nodeItem}
                style={{ height: `${itemHeight}px` }}
                onClick={() => onClick(node.name)}
              >
                {highlightMatch(node)}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default VirtualNodeList;
