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

// hooks/useHierarchyGraph.ts
import { useEffect, useRef, useState } from 'react';
import { isEmpty, throttle } from 'lodash';
import {
  BENCH_PREFIX,
  DATA_COMMUNICATION,
  EXPAND_MATCHED_NODE,
  GRAPH_TYPE,
  initTransform,
  MAX_SCALE,
  MIN_SCALE,
  MOVE_STEP,
  NODE_TYPE,
  NPU_PREFIX,
  PREFIX_MAP,
  SCALE_STEP,
} from '../../../../../common/constant';
import type { GraphType, HierarchyNodeType, HierarchyObjectType } from '../../type';
import { changeGraphPosition, getContainerTransform, parseTransform } from '../../..//../../common/utils';
import { requestChangeNodeExpandState } from '../../..//../../api/board';
import useGraphStore from '../../../../../store/useGraphStore';
import { Button, Menu, Tag, type MenuProps } from 'antd';
import { AimOutlined, NodeExpandOutlined, ShareAltOutlined, NodeCollapseOutlined } from '@ant-design/icons';
import { calcColorByPrecision, renderGraph } from './useRenderGraph';
import { resources } from '../../../../../common/i18n';

const DATA_COMMUNICATION_ICON = {
  send: <NodeExpandOutlined />,
  receive: <NodeCollapseOutlined />,
  send_receive: <ShareAltOutlined />,
};

export const useHierarchyGraph = (graphType: GRAPH_TYPE) => {
  const containerRef = useRef<SVGGElement>(null);
  const graphRef = useRef<SVGSVGElement>(null);
  const needChangeNodeCenter = useRef(true);
  const cleanEventListener = useRef<() => void>(undefined);
  const hierarchyObjectRef = useRef<HierarchyObjectType>({});
  const rootNameRef = useRef<string>('');
  // 从全局 store 获取必要状态
  const colors = useGraphStore((state) => state.colors);
  const messageApi = useGraphStore((state) => state.messageApi);
  const selectedNode = useGraphStore((state) => state.selectedNode);
  const metaFileOptions = useGraphStore((state) => state.metaFileOptions);
  const currentMetaDir = useGraphStore((state) => state.currentMetaDir);
  const currentMetaFile = useGraphStore((state) => state.currentMetaFile);
  const currentMetaRank = useGraphStore((state) => state.currentMetaRank);
  const currentMetaStep = useGraphStore((state) => state.currentMetaStep);
  const currentMetaMicroStep = useGraphStore((state) => state.currentMetaMicroStep);
  const currentMetaFileType = useGraphStore((state) => state.currentMetaFileType);
  const hightLightMatchedNode = useGraphStore((state) => state.hightLightMatchedNode);

  const setSelectedNode = useGraphStore((state) => state.setSelectedNode);
  const setCurrentMetaRank = useGraphStore((state) => state.setCurrentMetaRank);
  const setCurrentMetaFile = useGraphStore((state) => state.setCurrentMetaFile);
  const getCurrentSelection = useGraphStore((state) => state.getCurrentMetaData);
  const setHightLightMatchedNode = useGraphStore((state) => state.setHightLightMatchedNode);
  // 局部状态
  const [expanding, setExpanding] = useState(false);
  const [transform, setTransform] = useState(initTransform);
  const [contextMenuItems, setContextMenuItems] = useState<MenuProps['items']>([]); // 右键菜单

  // ======================
  // API 与状态更新
  // ======================
  const changeSelectNode = async (selectedNode: string) => {
    if (!selectedNode) {
      return;
    }
    const selectedNodeType = selectedNode.startsWith(NPU_PREFIX) ? 'NPU' : 'Bench';
    if (graphType !== 'Single' && selectedNodeType !== graphType) {
      return;
    }
    const hierarchyObject = hierarchyObjectRef.current;
    // 如果选中的节点类型和当前图类型不一致，则不处理
    const nodeName = selectedNode.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), ''); // 去掉前缀
    // 如果选中节点是当前图节点图中不存在，则展开其父节点，直到图中存在
    if (!hierarchyObject[nodeName]) {
      const nodeInfo = {
        nodeName,
        nodeType: graphType,
      };
      await changeNodeExpandState(nodeInfo);
    }
    // 是否需要居中
    if (needChangeNodeCenter.current) {
      changeNodeCenter(nodeName);
    } else {
      needChangeNodeCenter.current = true;
    }
    // 高亮匹配的节点
    if (graphType !== GRAPH_TYPE.SINGLE) {
      const matchedNodes = hierarchyObject[nodeName]?.matchedNodeLink;
      if (!matchedNodes) {
        return;
      }
      const matchedNodesName = matchedNodes[matchedNodes.length - 1];
      setHightLightMatchedNode({
        [graphType]: selectedNode,
        [graphType === GRAPH_TYPE.NPU ? GRAPH_TYPE.BENCH : GRAPH_TYPE.NPU]: matchedNodesName,
      });
    }
  };

  const changeNodeExpandState = async (nodeInfo: { nodeName: string; nodeType: GraphType }) => {
    const params = {
      nodeInfo,
      metaData: getCurrentSelection(),
    };
    setExpanding(true);
    const { success, data, error } = await requestChangeNodeExpandState<HierarchyObjectType>(params);
    setExpanding(false);
    if (success) {
      hierarchyObjectRef.current = data || {};
    } else {
      messageApi.error(error);
    }
    return { success, data };
  };

  //  当前节点居中;

  const changeNodeCenter = (nodeName: string) => {
    if (!nodeName) {
      return getContainerTransform(containerRef.current);
    }
    const hierarchyObject = hierarchyObjectRef.current;
    const nodeNameReal = nodeName?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), ''); // 去掉前缀
    const selectedNode = hierarchyObject[nodeNameReal]; // 获取当前节点
    if (!selectedNode) {
      return getContainerTransform(containerRef.current);
    }
    const transformStr = containerRef.current?.getAttribute('transform') || '';
    const initialTransform = parseTransform(transformStr); // 保存初始位置
    const clientWidth = graphRef.current?.clientWidth || 0;
    const clientHeight = graphRef.current?.clientHeight || 0;
    const root = hierarchyObject[rootNameRef.current];
    const newX = clientWidth / 2 - (root?.width * initialTransform.scale) / 2;
    const newY = clientHeight / 2 - (selectedNode?.y * initialTransform.scale + 7.5) - 100;
    updateTransform(
      containerRef.current as unknown as HTMLElement,
      {
        x: newX,
        y: newY,
        scale: initialTransform.scale,
      },
      16,
    );
    return { x: newX, y: newY, scale: initialTransform.scale };
  };

  const updateTransform = (
    container: HTMLElement,
    transform: { x: number; y: number; scale: number },
    duration = 16,
  ) => {
    setTransform(transform);
    changeGraphPosition(container, transform.x, transform.y, transform.scale, duration);
  };
  const findMatchedNodeName = (tempSelectedNode: string) => {
    let nodeName = tempSelectedNode?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '');
    const hierarchyObject = hierarchyObjectRef.current;
    let selectedNode = hierarchyObject[nodeName];
    while (isEmpty(selectedNode?.matchedNodeLink) && selectedNode?.parentNode) {
      nodeName = selectedNode.parentNode?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '');
      selectedNode = hierarchyObject[nodeName];
    }
    if (!isEmpty(selectedNode?.matchedNodeLink)) {
      let matchedNodeName = selectedNode.matchedNodeLink[selectedNode.matchedNodeLink.length - 1];
      const matchedPrefix = graphType === GRAPH_TYPE.NPU ? BENCH_PREFIX : NPU_PREFIX;
      matchedNodeName = matchedNodeName.startsWith(matchedPrefix) ? matchedNodeName : matchedPrefix + matchedNodeName; // 加上前缀
      return { matchedNodeName, selectedNode };
    } else {
      return { matchedNodeName: '', selectedNode: {} as HierarchyNodeType };
    }
  };

  // ======================
  // 绑定事件
  // ======================
  // 总绑定事件方法，管理所有事件的绑定和解绑
  const bindEventListener = (container: SVGGElement, graph: SVGSVGElement) => {
    const cleanDragEvent = bindDragEvent(container, graph);
    const cleanWheelEvent = bindWheelEvent(container, graph);
    const cleanFitScreenEvent = bindFitScreenEvent(container);
    const cleanContextMenuEvent = bindContextMenuEvent(graph);
    const cleanSelectedNodeEvent = bindSelectedNodeEvent(container);
    const cleanBindKeyboardEvent = bindKeyboardEvent(container, graph);
    const cleanChangeNodeExpandStateEvent = bindChangeNodeExpandStateEvent(container, graph);
    return () => {
      cleanDragEvent();
      cleanWheelEvent();
      cleanFitScreenEvent();
      cleanContextMenuEvent();
      cleanSelectedNodeEvent();
      cleanBindKeyboardEvent();
      cleanChangeNodeExpandStateEvent();
    };
  };

  const bindSelectedNodeEvent = (container: SVGGElement) => {
    const onSelectNodeEvent = (event: any) => {
      event.preventDefault();
      needChangeNodeCenter.current = false;
      const target: HTMLElement = event.target as HTMLElement;
      const selectedNode = target.getAttribute('name');
      if (selectedNode) {
        setSelectedNode(selectedNode);
      }
    };
    const throttleSelectNodeEvent = throttle(onSelectNodeEvent, 16);
    container.addEventListener('click', throttleSelectNodeEvent);
    return () => {
      container.removeEventListener('click', throttleSelectNodeEvent);
    };
  };

  const bindFitScreenEvent = (container: SVGGElement) => {
    const onFitScreenEvent = (event: any) => {
      event.preventDefault();
      updateTransform(container as unknown as HTMLElement, { x: 0, y: 1, scale: 1 });
    };
    document.addEventListener('fitScreen', onFitScreenEvent);
    return () => {
      document.removeEventListener('fitScreen', onFitScreenEvent);
    };
  };

  const bindChangeNodeExpandStateEvent = (container: SVGGElement, graph: SVGSVGElement) => {
    const onDoubleClickNodeEvent = async (event: any) => {
      event.preventDefault();
      let target;
      let selectedNode;
      const hierarchyObject = hierarchyObjectRef.current;
      //判断是点击展开，还是同步展开
      const isClickGraph = isEmpty(event.detail?.nodeName);

      if (isClickGraph) {
        target = event.target as HTMLElement;
        selectedNode = target.getAttribute('name');
        setSelectedNode(selectedNode || '');
      } else {
        selectedNode = event.detail.nodeName;
        setSelectedNode(selectedNode);
        selectedNode = selectedNode?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '');
        const curGraphType = event.detail.graphType;
        const originNodeExpandState = event.detail.nodeExpandState;
        const targetNodeExpandState = hierarchyObject[selectedNode]?.expand;
        //保持展开状态同步,如果一侧展开，一侧为展开，则不触发对应测的展开或者收起的操作
        if (curGraphType === graphType || originNodeExpandState === targetNodeExpandState) {
          return;
        }
      }
      const nodeName = selectedNode?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), ''); // 去掉前缀
      const nodeInfo = {
        nodeName,
        nodeType: graphType,
      };
      const nodeType = hierarchyObject[nodeInfo.nodeName || '']?.nodeType;
      if (!nodeName || nodeType === NODE_TYPE.UNEXPANDED_NODE || nodeName === rootNameRef.current) {
        return;
      }
      await changeNodeExpandState(nodeInfo);
      needChangeNodeCenter.current = true;
      // 如果是点击展开，触发同步展开事件，通知展开对应节点
      const { isSyncExpand } = useGraphStore.getState();
      if (isClickGraph && isSyncExpand && graphType !== GRAPH_TYPE.SINGLE) {
        const findRes = findMatchedNodeName(nodeName);
        const changeMatchNodeExpandState = new CustomEvent('changeMatchNodeExpandState', {
          detail: {
            nodeName: findRes.matchedNodeName, // 通知通信图展开对应节点
            nodeExpandState: findRes?.selectedNode?.expand,
            graphType: graphType,
          },
          bubbles: true, // 允许事件冒泡
        });
        container.dispatchEvent(changeMatchNodeExpandState);
      }
      changeNodeCenter(nodeName);
    };
    const onDoubleClickGraphEvent = (event: any) => {
      event.preventDefault();
    };
    const throttleDoubleClickNodeEvent = throttle(onDoubleClickNodeEvent, 16);
    container.addEventListener('dblclick', throttleDoubleClickNodeEvent); // 防止双击选中文本
    graph?.addEventListener('dblclick', onDoubleClickGraphEvent);
    document.addEventListener('changeMatchNodeExpandState', throttleDoubleClickNodeEvent);

    return () => {
      container.removeEventListener('dblclick', throttleDoubleClickNodeEvent);
      graph?.removeEventListener('dblclick', onDoubleClickGraphEvent);
      document.removeEventListener('changeMatchNodeExpandState', throttleDoubleClickNodeEvent);
    };
  };

  const bindWheelEvent = (container: SVGGElement, graph: SVGSVGElement) => {
    const onwheelEvent = (event: any) => {
      const transformStr = container?.getAttribute('transform') || '';
      const transform = parseTransform(transformStr);
      const delta = event.deltaY > 0 ? -MOVE_STEP : MOVE_STEP;
      transform.y = transform.y + delta;
      updateTransform(container as unknown as HTMLElement, transform);
    };
    const throttleWheelEvent = throttle(onwheelEvent, 16);
    graph?.addEventListener('wheel', throttleWheelEvent);
    return () => {
      graph?.removeEventListener('wheel', throttleWheelEvent);
    };
  };
  const bindDragEvent = (container: SVGGElement, graph: SVGSVGElement) => {
    let isDragging = false; // 是否正在拖拽
    let startX = 0; // 鼠标按下时的初始 X 坐标
    let startY = 0; // 鼠标按下时的初始 Y 坐标
    let initialTransform = { x: 0, y: 0, scale: 1.8 }; // 初始平移值
    const handleMouseDown = (event: any) => {
      event.preventDefault();
      isDragging = true;
      startX = event.clientX;
      startY = event.clientY;
      const transformStr = container.getAttribute('transform') || '';
      initialTransform = parseTransform(transformStr);
    };
    const handleMouseMove = (event: any) => {
      if (isDragging) {
        const dx = event.clientX - startX;
        const dy = event.clientY - startY;
        let newX = initialTransform.x + dx;
        let newY = initialTransform.y + dy;
        const scale = initialTransform.scale;
        updateTransform(container as unknown as HTMLElement, { x: newX, y: newY, scale });
      }
    };
    const handleMouseUp = () => {
      if (isDragging) {
        isDragging = false;
      }
    };
    const throttledMouseMove = throttle(handleMouseMove, 16);
    graph?.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('mousemove', throttledMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    // 返回清理函数
    return () => {
      graph?.removeEventListener('mousedown', handleMouseDown);
      document.removeEventListener('mousemove', throttledMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  };

  const bindKeyboardEvent = (container: SVGGElement, graph: SVGSVGElement) => {
    let isMouseInside = false;
    const handleMouseEnter = () => {
      isMouseInside = true;
    };

    const handleMouseLeave = () => {
      isMouseInside = false;
    };
    const handleKeyDown = (event: any) => {
      if (!isMouseInside) {
        return;
      }

      const transformStr = container.getAttribute('transform') || '';
      const transform = parseTransform(transformStr);

      switch (event.key) {
        case 'w':
        case 'W': // 放大
          transform.scale += SCALE_STEP;
          if (transform.scale > MAX_SCALE) {
            return;
          }
          break;
        case 's':
        case 'S': // 缩小
          transform.scale -= SCALE_STEP;
          if (transform.scale < MIN_SCALE) {
            return;
          }
          break;
        case 'a':
        case 'A': // 左移
          transform.x -= MOVE_STEP;
          break;
        case 'd':
        case 'D': // 右移
          transform.x += MOVE_STEP;
          break;
        default: {
          return;
        } // 如果不是指定键，则退出
      }

      // 更新图形位置
      updateTransform(container as unknown as HTMLElement, transform);
    };

    // 使用 throttle 包装键盘事件处理函数
    const throttledHandleKeyDown = throttle(handleKeyDown, 16);

    graph?.addEventListener('mouseenter', handleMouseEnter);
    graph?.addEventListener('mouseleave', handleMouseLeave);
    document.addEventListener('keydown', throttledHandleKeyDown);

    // 返回清理函数
    return () => {
      graph?.removeEventListener('mouseenter', handleMouseEnter);
      graph?.removeEventListener('mouseleave', handleMouseLeave);
      document.removeEventListener('keydown', throttledHandleKeyDown);
    };
  };

  const bindContextMenuEvent = (graph: SVGSVGElement) => {
    const onExpandMatchedNode = (selectedNode: string | undefined) => {
      const { matchedNodeName } = findMatchedNodeName(selectedNode || '');
      const matchedNodeType = graphType == GRAPH_TYPE.NPU ? GRAPH_TYPE.BENCH : GRAPH_TYPE.NPU;
      const hightLightMatchedNode = {
        [graphType]: selectedNode || undefined,
        [matchedNodeType]: matchedNodeName,
      };
      setHightLightMatchedNode(hightLightMatchedNode);
      setSelectedNode(matchedNodeName || '');
    };

    const onCommunicateNodeSelected: MenuProps['onSelect'] = (event) => {
      const { key } = event;
      const keyInfo = key.split('-');
      const rankId = Number(keyInfo[0]);
      const nodeName = keyInfo[1];
      switch (currentMetaFileType) {
        case 'db':
          if (rankId !== undefined) {
            setCurrentMetaRank(rankId);
          } else {
            messageApi.error('rankId错误');
          }
          break;
        case 'json':
          const newMetaFile = metaFileOptions[rankId]?.value;
          console.log('newMetaFile', newMetaFile);
          if (newMetaFile) {
            setCurrentMetaFile(String(newMetaFile));
          } else {
            console.log('rankId', rankId);
            messageApi.error('rankId错误');
          }
          break;
      }
      //ToDO：待优化： 等待图加载完成之后再触发，可能会存在问题
      setTimeout(() => {
        setSelectedNode(nodeName);
      }, 1000);
    };

    const onContextmenuEvent = (event: any) => {
      event.preventDefault();
      const target = event.target as HTMLElement;
      // 图外点击，不显示右键菜单
      const lang = useGraphStore.getState().currentLang;
      const translation = resources[lang].translation;
      if (target.tagName.toLowerCase() !== 'rect' && target.tagName.toLowerCase() !== 'text') {
        event.stopPropagation();
      } else {
        const contextMenuItems: MenuProps['items'] = [];
        const selectedNode = target.getAttribute('name') || '';

        if (graphType !== 'Single') {
          contextMenuItems.push({
            label: (
              <Button
                type="text"
                onClick={() => {
                  onExpandMatchedNode(selectedNode);
                }}
                icon={<AimOutlined />}
              >
                {translation.positionMatchNode}
              </Button>
            ),
            key: EXPAND_MATCHED_NODE,
          });
        }

        const nodeName = selectedNode?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '') ?? '';
        const hierarchyObject = hierarchyObjectRef.current;
        const nodeData = hierarchyObject[nodeName];
        if (!isEmpty(nodeData?.matchedDistributed)) {
          const matchedDistributed = nodeData?.matchedDistributed;
          const communicationsType = matchedDistributed?.communications_type || 'send';
          const nodeInfo = matchedDistributed?.nodes_info || {};
          const rankIds = Object.keys(nodeInfo);
          const children = rankIds.map((rankId) => {
            const communicateNode = nodeInfo[Number(rankId)];
            const precision_index = Number(communicateNode?.[0]) || 0;
            const communicateNodeName = communicateNode?.[1]; // 通信节点名称
            const prefix = PREFIX_MAP[graphType];
            const communicateNodeLabel = `${prefix}${communicateNodeName}`;
            const { colors } = useGraphStore.getState(); // 不要用外层的color，闭包会导致拿不到最新的数据
            const communicateNodePrecisionColor = calcColorByPrecision(precision_index, colors);
            return {
              key: `${rankId}-${communicateNodeLabel}`,
              label: (
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <Tag key={rankId} color={communicateNodePrecisionColor} style={{ width: 30, height: 15 }}></Tag>
                  {`rank${rankId}`}
                </div>
              ),
            };
          });
          const dataCommunication = translation.dataCommunication;
          const options: Required<MenuProps>['items'] = [
            {
              key: communicationsType,
              icon: DATA_COMMUNICATION_ICON[communicationsType],
              label: dataCommunication[communicationsType],
              children,
            },
          ];

          const menuItem = {
            label: <Menu items={options} onSelect={onCommunicateNodeSelected} />,
            key: DATA_COMMUNICATION,
          };
          contextMenuItems.push(menuItem);
        }

        setContextMenuItems(contextMenuItems);
        setSelectedNode(selectedNode || '');
        needChangeNodeCenter.current = false; // 点击不需要改变中心节点
      }
    };
    const throttleContextMenuEvent = throttle(onContextmenuEvent, 16);
    graph?.addEventListener('contextmenu', throttleContextMenuEvent);
    return () => {
      graph?.removeEventListener('contextmenu', throttleContextMenuEvent);
    };
  };

  // 初始化
  const initHierarchy = async (selection: any) => {
    if (isEmpty(selection) || !graphType) return;
    const { success, data } = await changeNodeExpandState({ nodeName: 'root', nodeType: graphType });
    if (success && !isEmpty(data)) {
      // 清空container下面的所有子元素
      cleanEventListener.current?.();
      if (containerRef.current && graphRef.current) {
        setSelectedNode(''); // 初始化时，清空选中节点
        containerRef.current.innerHTML = '';
        cleanEventListener.current = bindEventListener(containerRef.current, graphRef.current);
        rootNameRef.current = Object.keys(data)[0];
      }
    }
  };

  // 节点卸载时，清空监听事件，避免内存泄漏
  useEffect(() => {
    return () => {
      cleanEventListener.current?.();
    };
  }, []);

  // 监听小视图更新transform，大视图同步更新
  useEffect(() => {
    renderGraph(hierarchyObjectRef.current, selectedNode, transform, containerRef.current, graphType, colors);
  }, [transform]);

  // 图节点更新，自动重回
  useEffect(() => {
    const hierarchyObject = hierarchyObjectRef.current;
    if (!isEmpty(hierarchyObject)) {
      changeSelectNode(selectedNode);
      renderGraph(hierarchyObject, selectedNode, transform, containerRef.current, graphType, colors);
    }
  }, [hierarchyObjectRef.current, colors, selectedNode, graphType]);
  // 高亮匹配节点
  useEffect(() => {
    const hierarchyObject = hierarchyObjectRef.current;
    const hightLightNodeName = hightLightMatchedNode[graphType];
    renderGraph(hierarchyObject, hightLightNodeName || '', transform, containerRef.current, graphType, colors);
  }, [hightLightMatchedNode, graphType]);

  // 切换文件或者目录等，重新加载图
  useEffect(() => {
    if (!currentMetaDir || !currentMetaFile || currentMetaRank == undefined || currentMetaStep === undefined) return;
    setTimeout(() => {
      setTransform(initTransform);
      initHierarchy(getCurrentSelection());
      updateTransform(containerRef.current as unknown as HTMLElement, initTransform);
    }, 100); // 延迟初始化，等待s数据加载完成
  }, [currentMetaFile, currentMetaRank, currentMetaStep, currentMetaMicroStep, graphType]);

  return {
    graphRef,
    expanding,
    transform,
    containerRef,
    contextMenuItems,
    hierarchyObjectRef,
    setTransform,
  };
};
