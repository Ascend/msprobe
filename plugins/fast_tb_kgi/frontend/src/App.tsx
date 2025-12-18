/* -------------------------------------------------------------------------
  This file is part of the MindStudio project.
 Copyright (c) 2025 Huawei Technologies Co., Ltd.

 MindStudio is licensed under Mulan PSL v2.
 You can use this software according to the terms and conditions of the Mulan PSL v2.
 You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

 THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 See the Mulan PSL v2 for more details.
--------------------------------------------------------------------------------------------*/

import './App.css';

import { UploadOutlined } from '@ant-design/icons';
import {
  Button,
  Checkbox,
  ConfigProvider,
  Divider,
  Input,
  message,
  Segmented,
  Space,
  Spin,
  Splitter,
  Upload
} from 'antd';
import { useEffect, useRef, useState } from 'react';
import { Network } from 'vis-network';
import { DataSet } from 'vis-data';
import {
  ANCHOR_NODE_COLOR,
  DEFAULT_BUTTON_TEXT_COLOR,
  DEFAULT_NODE_COLOR,
  genNodesColor,
  MATCH_NODE_COLOR,
  MISMATCH_NODE_COLOR,
  SEGMENTED_COLOR,
  SELECTED_NODE_COLOR
} from './colorConfig';
import { createVisNetwork, genNodesPosition } from './visConfig';

function App() {
  const leftNetworkRef = useRef<HTMLDivElement>(null);
  const rightNetworkRef = useRef<HTMLDivElement>(null);
  const leftNodesRef = useRef<DataSet<any>>(new DataSet());
  const rightNodesRef = useRef<DataSet<any>>(new DataSet());
  const leftEdgesRef = useRef<DataSet<any>>(new DataSet());
  const rightEdgesRef = useRef<DataSet<any>>(new DataSet());

  const [messageApi, messageHolder] = message.useMessage();
  const [backendProcessing, setBackendProcessing] = useState(false);
  const [leftNetwork, setLeftNetwork] = useState<Network | null>(null);
  const [rightNetwork, setRightNetwork] = useState<Network | null>(null);
  const [compareAllMode, setCompareAllMode] = useState(false);
  const [compareModeText, setCompareModeText] = useState('');
  const [hasCompared, setHasCompared] = useState(false);
  const [ignoreLeftDataOps, setIgnoreLeftDataOps] = useState(false);
  const [ignoreRightDataOps, setIgnoreRightDataOps] = useState(false);
  const executeOrderTypeList = ['MindSpore', 'PyTorch']
  const [leftExecuteOrderType, setLeftExecuteOrderType] = useState('MindSpore');
  const [rightExecuteOrderType, setRightExecuteOrderType] = useState('MindSpore');
  const [leftHiddenEdges, setLeftHiddenEdges] = useState(true);
  const [rightHiddenEdges, setRightHiddenEdges] = useState(true);
  const [leftAnchorLineId, setLeftAnchorLineId] = useState('');
  const [rightAnchorLineId, setRightAnchorLineId] = useState('');
  const [leftFocusNodeLineId, setLeftFocusNodeLineId] = useState('');
  const [rightFocusNodeLineId, setRightFocusNodeLineId] = useState('');
  const [leftZoomScale, setLeftZoomScale] = useState('');
  const [rightZoomScale, setRightZoomScale] = useState('');
  const [leftNodeInfo, setLeftNodeInfo] = useState<string | undefined>(undefined);
  const [rightNodeInfo, setRightNodeInfo] = useState<string | undefined>(undefined);

  // 设置CSS变量
  useEffect(() => {
    const root = document.documentElement;
    root.style.setProperty('--default-node-color', DEFAULT_NODE_COLOR);
    root.style.setProperty('--selected-node-color', SELECTED_NODE_COLOR);
    root.style.setProperty('--match-node-color', MATCH_NODE_COLOR);
    root.style.setProperty('--mismatch-node-color', MISMATCH_NODE_COLOR);
    root.style.setProperty('--anchor-node-color', ANCHOR_NODE_COLOR);
  }, []);

  const sendPostRequest = async (url: string, data: any, messageInfo: string): Promise<any> => {
    try {
      setBackendProcessing(true);
      let response: any;
      if (data === null)
        response = await fetch(url);
      else {
        response = await fetch(url, data);
      }
      setBackendProcessing(false);

      if (response.ok) {
        return await response.json();
      } else {
        messageApi.error(messageInfo + '失败: ' + response.statusText);
        return null;
      }
    } catch (error) {
      messageApi.error(messageInfo + '错误: ' + error);
      setBackendProcessing(false);
      return null;
    }
  };

  const getNodeInfo = async (side: 'left' | 'right', nodeId: any) => {
    const genData = (): any => {
      return {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          side: side,
          node_id: nodeId
        })
      }
    };

    const processRes = (res: any) => {
      const infoTextLines = [];

      const lineId = res.line_id;
      infoTextLines.push(`Line ID: ${lineId}\n`);

      const opType = res.op_type;
      infoTextLines.push(`Op Type: ${opType}\n`);

      const inputMemoryNum = res.input_memory_num;
      if (inputMemoryNum) {
        infoTextLines.push(`Input Memory Number: ${inputMemoryNum}\n`);
        const inputMemoryIds = res.input_memory_ids;
        infoTextLines.push(`Input Memory IDs: `);
        inputMemoryIds.forEach((item: any) => {
          infoTextLines.push(`${item} `);
        });
        infoTextLines.push('\n');
        const inputMemoryAttrs = res.input_memory_attrs;
        infoTextLines.push(`Input Memory Attributes: `);
        inputMemoryAttrs.forEach((item: any) => {
          infoTextLines.push(`${item} `);
        });
        infoTextLines.push('\n');
      }

      const outputMemoryNum = res.output_memory_num;
      if (outputMemoryNum) {
        infoTextLines.push(`Output Memory Number: ${outputMemoryNum}\n`);
        const outputMemoryIds = res.output_memory_ids;
        infoTextLines.push(`Output Memory IDs: `);
        outputMemoryIds.forEach((item: any) => {
          infoTextLines.push(`${item} `);
        });
        infoTextLines.push('\n');
        const outputMemoryAttrs = res.output_memory_attrs;
        infoTextLines.push(`Output Memory Attributes: `);
        outputMemoryAttrs.forEach((item: any) => {
          infoTextLines.push(`${item} `);
        });
        infoTextLines.push('\n');
      }

      const scope = res.scope;
      if (scope) {
        infoTextLines.push(`Scope:\n`);
        scope.forEach((item: any) => {
          infoTextLines.push(`\t${item}\n`);
        });
      }

      const stack = res.stack;
      if (stack) {
        infoTextLines.push(`Stack:\n`);
        stack.forEach((item: any) => {
          infoTextLines.push(`\t${item}\n`);
        });
      }

      if (side === 'left') {
        setLeftNodeInfo(infoTextLines.join(''));
      } else {
        setRightNodeInfo(infoTextLines.join(''));
      }
    };

    const res = await sendPostRequest(`api/get_node_info`, genData(), '获取节点信息');
    if (!res)
      return;
    processRes(res);
  };

  // 初始化网络
  useEffect(() => {
    const network = createVisNetwork(leftNetworkRef.current, leftNodesRef.current, leftEdgesRef.current);

    const leftResizeObserver = new ResizeObserver(() => {
      if (!network)
        return;
      if (!leftNetworkRef.current)
        return;
      const scale = network.getScale();
      const position = network.getViewPosition();
      network.setSize(
        leftNetworkRef.current.clientWidth + 'px',
        leftNetworkRef.current.clientHeight + 'px'
      );
      network.moveTo({ scale, position, animation: false });
    });

    const updateLeftZoomScale = () => {
      if (!network)
        return;
      const scale = network.getScale() * 100;
      setLeftZoomScale(scale.toFixed(2));
    };

    const updateLeftAnchorLineId = (params: any) => {
      if (params.nodes.length <= 0)
        return;
      const nodeId = params.nodes[0];
      setLeftAnchorLineId(nodeId.toString());
    };

    const getLeftNodeInfo = async (params: any) => {
      if (params.nodes.length <= 0)
        return;
      const nodeId = params.nodes[0];
      await getNodeInfo('left', nodeId);
    };

    const clearLeftNodeInfo = (params: any) => {
      if (params.nodes.length === 0)
        setLeftNodeInfo(undefined);
    };

    const handleLeftClick = (params: any) => {
      updateLeftAnchorLineId(params); // 点击图中节点时，自动填入锚点行号
      getLeftNodeInfo(params); // 点击图中节点时，获取节点信息
      clearLeftNodeInfo(params); // 点击图中空白处，清除节点信息显示
    };

    leftResizeObserver.observe(leftNetworkRef.current!);
    network?.on('zoom', updateLeftZoomScale); // zoom时，自动获取缩放比例
    network?.on('click', handleLeftClick);
    setLeftNetwork(network);

    // 组件卸载时清除网络图表
    return () => {
      leftResizeObserver.disconnect();
      network?.off('zoom', updateLeftZoomScale);
      network?.off('click', handleLeftClick);
      network?.destroy();
    }
  }, []);

  useEffect(() => {
    const network = createVisNetwork(rightNetworkRef.current, rightNodesRef.current, rightEdgesRef.current);

    const rightResizeObserver = new ResizeObserver(() => {
      if (!network)
        return;
      if (!rightNetworkRef.current)
        return;
      const scale = network.getScale();
      const position = network.getViewPosition();
      network.setSize(
        rightNetworkRef.current.clientWidth + 'px',
        rightNetworkRef.current.clientHeight + 'px'
      );
      network.moveTo({ scale, position, animation: false });
    });

    const updateRightZoomScale = () => {
      if (!network)
          return;
      const scale = network.getScale() * 100;
      setRightZoomScale(scale.toFixed(2));
    };

    const updateRightAnchorLineId = (params: any) => {
      if (params.nodes.length <= 0)
        return
      const nodeId = params.nodes[0];
      setRightAnchorLineId(nodeId.toString());
    };

    const getRightNodeInfo = async (params: any) => {
      if (params.nodes.length <= 0)
        return;
      const nodeId = params.nodes[0];
      await getNodeInfo('right', nodeId);
    };

    const clearRightNodeInfo = (params: any) => {
      if (params.nodes.length === 0)
        setRightNodeInfo(undefined);
    };

    const handleRightClick = (params: any) => {
      updateRightAnchorLineId(params);
      getRightNodeInfo(params);
      clearRightNodeInfo(params);
    };

    rightResizeObserver.observe(rightNetworkRef.current!);
    network?.on('zoom', updateRightZoomScale);
    network?.on('click', handleRightClick);
    setRightNetwork(network);

    return () => {
      rightResizeObserver.disconnect();
      network?.off('zoom', updateRightZoomScale);
      network?.off('click', handleRightClick);
      network?.destroy();
    }
  }, []);

  const updateCompareMode = (compareAllMode: boolean) => {
    if (compareAllMode) {
      setCompareAllMode(true);
      setCompareModeText('比较所有模式');
    } else {
      setCompareAllMode(false);
      setCompareModeText('差异暂停模式');
    }
  };

  const getControlsInfo = async () => {
    const genData = (): any => {
      return {
        method: 'POST'
      }
    };

    const processRes = (res: any) => {
      updateCompareMode(res.compare_all_mode);
      setHasCompared(res.has_compared);
    };

    const res = await sendPostRequest(`api/get_controls_info`, genData(), '获取控制信息');
    if (!res)
      return;
    processRes(res);
  };

  useEffect(() => {
    getControlsInfo();
  }, []);

  const updateZoomScale = (network: Network | null, setZoomScale: any) => {
    if (!network)
      return;
    const scale = network.getScale() * 100;
    setZoomScale(scale.toFixed(2));
  };

  const getGraph = async (side: 'left' | 'right') => {
    const genData = (): any => {
      return {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          side: side
        })
      }
    };

    const processRes = (res: any) => {
      let nodes = genNodesPosition(res.nodes);
      nodes = genNodesColor(nodes);
      const edges = res.edges;
      if (side === 'left' && leftNetwork) {
        leftNodesRef.current.add(nodes);
        leftEdgesRef.current.add(edges);
        leftNetwork.fit();
        updateZoomScale(leftNetwork, setLeftZoomScale);
      } else if (side === 'right' && rightNetwork) {
        rightNodesRef.current.add(nodes);
        rightEdgesRef.current.add(edges);
        rightNetwork.fit();
        updateZoomScale(rightNetwork, setRightZoomScale);
      }
    };

    const res = await sendPostRequest(`api/get_graph`, genData(), '获取图');
    if (!res)
      return;
    processRes(res);
  };

  useEffect(() => {
    if (leftNetwork) {
      getGraph('left');
    }
  }, [leftNetwork]);

  useEffect(() => {
    if (rightNetwork) {
      getGraph('right');
    }
  }, [rightNetwork]);

  // 隐藏显示边
  useEffect(() => {
    if (leftNetwork) {
      leftNetwork.setOptions({
        edges: {
          hidden: leftHiddenEdges
        }
      });

      return () => {
        leftNetwork.setOptions({
          edges: {
            hidden: true
          }
        });
      };
    }
  }, [leftHiddenEdges]);

  useEffect(() => {
    if (rightNetwork) {
      rightNetwork.setOptions({
        edges: {
          hidden: rightHiddenEdges
        }
      });

      return () => {
        rightNetwork.setOptions({
          edges: {
            hidden: true
          }
        });
      };
    }
  }, [rightHiddenEdges]);

  const handleChangeCompareMode = async () => {
    const genData = (): any => {
      return {
        method: 'POST'
      };
    };

    const processRes = (res: any) => {
      if (leftNetwork) {
        let leftNodes = genNodesPosition(res.left_nodes);
        leftNodes = genNodesColor(leftNodes);
        leftNodesRef.current.update(leftNodes);
      }
      if (rightNetwork) {
        let rightNodes = genNodesPosition(res.right_nodes);
        rightNodes = genNodesColor(rightNodes);
        rightNodesRef.current.update(rightNodes);
      }
      updateCompareMode(res.compare_all_mode);
      setHasCompared(false);
    };

    const res = await sendPostRequest(`api/change_comapre_mode`, genData(), '切换模式');
    if (!res)
      return;
    processRes(res);
  };

  // 设置图
  const setGraph = async (
    file: File | null,
    side: 'left' | 'right'
  ) => {
    if (!file) {
      messageApi.error('文件上传失败: 未选择文件');
      return;
    }

    const genData = (): any => {
      const formData = new FormData();
      formData.append('side', side);
      formData.append('file', file);
      formData.append('ignore_data_ops', (side === 'left' ?
        ignoreLeftDataOps.toString() : ignoreRightDataOps.toString()));
      formData.append('execute_order_type', (side === 'left' ?
        leftExecuteOrderType.toString() : rightExecuteOrderType.toString()));
      return {
        method: 'POST',
        body: formData
      }
    };

    const processRes = (res: any) => {
      if (side === 'left' && leftNetwork) {
        const leftNodes = genNodesPosition(res.left_nodes);
        leftNodesRef.current.clear();
        leftNodesRef.current.add(leftNodes);
        const leftEdges = res.left_edges;
        leftEdgesRef.current.clear();
        leftEdgesRef.current.add(leftEdges);
        leftNetwork.fit();
        updateZoomScale(leftNetwork, setLeftZoomScale);
        let rightNodes = genNodesPosition(res.right_nodes);
        rightNodes = genNodesColor(rightNodes);
        rightNodesRef.current.update(rightNodes);
      } else if (side === 'right' && rightNetwork) {
        let leftNodes = genNodesPosition(res.left_nodes);
        leftNodes = genNodesColor(leftNodes);
        leftNodesRef.current.update(leftNodes);
        const rightNodes = genNodesPosition(res.right_nodes);
        rightNodesRef.current.clear();
        rightNodesRef.current.add(rightNodes);
        const rightEdges = res.right_edges;
        rightEdgesRef.current.clear();
        rightEdgesRef.current.add(rightEdges);
        rightNetwork.fit();
        updateZoomScale(rightNetwork, setRightZoomScale);
      }
      setHasCompared(false);
    };

    const res = await sendPostRequest(`api/set_graph`, genData(), '设置图');
    if (!res)
      return;
    processRes(res);
  };

  const handleSetLeftGraph = (file: File | null) => {
    setGraph(file, 'left');
  };

  const handleSetRightGraph = (file: File | null) => {
    setGraph(file, 'right');
  };

  const handleChangeToWholeGraph = async () => {
    const genData = (): any => {
      return {
        method: 'POST'
      };
    };

    const processRes = (res: any) => {
      if (leftNetwork) {
        const leftNodes = genNodesPosition(res.left_nodes);
        leftNodesRef.current.clear();
        leftNodesRef.current.add(leftNodes);
        const leftEdges = res.left_edges;
        leftEdgesRef.current.clear();
        leftEdgesRef.current.add(leftEdges);
        leftNetwork.fit();
        updateZoomScale(leftNetwork, setLeftZoomScale);
      }

      if (rightNetwork) {
        const rightNodes = genNodesPosition(res.right_nodes);
        rightNodesRef.current.clear();
        rightNodesRef.current.add(rightNodes);
        const rightEdges = res.right_edges;
        rightEdgesRef.current.clear();
        rightEdgesRef.current.add(rightEdges);
        rightNetwork.fit();
        updateZoomScale(rightNetwork, setRightZoomScale);
      }
      setHasCompared(false);
    };

    const res = await sendPostRequest(`api/change_to_whole_graph`, genData(), '切至整图');
    if (!res)
      return;
    processRes(res);
  };

  const setAnchorPreCheck = async (side: 'left' | 'right'): Promise<boolean> => {
    const genData = (): any => {
      const lineId = side === 'left' ? leftAnchorLineId : rightAnchorLineId;
      return {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          side: side,
          line_id: lineId
        })
      }
    };

    const processRes = (res: any): boolean => {
      const nodeExist = res.node_exist;
      if (nodeExist === true) {
        return true;
      } else {
        messageApi.error('输入行号无效');
        return false;
      }
    };

    const res = await sendPostRequest(`api/set_anchor_pre_check`, genData(), '设置锚点预检查');
    if (!res)
      return false;
    return processRes(res);
  };

  const setAnchor = async (side: 'left' | 'right') => {
    const genData = (): any => {
      const lineId = side === 'left' ? leftAnchorLineId : rightAnchorLineId;
      return {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          side: side,
          line_id: lineId
        })
      }
    };

    const processRes = (res: any) => {
      if (side === 'left' && leftNetwork) {
        let leftNodes = genNodesPosition(res.left_nodes);
        leftNodes = genNodesColor(leftNodes);
        leftNodesRef.current.clear();
        leftNodesRef.current.add(leftNodes);
        const leftEdges = res.left_edges;
        leftEdgesRef.current.clear();
        leftEdgesRef.current.add(leftEdges);
        leftNetwork.fit();
        updateZoomScale(leftNetwork, setLeftZoomScale);
        let rightNodes = genNodesPosition(res.right_nodes);
        rightNodes = genNodesColor(rightNodes);
        rightNodesRef.current.update(rightNodes);
      } else if (side === 'right' && rightNetwork) {
        let leftNodes = genNodesPosition(res.left_nodes);
        leftNodes = genNodesColor(leftNodes);
        leftNodesRef.current.update(leftNodes)
        let rightNodes = genNodesPosition(res.right_nodes);
        rightNodes = genNodesColor(rightNodes);
        rightNodesRef.current.clear();
        rightNodesRef.current.add(rightNodes);
        const rightEdges = res.right_edges;
        rightEdgesRef.current.clear();
        rightEdgesRef.current.add(rightEdges);
        rightNetwork.fit();
        updateZoomScale(rightNetwork, setRightZoomScale);
      }
      setHasCompared(false);
    };

    const res = await sendPostRequest(`api/set_anchor`, genData(), '设置锚点');
    if (!res)
      return;
    return processRes(res);
  }

  const handleSetLeftAnchor = async () => {
    if (!leftAnchorLineId) {
      messageApi.error('请输入锚点行数');
      return;
    }
    if (!await setAnchorPreCheck('left')) {
      return;
    }
    setAnchor('left');
  };

  const handleSetRightAnchor = async () => {
    if (!rightAnchorLineId) {
      messageApi.error('请输入锚点行数');
      return;
    }
    if (!await setAnchorPreCheck('right')) {
      return;
    }
    setAnchor('right');
  };

  const focusNode = (focusNodeLineId: string, network: Network) => {
    try {
      network.focus(focusNodeLineId, {
        scale: 1,
        animation: {
          duration: 100,
          easingFunction: 'easeInOutQuad'
        }
      });
      network.selectNodes([focusNodeLineId], true);
    } catch (error) {
      messageApi.error('聚焦节点失败: ' + error);
    }
  };

  const handleFocusLeftNode = () => {
    if (!leftFocusNodeLineId) {
      messageApi.error('请输入节点行数');
      return;
    }
    if (!leftNetwork) {
      messageApi.error('聚焦节点失败：左图不存在');
      return;
    }
    focusNode(leftFocusNodeLineId, leftNetwork);
  };

  const handleFocusRightNode = () => {
    if (!rightFocusNodeLineId) {
      messageApi.error('请输入节点行数');
      return;
    }
    if (!rightNetwork) {
      messageApi.error('聚焦节点失败：右图不存在');
      return;
    }
    focusNode(rightFocusNodeLineId, rightNetwork);
  };

  const handleUpCompare = async () => {
    const genData = (): any => {
      return {
        method: 'POST'
      };
    };

    const processRes = (res: any) => {
      if (leftNetwork) {
        let leftNodes = genNodesPosition(res.left_nodes);
        leftNodes = genNodesColor(leftNodes);
        leftNodesRef.current.update(leftNodes);
      }

      if (rightNetwork) {
        let rightNodes = genNodesPosition(res.right_nodes);
        rightNodes = genNodesColor(rightNodes);
        rightNodesRef.current.update(rightNodes);
      }
      setHasCompared(true);
    };

    const res = await sendPostRequest(`api/up_compare`, genData(), '向上比较');
    if (!res)
      return;
    processRes(res);
  };

  const handleDownCompare = async () => {
    const genData = (): any => {
      return {
        method: 'POST'
      };
    };

    const processRes = (res: any) => {
      if (leftNetwork) {
        let leftNodes = genNodesPosition(res.left_nodes);
        leftNodes = genNodesColor(leftNodes);
        leftNodesRef.current.update(leftNodes);
      }

      if (rightNetwork) {
        let rightNodes = genNodesPosition(res.right_nodes);
        rightNodes = genNodesColor(rightNodes);
        rightNodesRef.current.update(rightNodes);
      }
      setHasCompared(true);
    };

    const res = await sendPostRequest(`api/down_compare`, genData(), '向下比较');
    if (!res)
      return;
    processRes(res);
  };

  const replaceEqualSubgraphPreCheck = async (side: 'left' | 'right', nodesId: any): Promise<boolean> => {
    const genData = (): any => {
      return {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          side: side,
          nodes_id: nodesId
        })
      }
    };

    const processRes = (res: any): boolean => {
      const hasCycle = res.has_cycle;
      if (hasCycle) {
        if (side === 'left')
          messageApi.error('等价子图替换失败: 左图融合后存在环路');
        else
          messageApi.error('等价子图替换失败: 右图融合后存在环路');
        return false;
      } else {
        return true;
      }
    };

    const res = await sendPostRequest(`api/replace_equal_subgraph_pre_check`, genData(), '融合节点预检查');
    if (!res)
      return false;
    return processRes(res);
  };

  const handleReplaceEqualSubgraph = async () => {
    if (leftNetwork === null) {
      messageApi.error('等价子图替换失败: 左图不存在');
      return;
    }
    if (rightNetwork === null) {
      messageApi.error('等价子图替换失败: 右图不存在');
      return;
    }
    const leftNodesId = leftNetwork.getSelectedNodes();
    const rightNodesId = rightNetwork.getSelectedNodes();

    if (leftNodesId.length < 1) {
        messageApi.warning("请选择左图子图范围");
        return;
    }
    if (rightNodesId.length < 1) {
        messageApi.warning("请选择右图子图范围");
        return;
    }
    if (leftNodesId.length === 1) { // 只需融合右侧
        if (!await replaceEqualSubgraphPreCheck('right', rightNodesId)) {
            return;
        }
    } else if (rightNodesId.length === 1) { // 只需融合左侧
        if (!await replaceEqualSubgraphPreCheck('left', leftNodesId)) {
            return;
        }
    } else { // 需融合两侧
        if (!await replaceEqualSubgraphPreCheck('right', rightNodesId)) {
            return;
        }
        if (!await replaceEqualSubgraphPreCheck('left', leftNodesId)) {
            return;
        }
    }

    const genData = (): any => {
      return {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          left_nodes_id: leftNodesId,
          right_nodes_id: rightNodesId
        })
      }
    };

    const processRes = (res: any) => {
      if (leftNetwork) {
        const leftDelNodes = res.left_del_nodes;
        leftNodesRef.current.remove(leftDelNodes);
        let leftNodes = genNodesPosition(res.left_nodes);
        leftNodes = genNodesColor(leftNodes);
        leftNodesRef.current.update(leftNodes);
        const leftDelEdges = res.left_del_edges;
        leftEdgesRef.current.remove(leftDelEdges);
        const leftAddEdges = res.left_add_edges;
        leftEdgesRef.current.add(leftAddEdges);
        leftNetwork.unselectAll();
      }

      if (rightNetwork) {
        const rightDelNodes = res.right_del_nodes;
        rightNodesRef.current.remove(rightDelNodes);
        let rightNodes = genNodesPosition(res.right_nodes);
        rightNodes = genNodesColor(rightNodes);
        rightNodesRef.current.update(rightNodes);
        const rightDelEdges = res.right_del_edges;
        rightEdgesRef.current.remove(rightDelEdges);
        const rightAddEdges = res.right_add_edges;
        rightEdgesRef.current.add(rightAddEdges);
        rightNetwork.unselectAll();
      }
      setHasCompared(false);
    };

    const res = await sendPostRequest(`api/replace_equal_subgraph`, genData(), '替换等价子图');
    if (!res)
      return;
    processRes(res);
  };

  const delNodesPreCheck = async (side: 'left' | 'right', nodesId: any): Promise<boolean> => {
    const genData = (): any => {
      return {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          side: side,
          nodes_id: nodesId
        })
      }
    };

    const processRes = (res: any): boolean => {
      const hasAnchor = res.has_anchor;
      if (hasAnchor) {
        if (side === 'left')
          messageApi.error('删除节点失败: 左侧待删除节点包括锚点');
        else
          messageApi.error('删除节点失败: 右侧待删除节点包括锚点');
        return false;
      } else {
        return true;
      }
    };

    const res = await sendPostRequest(`api/del_nodes_pre_check`, genData(), '删除节点预检查');
    if (!res)
      return false;
    return processRes(res);
  };

  const handleDelNodes = async () => {
    if (leftNetwork === null) {
      messageApi.error('删除节点失败: 左图不存在');
      return;
    }
    if (rightNetwork === null) {
      messageApi.error('删除节点失败: 右图不存在');
      return;
    }
    const leftNodesId = leftNetwork.getSelectedNodes();
    const rightNodesId = rightNetwork.getSelectedNodes();

    if (leftNodesId.length < 1 && rightNodesId.length < 1) {
        messageApi.warning("请选择要删除的节点");
        return;
    }
    if (!await delNodesPreCheck('left', leftNodesId)) {
        return;
    }
    if (!await delNodesPreCheck('right', rightNodesId)) {
        return;
    }

    const genData = (): any => {
      return {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          left_nodes_id: leftNodesId,
          right_nodes_id: rightNodesId
        })
      }
    };

    const processRes = (res: any) => {
      if (leftNetwork) {
        const leftDelNodes = res.left_del_nodes;
        leftNodesRef.current.remove(leftDelNodes);
        let leftNodes = genNodesPosition(res.left_nodes);
        leftNodes = genNodesColor(leftNodes);
        leftNodesRef.current.update(leftNodes);
        const leftDelEdges = res.left_del_edges;
        leftEdgesRef.current.remove(leftDelEdges);
        leftNetwork.unselectAll();
      }

      if (rightNetwork) {
        const rightDelNodes = res.right_del_nodes;
        rightNodesRef.current.remove(rightDelNodes);
        let rightNodes = genNodesPosition(res.right_nodes);
        rightNodes = genNodesColor(rightNodes);
        rightNodesRef.current.update(rightNodes);
        const rightDelEdges = res.right_del_edges;
        rightEdgesRef.current.remove(rightDelEdges);
        rightNetwork.unselectAll();
      }
      setHasCompared(false);
    };

    const res = await sendPostRequest(`api/del_nodes`, genData(), '删除节点');
    if (!res)
      return;
    processRes(res);
  };

  const getSelectedEdges = (network: any, edgesId: any) => {
    return edgesId.map((edgeId: any) => {
        const edge = network.body.data.edges.get(edgeId);
        return [edge.from, edge.to];
    });
  }

  const handleDelEdges = async () => {
    if (leftNetwork === null) {
      messageApi.error('删除边失败: 左图不存在');
      return;
    }
    if (rightNetwork === null) {
      messageApi.error('删除边失败: 右图不存在');
      return;
    }
    const leftEdges = leftNetwork.getSelectedEdges();
    const rightEdges = rightNetwork.getSelectedEdges();

    if (leftEdges.length < 1 && rightEdges.length < 1) {
        messageApi.warning("请选择要删除的边");
        return;
    }

    const genData = (): any => {
      return {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          left_edges: getSelectedEdges(leftNetwork, leftEdges),
          right_edges: getSelectedEdges(rightNetwork, rightEdges)
        })
      }
    };

    const processRes = (res: any) => {
      if (leftNetwork) {
        const leftDelNodes = res.left_del_nodes;
        leftNodesRef.current.remove(leftDelNodes);
        let leftNodes = genNodesPosition(res.left_nodes);
        leftNodes = genNodesColor(leftNodes);
        leftNodesRef.current.update(leftNodes);
        const leftDelEdges = res.left_del_edges;
        leftEdgesRef.current.remove(leftDelEdges);
        leftNetwork.unselectAll();
      }

      if (rightNetwork) {
        const rightDelNodes = res.right_del_nodes;
        rightNodesRef.current.remove(rightDelNodes);
        let rightNodes = genNodesPosition(res.right_nodes);
        rightNodes = genNodesColor(rightNodes);
        rightNodesRef.current.update(rightNodes);
        const rightDelEdges = res.right_del_edges;
        rightEdgesRef.current.remove(rightDelEdges);
        rightNetwork.unselectAll();
      }
      setHasCompared(false);
    };

    const res = await sendPostRequest(`api/del_edges`, genData(), '删除边');
    if (!res)
      return;
    processRes(res);
  };

  const handleSetSecondLevelAnchor = async () => {
    if (leftNetwork === null) {
      messageApi.error('设置二级锚点失败: 左图不存在');
      return;
    }
    if (rightNetwork === null) {
      messageApi.error('设置二级锚点失败: 右图不存在');
      return;
    }
    const leftNodesId = leftNetwork.getSelectedNodes();
    const rightNodesId = rightNetwork.getSelectedNodes();

    if (leftNodesId.length !== 1) {
        messageApi.warning("左图请选择一个节点");
        return;
    }
    if (rightNodesId.length !== 1) {
        messageApi.warning("右图请选择一个节点");
        return;
    }

    const genData = (): any => {
      return {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          left_node_id: leftNodesId[0],
          right_node_id: rightNodesId[0],
        })
      }
    };

    const processRes = (res: any) => {
      if (leftNetwork) {
        let leftNodes = genNodesPosition(res.left_nodes);
        leftNodes = genNodesColor(leftNodes);
        leftNodesRef.current.update(leftNodes);
        leftNetwork.unselectAll();
      }

      if (rightNetwork) {
        let rightNodes = genNodesPosition(res.right_nodes);
        rightNodes = genNodesColor(rightNodes);
        rightNodesRef.current.update(rightNodes);
        rightNetwork.unselectAll();
      }
    };

    const res = await sendPostRequest(`api/set_second_level_anchor`, genData(), '设置二级锚点');
    if (!res)
      return;
    processRes(res);
  };

  const delSecondeLevelAnchorPreCheck = async (leftNodesId: any[], rightNodesId: any[]) => {
    const genData = (): any => {
      return {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          left_nodes_id: leftNodesId,
          right_nodes_id: rightNodesId,
        })
      }
    };

    const processRes = (res: any): boolean => {
      let checkRes = true;
      const leftNotAnchorNodesId = res.left_not_anchor_nodes_id;
      if (leftNotAnchorNodesId.length > 0) {
        messageApi.error("左侧图中选中节点" + leftNotAnchorNodesId + "不是二级锚点");
        checkRes = false;
      }
      const rightNotAnchorNodesId = res.right_not_anchor_nodes_id;
      if (rightNotAnchorNodesId.length > 0) {
        messageApi.error("右侧图中选中节点" + rightNotAnchorNodesId + "不是二级锚点");
        checkRes = false;
      }
      return checkRes;
    };

    const res = await sendPostRequest(`api/del_second_level_anchor_pre_check`, genData(), '删除二级锚点预检查');
    if (!res)
      return;
    return processRes(res);
  };

  const handleDelSecondLevelAnchor = async () => {
    if (leftNetwork === null) {
      messageApi.error('删除二级锚点失败: 左图不存在');
      return;
    }
    if (rightNetwork === null) {
      messageApi.error('删除二级锚点失败: 右图不存在');
      return;
    }
    const leftNodesId = leftNetwork.getSelectedNodes();
    const rightNodesId = rightNetwork.getSelectedNodes();

    if (leftNodesId.length < 1 && rightNodesId.length < 1) {
        messageApi.warning("请选择要删除的二级锚点");
        return;
    }
    if (!await delSecondeLevelAnchorPreCheck(leftNodesId, rightNodesId)) {
      return;
    }

    const genData = (): any => {
      return {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          left_nodes_id: leftNodesId,
          right_nodes_id: rightNodesId,
        })
      }
    };

    const processRes = (res: any) => {
      if (leftNetwork) {
        let leftNodes = genNodesPosition(res.left_nodes);
        leftNodes = genNodesColor(leftNodes);
        leftNodesRef.current.update(leftNodes);
        leftNetwork.unselectAll();
      }

      if (rightNetwork) {
        let rightNodes = genNodesPosition(res.right_nodes);
        rightNodes = genNodesColor(rightNodes);
        rightNodesRef.current.update(rightNodes);
        rightNetwork.unselectAll();
      }
    };

    const res = await sendPostRequest(`api/del_second_level_anchor`, genData(), '删除二级锚点');
    if (!res)
      return;
    processRes(res);
  };

  return (
    <div className="app-container">
      <ConfigProvider
        theme={{
          components: {
            Button: {
              defaultColor: DEFAULT_BUTTON_TEXT_COLOR,
              fontWeight: 'bold',
            },
            Segmented: {
              itemSelectedBg: SEGMENTED_COLOR,
              itemSelectedColor: 'white',
            },
          },
        }}
      >
        {messageHolder}
        <Spin spinning={backendProcessing} size="large" tip="处理中..." fullscreen/>
        <Splitter className="app-splitter" layout="horizontal">
          <Splitter.Panel
            className="controls-panel"
            min={365}
            max={365}
            resizable={false}
          >
            <Space
              className="controls-panel-space"
              direction='vertical'
              align='start'
              size='middle'
              split={<Divider className="controls-panel-space-divider" type="horizontal" variant="dashed" />}
            >
              <Space direction='vertical' align='start' size='small'>
                <Checkbox checked={ignoreLeftDataOps} onChange={(e) => setIgnoreLeftDataOps(e.target.checked)}>
                  忽略左图data算子
                </Checkbox>
                <div className="segmented-div">
                  <Segmented<string>
                    options={executeOrderTypeList}
                    defaultValue={leftExecuteOrderType}
                    onChange={(value) => setLeftExecuteOrderType(value)}
                    size="small"
                    shape="round"
                    block
                  />
                </div>
                <Upload
                  maxCount={1}
                  accept=".txt,.ir"
                  beforeUpload={(file) => {
                    handleSetLeftGraph(file);
                    return false;
                  }}
                >
                  <Button
                    className="set-graph-button"
                    size="small"
                    color="primary"
                    variant="solid"
                    icon={<UploadOutlined />}
                    shape="round"
                  >
                    设置左图
                  </Button>
                </Upload>
                <Checkbox checked={ignoreRightDataOps} onChange={(e) => setIgnoreRightDataOps(e.target.checked)}>
                  忽略右图data算子
                </Checkbox>
                <div className="segmented-div">
                  <Segmented<string>
                    options={executeOrderTypeList}
                    defaultValue={rightExecuteOrderType}
                    onChange={(value) => setRightExecuteOrderType(value)}
                    size="small"
                    shape="round"
                    block
                  />
                </div>
                <Upload
                  maxCount={1}
                  accept=".txt,.ir"
                  beforeUpload={(file) => {
                    handleSetRightGraph(file);
                    return false;
                  }}
                >
                  <Button
                    className="set-graph-button"
                    size="small"
                    color="primary"
                    variant="solid"
                    icon={<UploadOutlined />}
                    shape="round"
                  >
                    设置右图
                  </Button>
                </Upload>
              </Space>
              <Space>
                <Checkbox checked={leftHiddenEdges} onChange={(e) => setLeftHiddenEdges(e.target.checked)}>
                  隐藏左图边
                </Checkbox>
                <Checkbox checked={rightHiddenEdges} onChange={(e) => setRightHiddenEdges(e.target.checked)}>
                  隐藏右图边
                </Checkbox>
              </Space>
              <Space direction='vertical' align='start' size='small'>
                <Button
                  className="change-whole-graph-button"
                  size="small"
                  variant="filled"
                  shape="round"
                  onClick={handleChangeToWholeGraph}
                >
                  切换至整图
                </Button>
                <Space.Compact>
                  <Input
                    className="anchor-line-id-input"
                    value={leftAnchorLineId}
                    onChange={(e) => setLeftAnchorLineId(e.target.value)}
                    placeholder="输入行号"
                    size="small"
                  />
                  <Button
                    className="set-anchor-button"
                    size="small"
                    color="default"
                    variant="outlined"
                    onClick={handleSetLeftAnchor}
                  >
                    设置左图锚点
                  </Button>
                </Space.Compact>
                <Space.Compact>
                  <Input
                    className="anchor-line-id-input"
                    value={rightAnchorLineId}
                    onChange={(e) => setRightAnchorLineId(e.target.value)}
                    placeholder="输入行号"
                    size="small"
                  />
                  <Button
                    className="set-anchor-button"
                    size="small"
                    color="default"
                    variant="outlined"
                    onClick={handleSetRightAnchor}
                  >
                    设置右图锚点
                  </Button>
                </Space.Compact>
              </Space>
              <Space direction='vertical' align='start' size='small'>
                <Space.Compact>
                  <Input
                    className="focus-node-line-id-input"
                    placeholder="输入行号"
                    size="small"
                    value={leftFocusNodeLineId}
                    onChange={(e) => setLeftFocusNodeLineId(e.target.value)}
                  />
                  <Button
                    className="focus-node-button"
                    size="small"
                    color="default"
                    variant="outlined"
                    onClick={handleFocusLeftNode}
                  >
                    聚焦左图节点
                  </Button>
                </Space.Compact>
                <Space.Compact>
                  <Input
                    className="focus-node-line-id-input"
                    placeholder="输入行号"
                    size="small"
                    value={rightFocusNodeLineId}
                    onChange={(e) => setRightFocusNodeLineId(e.target.value)}
                  />
                  <Button
                    className="focus-node-button"
                    size="small"
                    color="default"
                    variant="outlined"
                    onClick={handleFocusRightNode}
                  >
                    聚焦右图节点
                  </Button>
                </Space.Compact>
              </Space>
              <Space direction='vertical' align='start' size='small'>
                <Space size={0}>
                  <p className='controls-panel-space-mode-text'>{compareModeText}</p>
                  <Button
                    className="change-mode-button"
                    size="small"
                    color="orange"
                    variant="solid"
                    onClick={handleChangeCompareMode}
                  >
                    切换模式
                  </Button>
                </Space>
                <Button
                  className="up-compare-button"
                  size="small"
                  color="default"
                  variant="solid"
                  shape="round"
                  onClick={handleUpCompare}
                >
                  向上比较
                </Button>
                <Button
                  className="down-compare-button"
                  size="small"
                  color="default"
                  variant="solid"
                  shape="round"
                  onClick={handleDownCompare}
                >
                  向下比较
                </Button>
                {!compareAllMode && hasCompared && (
                  <>
                    <Button
                      className="set-second-level-anchor-button"
                      size="small"
                      color="default"
                      variant="solid"
                      shape="round"
                      onClick={handleSetSecondLevelAnchor}
                    >
                      设置二级锚点
                    </Button>
                    <Button
                      className="del-second-level-anchor-button"
                      size="small"
                      color="default"
                      variant="solid"
                      shape="round"
                      onClick={handleDelSecondLevelAnchor}
                    >
                      删除二级锚点
                    </Button>
                  </>
                )}
              </Space>
              <Space direction='vertical' align='start' size='small'>
                <Button
                  className="replace-equal-subgraph-button"
                  size="small"
                  color="default"
                  variant="solid"
                  shape="round"
                  onClick={handleReplaceEqualSubgraph}
                >
                  替换等价子图
                </Button>
                <Button
                  className="del-nodes-button"
                  size="small"
                  color="default"
                  variant="solid"
                  shape="round"
                  onClick={handleDelNodes}
                >
                  删除节点
                </Button>
                <Button
                  className="del-edges-button"
                  size="small"
                  color="default"
                  variant="solid"
                  shape="round"
                  onClick={handleDelEdges}
                >
                  删除边
                </Button>
              </Space>
            </Space>
          </Splitter.Panel>
          <Splitter.Panel>
            <Splitter layout="vertical">
              <Splitter.Panel min={40} max={40} resizable={false}>
                <Space className="legends-panel-space">
                  <div className="default-node-legend-div"/>
                  <span>未比较节点</span>
                  <div className="selected-node-legend-div"/>
                  <span>选中节点</span>
                  <div className="match-node-legend-div"/>
                  <span>匹配节点</span>
                  <div className="mismatch-node-legend-div"/>
                  <span>不匹配节点</span>
                  <div className="anchor-node-legend-div"/>
                  <span>锚点</span>
                </Space>
              </Splitter.Panel>
              <Splitter.Panel>
                <Splitter layout='horizontal'>
                  <Splitter.Panel className="network-splitter-panel" defaultSize='50%' resizable={false}>
                    <div className="network-div" ref={leftNetworkRef}/>
                    <div className="zoom-scale-div">缩放比例: {leftZoomScale}%</div>
                  </Splitter.Panel>
                  <Splitter.Panel className="network-splitter-panel">
                    <div className="network-div" ref={rightNetworkRef}/>
                    <div className="zoom-scale-div">缩放比例: {rightZoomScale}%</div>
                  </Splitter.Panel>
                </Splitter>
              </Splitter.Panel>
              <Splitter.Panel min={150} max={150} resizable={false}>
                <Splitter layout='horizontal'>
                  <Splitter.Panel min='50%' max='50%' resizable={false}>
                    <Input.TextArea
                      style={{height: '100%', whiteSpace: 'pre', overflowX: 'auto'}}
                      placeholder='点击节点查看节点信息'
                      value={leftNodeInfo}
                      readOnly
                    />
                  </Splitter.Panel>
                  <Splitter.Panel>
                    <Input.TextArea
                      style={{height: '100%', whiteSpace: 'pre', overflowX: 'auto'}}
                      placeholder='点击节点查看节点信息'
                      value={rightNodeInfo}
                      readOnly
                    />
                  </Splitter.Panel>
                </Splitter>
              </Splitter.Panel>
            </Splitter>
          </Splitter.Panel>
        </Splitter>
      </ConfigProvider>
    </div>
  );
}

export default App