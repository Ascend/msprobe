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

import { Network } from "vis-network";
import { DataSet } from 'vis-data';
import { DEFAULT_EDGE_COLOR, DEFAULT_NODE_COLOR, SELECTED_EDGE_COLOR, SELECTED_NODE_COLOR } from "./colorConfig";

const edgesOptions = {
  arrows: {
    to: {
      enabled: true,
      scaleFactor: 0.5,
      type: "arrow"
    },
    middle: {
      enabled: false
    },
    from: {
      enabled: false
    }
  },
  color: {
    color: DEFAULT_EDGE_COLOR,
    highlight: SELECTED_EDGE_COLOR,
    opacity: 0.1
  },
  hidden: true,
  physics: false,
  smooth: false,
  width: 1
};

const nodesOptions = {
  borderWidth: 0,
  color: {
    background: DEFAULT_NODE_COLOR,
    highlight: {
      background: SELECTED_NODE_COLOR
    }
  },
  opacity: 1,
  fixed: {
    x: false,
    y: false
  },
  hidden: false,
  physics: false,
  shape: 'dot',
  size: 10,
  widthConstraint: {
    minimum: 50,
    maximum: 90
  }
};

const layoutOptions = {
  improvedLayout: false,
  hierarchical: {
    enabled: false
  }
};

const interactionOptions = {
  dragNodes: true,
  dragView: true,
  hideEdgesOnDrag: true,
  hideEdgesOnZoom: true,
  hideNodesOnDrag: false,
  multiselect: true,
  selectable: true,
  selectConnectedEdges: true,
  zoomSpeed: 1,
  zoomView: true
};

const manipulationOptions = {
  enabled: false
};

const physicsOptions = {
  enabled: false
};

export const createVisNetwork = (
  container: HTMLDivElement | null,
  nodes: DataSet<any>,
  edges: DataSet<any>
): Network | null => {
  if (!container) {
    console.error('Create vis network failed! Because container is null.');
    return null;
  }

  const configureFilter = (option: string, path: Array<string>): boolean => {
    const hasOption = option === 'background';
    const inPath = path.length === 2 && path[0] === 'nodes' && path[1] === 'color';
    return hasOption && inPath;
  };

  // 开发阶段在页面中调试option，调试后生成option配置
  const configureOptions = {
    enabled: false,
    filter: configureFilter,
    container: container,
    showButton: true
  };

  const options = {
    autoResize: false,
    width: '100%',
    height: '100%',
    locale: 'en',
    clickToUse: false,
    configure: configureOptions,
    edges: edgesOptions,
    nodes: nodesOptions,
    layout: layoutOptions,
    interaction: interactionOptions,
    manipulation: manipulationOptions,
    physics: physicsOptions
  };

  const data = {
    nodes: nodes,
    edges: edges
  };

  return new Network(container, data, options);
};

const NODE_X_INTERVAL = 100;
const NODE_Y_INTERVAL = 100;
export const genNodesPosition = (nodes: any[]): any[] => {
  return nodes.map((node: any) => ({
    ...node,
    x: node.x * NODE_X_INTERVAL,
    y: node.y * -NODE_Y_INTERVAL
  }));
};