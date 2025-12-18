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

export const DEFAULT_NODE_COLOR = '#4A90E2FF';
export const SELECTED_NODE_COLOR = '#FAAD14FF';
export const MATCH_NODE_COLOR = '#52C41AFF';
export const MISMATCH_NODE_COLOR = '#FF4D4FFF';
export const ANCHOR_NODE_COLOR = '#722ED1FF';
export const DEFAULT_EDGE_COLOR = '#BDC0C2FF';
export const SELECTED_EDGE_COLOR = '#000000FF';
export const DEFAULT_BUTTON_TEXT_COLOR = '#176FDCFF';
export const SEGMENTED_COLOR = '#FA8C16FF';

export const genNodesColor = (nodes: any[]): any[] => {
  return nodes.map((node: any) => {
    if (node.is_anchor === true)
      return {
        ...node,
        color: {background: ANCHOR_NODE_COLOR}
      }
    if (node.mismatch === true)
      return {
        ...node,
        color: {background: MISMATCH_NODE_COLOR}
      }
    if (node.match === true)
      return {
        ...node,
        color: {background: MATCH_NODE_COLOR}
      }
    return {
      ...node,
      color: {background: DEFAULT_NODE_COLOR}
    }
  });
};