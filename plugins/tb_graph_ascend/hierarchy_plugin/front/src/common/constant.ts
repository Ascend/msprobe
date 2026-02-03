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

export enum CURRENT_TAB {
  FILE_TAB,
  PRECISION_TAB,
  MATCH_TAB,
  SEARCH_TAB,
  VISUALIZED_TAB,
}

export enum CURRENT_PAGE {
  DASHBOARD,
  VISUALIZATION,
}

export enum CURRENT_BOARD_SIDER {
  PRECISION_SIDER,
  MATCH_SIDER,
  SEARCH_SIDER,
}

export enum BUILD_STEP {
  BUILD_CONFIG = 1, // 构建配置
  BUILD_PROGRESS = 2, // 构建图
  BUILD_RESULT = 3, // 构建成功
}

export enum ControlButton {
  NPU_MAP = 'NPU_MAP',
  BENCH_MAP = 'BENCH_MAP',
  MATCH_SYNC = 'MATCH_SYNC',
  PROFILE = 'PROFILE',
  EXPAND = 'EXPAND',
}

export const INIT_TRANSFORM = { x: 36, y: 72, scale: 1.8 };

export const NPU_PREFIX = 'N___';

export const BENCH_PREFIX = 'B___';

export const UNMATCHED_COLOR = '#C7C7C7';

export const JSON_TYPE = 'json';
export const DB_TYPE = 'db';

export const MIN_GRAPH_WIDTH = 200;

export const EXPAND_MATCHED_NODE = 1;
export const DATA_COMMUNICATION = 2;

export const DATA_COMMUNICATION_TYPE = {
  send: '数据发送',
  receive: '数据接收',
  send_receive: '数据发送接收',
};

export const DATA_COMMUNICATION_ICON = {
  send: 'send',
  receive: 'receive',
  send_receive: 'send_receive',
};

export const INIT_CONVERT_GRAPH_ARGS = {
  is_print_compare_log: false,
  parallel_merge: {
    npu: {
      rank_size: 1,
      tp: 1,
      pp: 1,
      vpp: 1,
      order: 'tp-cp-ep-dp-pp',
    },
    bench: {
      rank_size: 1,
      tp: 1,
      pp: 1,
      vpp: 1,
      order: 'tp-cp-ep-dp-pp',
    },
  },
  overflow_check: false,
  fuzzy_match: false,
};

export const defaultColorSetting = [
  { key: '#FFFCF3', values: [0, 0.2] },
  { key: '#FFEDBE', values: [0.2, 0.4] },
  { key: '#FFDC7F', values: [0.4, 0.6] },
  { key: '#FFC62E', values: [0.6, 0.8] },
  { key: '#ff704d', values: [0.8, 1] },
];

export const defaultColorSelects = [{ key: 'NaN', values: [NaN, NaN] }]; // 预设颜色设置项

export enum GRAPH_TYPE {
  SINGLE = 'Single',
  NPU = 'NPU',
  BENCH = 'Bench',
}

export enum NODE_TYPE {
  MODULE = 0, // 圆角矩形，有可展开，不可展开两种情况，可展开的宽度较宽，不可展开，宽度较窄
  UNEXPANDED_NODE = 1, // 椭圆形，不可展开,API
  API_LIST = 9, // API列表
  MULTI_COLLECTION = 8, // 融合算子
}

export const DURATION_TIME = 160; // 动画时间
export const PRECIS_SELECTED_STROKE_COLOR = 'rgba(128, 150, 247, 1)'; // 选中节点颜色
export const OVERFLOW_SELECTED_STROKE_COLOR = 'rgba(243, 183, 101, 1)'; // 选中节点颜色
export const BENCH_NODE_COLOR = 'rgba(255, 255, 255, 1)'; // 基准模型节点颜色
export const BENCH_STROKE_COLOR = 'rgb(161, 161, 161)'; // 基准模型边框颜色
export const NO_MATCHED_NODE_COLOR = 'rgb(199, 199, 199)'; // 未匹配节点颜色
export const BASE_NODE_COLOR = 'rgb(255, 255, 255)'; // 基准节点颜色，没有精度信息、API、FUSION的填充色
export const STROKE_WIDTH = 1; // 边框宽度
export const SELECTED_STROKE_WIDTH = 1.5; // 边框颜色

export const MOVE_STEP = 40; // 移动步长
export const SCALE_STEP = 0.2; // 缩放步长

export const MAX_SCALE = 3; // 最大缩放
export const MIN_SCALE = 1; // 最小缩放

export enum PRECISION_ERROR_COLOR {
  pass = 'rgba(250,242,223,1)',
  warning = 'rgba(248,207,141,1)',
  error = 'rgba(243,115,112,1)',
  unmatched = '#E6E6E6',
}

export enum OVERFLOW_COLOR {
  medium = 'rgba(230, 243, 254, 1)',
  high = 'rgba(141, 197, 248, 1)',
  critical = 'rgba(22, 104, 220, 1)',
  default = 'rgb(255, 255, 255)',
}

export const NODE_TYPE_STYLES = {
  [NODE_TYPE.MODULE]: { strokeDasharray: '20,0', rx: '5', ry: '5' },
  [NODE_TYPE.UNEXPANDED_NODE]: { strokeDasharray: '20,0', rx: '50%', ry: '50%', fontSize: 6 },
  [NODE_TYPE.API_LIST]: { strokeDasharray: '15,1', rx: '5', ry: '5' },
  [NODE_TYPE.MULTI_COLLECTION]: { strokeDasharray: '2,1', rx: '5', ry: '5' },
};

export const PREFIX_MAP = {
  Single: '',
  NPU: NPU_PREFIX,
  Bench: BENCH_PREFIX,
};
