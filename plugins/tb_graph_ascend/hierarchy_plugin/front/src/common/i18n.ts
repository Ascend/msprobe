/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License"),
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, softwares
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import i18next from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
// 自定义 language detector 选项：优先从 localStorage 读取
const languageDetectorOptions = {
  // 检测顺序
  order: ['localStorage', 'navigator'],
  // 缓存用户选择到 localStorage
  caches: ['localStorage'],
  // localStorage 的 key 名
  localStorageKey: 'i18nextLng',
};

export const resources = {
  en: {
    translation: {
      risk_warning: 'Please be aware of the following risks',
      risk_confirm: 'I have read and agreed',
      risk_info:
        'Unauthorized path access may lead to information leakage and file content tampering. Excessively large files or abnormal formats may cause performance issues or service disruptions. The presence of symbolic links or improper permissions in the path may pose risks of privilege escalation and data tampering.',
      risk_warning_info:
        'Proceeding with the operation will result in your own responsibility for the consequences. If you are not fully aware of the risks, please cancel the operation and contact the administrator for assistance.',
      error_title: 'error message',

      build_info_desc_1: 'The graph structure file was not found in the current directory',
      build_info_desc_2: 'This tool requires the use of graph structure files for visualization',
      build_info_desc_3: 'How To Start',

      build_info_main_title: 'Build a graph structure file from the model data file. ',
      build_info_sub_title: 'Supports hierarchical visual graph comparison. ',
      build_info_sub_title_link_text: 'View Guide',
      step1_title: 'Step 1: Collect Model Data Files',
      step1_desc:
        'The msprobe tool primarily collects precision data by adding dump interfaces in the training script and launching training.',
      step1_link_text: 'Data Collection Guide for PyTorch',

      step2_title: 'Step 2: Build Graph Structure File (.vis.db)',
      step2_desc:
        'If you already have model data files, you can use the build tool below to generate the graph structure file.',

      build_graph_file_title: 'Build Graph Structure File',
      build_graph_file_desc: 'Build graph structure files from model data files.',
      view_guide: 'View Guide',

      file_param_config: 'File Parameter Configuration',
      label_npu_path: 'Debug-side Comparison Path (-tp)',
      label_bench_path: 'Benchmark-side Comparison Path (-gp)',
      label_output_path: 'Output Result File Path (-o)',
      checkbox_print_log: 'Enable per-operator log printing (--is_print_compare_log)',
      checkbox_parallel_merge: 'Enable graph merging under different partitioning strategies',

      debug_side: 'Debug Side',
      benchmark_side: 'Benchmark Side',
      label_rank_size_npu: 'Number of Accelerator Cards (Debug Side) (--rank_size)',
      label_tp_npu: 'Tensor Parallel Size (Debug Side) (--tp)',
      label_pp_npu: 'Pipeline Parallel Stages (Debug Side) (--pp)',
      label_vpp_npu: 'Virtual Pipeline Parallel Stages (Debug Side) (--vpp)',
      label_order_npu: 'Model Parallel Dimension Order (Debug Side) (--order)',

      label_rank_size_bench: 'Number of Accelerator Cards (Benchmark Side) (--rank_size)',
      label_tp_bench: 'Tensor Parallel Size (Benchmark Side) (--tp)',
      label_pp_bench: 'Pipeline Parallel Stages (Benchmark Side) (--pp)',
      label_vpp_bench: 'Virtual Pipeline Parallel Stages (Benchmark Side) (--vpp)',
      label_order_bench: 'Model Parallel Dimension Order (Benchmark Side) (--order)',

      more_options: 'More Options',
      label_layer_mapping: 'Cross-framework Mapping (--layer_mapping)',
      checkbox_overflow_check: 'Overflow Detection Mode (--overflow_check)',
      checkbox_fuzzy_match: 'Fuzzy Matching (--fuzzy_match)',

      placeholder_select: 'Please select',
      placeholder_input: 'Please enter',
      button_start_conversion: 'Start Conversion',
      cancel_build_title: 'Cancel Build',
      cancel_build_content:
        'After cancellation, the build progress will not be saved. If you wish to cancel, please manually press Ctrl + C on the server-side terminal to terminate the process.',

      building_graph_files: 'Building graph structure files...',
      config_info: 'Configuration',
      training_framework: 'Training Framework',
      debug_side_path: 'Debug-side Comparison Path',
      benchmark_side_path: 'Benchmark-side Comparison Path',
      output_path: 'Output Path',
      operator_log_printing: 'Per-operator Log Printing',
      graph_merge_strategy: 'Graph Merging Under Different Partitioning Strategies',
      cross_framework_mapping: 'Cross-framework Mapping',
      overflow_detection: 'Overflow Detection Mode',
      fuzzy_matching: 'Fuzzy Matching',
      expand_matched_node: 'Positioning Matching Node',
      enabled: 'Enabled',
      disabled: 'Disabled',
      not_enabled: 'Not Enabled',
      cancel_conversion: 'Cancel Conversion',
      zoomIn: 'Zoom In',
      zoomOut: 'Zoom Out',
      moveLeft: 'Move Left',
      moveRight: 'Move Right',
      scrollUpDown: 'Move Up/Down',
      scroll: 'Scroll Up/Down',
      legend: {
        moduleOrOperators: 'Module or Operators',
        unexpandedNodes: 'Unexpanded Nodes',
        apiList: 'API List',
        multiCollection: 'Multi Collection',
      },
      tooltip: {
        unexpandedNodes:
          'Non-expandable node: It can be an API, operator, or module. Since it contains no child nodes, it cannot be expanded.',
        apiList: 'A collection of standalone APIs between modules.',
        fusionNode: 'Fused operator collection.',
      },
      buildResult: {
        success: {
          title: 'Build Succeeded!',
          message: 'Graph structure file generated successfully. Output directory: {{outputPath}}',
        },
        failure: {
          title: 'Build Failed!',
          message: 'Failed to generate graph structure file.',
          logTitle: 'Error Log',
        },
        button: {
          loadFile: 'Load File',
          rebuild: 'Rebuild',
          back: 'Back',
        },
      },
      dashboard: {
        loading: {
          default: 'Loading',
          graphData: 'Loading graph data, please wait...',
          graphConfig: 'Loading graph configuration, please wait...',
          fileProgress: 'File size: {{size}}, Read: {{read}}, Progress: {{percent}}%',
        },
        error: {
          loadGraphDataFailed: 'Failed to load graph data',
          loadGraphConfigFailed: 'Failed to load graph configuration',
        },
      },
      dataCommunication: {
        send: 'data send',
        receive: 'data receive',
        send_receive: 'data send/receive',
      },
      positionMatchNode: 'Positioning matched node',
    },
  },
  zh: {
    translation: {
      risk_warning: '请知悉以下风险',
      risk_confirm: '我已知晓并同意',
      risk_info:
        '非授权路径访问可能存在信息泄露和文件内容篡改。文件过大或格式异常，可能导致性能问题或服务中断。路径中存在软链接或权限不当，可能存在越权访问和数据篡改风险。',
      risk_warning_info: '继续操作将由您自行承担相关后果。如非明确知晓风险，请取消操作并联系管理员处理。',
      error_title: '错误信息',

      build_info_desc_1: '当前目录下未找到图结构文件',
      build_info_desc_2: '本工具需要使用图结构文件进行可视化',
      build_info_desc_3: '如何开始',
      build_info_main_title: '构建图结构文件',
      build_info_sub_title: '从模型数据文件构建图结构文件。',
      build_info_sub_title_link_text: '查看指南',
      step1_title: 'Step1: 采集模型数据文件',
      step1_desc: 'msprobe工具主要通过在训练脚本内添加dump接口、启动训练的方式采集精度数据。',
      step1_link_text: 'pytorch场景的数据采集指南',

      step2_title: 'Step 2: 构建图结构文件(.vis.db)',
      step2_desc: '若您已拥有模型数据文件，可通过下方构建工具，构建图结构文件。',

      build_graph_file_title: '构建图结构文件',
      build_graph_file_desc: '将模型数据文件构建图结构文件。',
      view_guide: '查看指南',

      file_param_config: '文件参数配置',
      label_npu_path: '调试侧比对路径 (-tp)',
      label_bench_path: '标杆侧比对路径 (-gp)',
      label_output_path: '输出结果文件路径 (-o)',
      checkbox_print_log: '开启单个算子的日志打屏 (--is_print_compare_log)',
      checkbox_parallel_merge: '开启不同切分策略下的图合并',

      debug_side: '调试侧',
      benchmark_side: '标杆侧',
      label_rank_size_npu: '调试侧加速卡数量 (--rank_size)',
      label_tp_npu: '调试侧张量运行大小 (--tp)',
      label_pp_npu: '调试侧流水线并行的阶段数 (--pp)',
      label_vpp_npu: '调试侧虚拟流水线并行阶段数 (--vpp)',
      label_order_npu: '调试侧模型并行维度的排序顺序 (--order)',

      label_rank_size_bench: '标杆侧加速卡数量 (--rank_size)',
      label_tp_bench: '标杆侧张量运行大小 (--tp)',
      label_pp_bench: '标杆侧流水线并行的阶段数 (--pp)',
      label_vpp_bench: '标杆侧虚拟流水线并行阶段数 (--vpp)',
      label_order_bench: '标杆侧模型并行维度的排序顺序 (--order)',

      more_options: '更多选项',
      label_layer_mapping: '跨框架比对 (--layer_mapping)',
      checkbox_overflow_check: '溢出检测模式 (--overflow_check)',
      checkbox_fuzzy_match: '模糊匹配 (--fuzzy_match)',

      placeholder_select: '请选择',
      placeholder_input: '请输入',
      button_start_conversion: '开始转换',

      cancel_build_title: '取消构建',
      cancel_build_content: '取消后，构建进度将无法保留，如需取消，请在服务端键盘手动输入Ctrl + C 终止进程',

      building_graph_files: '正在构建图结构文件...',
      config_info: '配置信息',
      training_framework: '训练框架',
      debug_side_path: '调试侧比对路径',
      benchmark_side_path: '标杆侧比对路径',
      output_path: '输出路径',
      operator_log_printing: '单个算子的日志打屏',
      graph_merge_strategy: '不同切分策略下的图合并',
      cross_framework_mapping: '跨框架比对',
      overflow_detection: '溢出检测模式',
      fuzzy_matching: '模糊匹配',
      enabled: '已开启',
      disabled: '未开启',
      not_enabled: '未开启',
      cancel_conversion: '取消转换',
      expand_matched_node: '定位对应侧节点',
      zoomIn: '放大',
      zoomOut: '缩小',
      moveLeft: '左移',
      moveRight: '右移',
      scrollUpDown: '上下',
      scroll: '滚轮上下',
      legend: {
        moduleOrOperators: '模块或算子', // 可保留英文或写“模块或算子”
        unexpandedNodes: '不可扩展节点',
        apiList: '游离 API 列表', // 或 “游离 API 列表”
        multiCollection: '融合算子', // 或 “融合集合”
      },
      tooltip: {
        unexpandedNodes: '不可扩展节点：它可以是API、运算符或模块。由于其不包含子节点，因此无法展开',
        apiList: '模块之间游离API的集合',
        fusionNode: '融合算子集合',
      },
      buildResult: {
        success: {
          title: '构建成功！',
          message: '已成功生成图结构文件，文件所在目录为：{{outputPath}}',
        },
        failure: {
          title: '构建失败！',
          message: '图结构文件构建失败',
          logTitle: '异常日志',
        },
        button: {
          loadFile: '加载该文件',
          rebuild: '重新构建',
          back: '返回',
        },
      },
      dashboard: {
        loading: {
          default: '加载中',
          graphData: '正在加载图数据，请稍后......',
          graphConfig: '正在加载图配置，请稍后......',
          fileProgress: '文件大小: {{size}}, 已读取: {{read}}, 当前进度：{{percent}}%',
        },
        error: {
          loadGraphDataFailed: '加载图数据失败',
          loadGraphConfigFailed: '加载图配置失败',
        },
      },
      dataCommunication: {
        send: '数据发送',
        receive: '数据接收',
        send_receive: '数据发送接收',
      },
      positionMatchNode: '定位对应侧节点',
    },
  },
};

i18next
  .use(LanguageDetector)
  // 将i18n实例绑定到React
  .use(initReactI18next)
  .init({
    fallbackLng: 'zh',
    resources,
    detection: languageDetectorOptions,
    debug: false,
    interpolation: {
      escapeValue: false,
    },
  });

export default i18next;
