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
      dir: 'Directory',
      file: 'File',
      debug: 'Debug',
      bench: 'Bench',
      accuracy_error: 'Accuracy Error',
      overflow: 'Overflow',
      precision_desc_before: 'For detailed explanations of the indicators for each color, please refer to the ',
      precision_desc_link: 'documentation',
      risk_warning: 'Please be aware of the following risks',
      risk_confirm: 'I have read and agreed',
      risk_info:
        'Unauthorized path access may lead to information leakage and file content tampering. Excessively large files or abnormal formats may cause performance issues or service disruptions. The presence of symbolic links or improper permissions in the path may pose risks of privilege escalation and data tampering.',
      risk_warning_info:
        'Proceeding with the operation will result in your own responsibility for the consequences. If you are not fully aware of the risks, please cancel the operation and contact the administrator for assistance.',
      error_title: 'error message',
      sider: {
        dataSelection: 'Data Selection',
        precisionFiltering: 'Precision Error Filtering & Overflow Detection',
        nodeMatching: 'Node Matching',
        nodeSearch: 'Node Search',
        dumpVisualization: 'Dump Data Visualization Conversion',
        switchLanguage: 'Switch Language',
      },

      boardHeader: {
        tooltips: {
          npuMiniMap: 'Debug Side Mini Map',
          benchMiniMap: 'Bench Side Mini Map',
          syncExpand: 'Sync Expand Corresponding Nodes',
          shortcuts: 'Shortcuts',
          fitScreen: 'Fit to Screen',
        },
        shortcuts: {
          zoomIn: 'Zoom In',
          zoomOut: 'Zoom Out',
          moveLeft: 'Move Left',
          moveRight: 'Move Right',
          scrollUpDown: 'Scroll Up/Down',
          scroll: 'Mouse Wheel',
        },
      },

      build_info_desc_1: 'The graph structure file was not found in the current directory',
      build_info_desc_2: 'This tool requires the use of graph structure files for visualization',
      build_info_desc_3: 'How To Start',

      build_info_main_title: 'Build a graph structure file from the model data file. ',
      build_info_sub_title: 'Supports hierarchical visual graph comparison. ',

      build_info_pytorch_link_text: 'Hierarchical Visual Graph Comparison for PyTorch Scenario',
      build_info_mindspore_link_text: 'Hierarchical Visual Graph Comparison for MindSpore Scenario',

      step1_title: 'Step 1: Collect Model Data Files',
      step1_desc:
        'The msprobe tool primarily collects precision data by adding dump interfaces in the training script and launching training.',
      pytorch_link_text: 'Data Dump Guide for PyTorch',
      mindspore_link_text: 'Data Dump Guide for MindSpore',
      step2_title: 'Step 2: Build Graph Structure File (.vis.db)',
      step2_desc:
        'If you already have model data files, you can use the build tool below to generate the graph structure file.',

      build_graph_file_title: 'Build Graph Structure File',
      build_graph_file_desc: 'Build graph structure files from model data files.',

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
      noData: 'No Data',
      comparisonDetails: 'Comparison Details',
      nodeInfo: 'Node Information',
      nodeInfoPanel: {
        copySuccessful: 'Copy successful!',
        copyFailed: 'Copy failed: ',
        stackInfo: 'StackInfo',
        parallelMergedInfo: 'ParallelMergedInfo',
        debug: 'Debug: ',
        bench: 'Bench: ',
        matchedParams: 'Matched Parameters',
        unmatchedParams: 'Unmatched Parameters',
      },
      debugNode: 'Debug node: ',
      benchNode: 'Bench node: ',
      nodeList: 'Node List({{count}})',
      matched: 'Matched',
      unmatched: 'Unmatched',
      noOverflowData: 'The current data was not generated in overflow mode',
      searchName: 'Search by Name',
      loading: 'Loading...',
      copy: 'Copy',
      matchNodes: 'Match Nodes',
      unmatchNodes: 'Unmatch Nodes',
      configFile: 'Match Configuration File',
      selectConfigFile: 'Select Match Configuration File',
      selectConfigFileTooltip:
        'Select the corresponding configuration file, read the matching node information, and match the corresponding node',
      generateConfigFile: 'Generate Match Configuration File',
      generateConfigFileTooltipBefore: 'Save the manually matched node correspondence to a configuration file named ',
      generateConfigFileTooltipAfter:
        ' in the same directory. If the configuration file does not exist, it will be created.',
      notSelected: 'Not Selected',
      file_generate_success:
        'Operation successful: The file has been generated in the current directory with the filename ',
      manageDesendant: 'Simultaneously operate descendant nodes',
      bottomOfList: 'You have reached the bottom of the list',
      topOfList: 'You have reached the top of the list',
      selectDebugMatchNode: 'Please select the debug-side unmatched node!',
      selectBenchMatchNode: 'Please select the bench-side unmatched node!',
      selectMatchedNodes: 'Please select the nodes to be unmatched!',
      matchSuccessByConfig:
        'Successfully matched nodes by the configuration file. Total number of matched nodes: {{total}}, successful number: {{success}}, failed number: {{failed}}',
      matchSuccess: 'Match successful, corresponding node status has been updated',
      unmatchSuccess: 'Unmatch successful, corresponding node status has been updated',
      updateHierarchyFailed: 'Failed to update graph data',
      emptyNodeList: 'The node list is empty!',
      singleDetails: 'Input/Output Details',
    },
  },
  zh: {
    translation: {
      dir: '目录',
      file: '文件',
      debug: '调试侧',
      bench: '标杆侧',
      accuracy_error: '精度误差',
      overflow: '溢出筛选',
      precision_desc_before: '关于各颜色指标的详细含义请参见',
      precision_desc_link: '资料说明',
      risk_warning: '请知悉以下风险',
      risk_confirm: '我已知晓并同意',
      risk_info:
        '非授权路径访问可能存在信息泄露和文件内容篡改。文件过大或格式异常，可能导致性能问题或服务中断。路径中存在软链接或权限不当，可能存在越权访问和数据篡改风险。',
      risk_warning_info: '继续操作将由您自行承担相关后果。如非明确知晓风险，请取消操作并联系管理员处理。',
      error_title: '错误信息',

      sider: {
        dataSelection: '数据选择',
        precisionFiltering: '精度误差筛选和溢出检测筛选',
        nodeMatching: '节点匹配',
        nodeSearch: '节点搜索',
        dumpVisualization: 'Dump数据可视化转化',
        switchLanguage: '语言切换',
      },

      boardHeader: {
        tooltips: {
          npuMiniMap: '调试侧小视图',
          benchMiniMap: '标杆侧小视图',
          syncExpand: '同步展开对应侧节点',
          shortcuts: '快捷键',
          fitScreen: '自适应屏幕',
        },
        shortcuts: {
          zoomIn: '放大',
          zoomOut: '缩小',
          moveLeft: '左移',
          moveRight: '右移',
          scrollUpDown: '上下滚动',
          scroll: '滚轮',
        },
      },
      build_info_desc_1: '当前目录下未找到图结构文件',
      build_info_desc_2: '本工具需要使用图结构文件进行可视化',
      build_info_desc_3: '如何开始',
      build_info_main_title: '构建图结构文件',
      build_info_sub_title: '从模型数据文件构建图结构文件。',

      build_info_pytorch_link_text: 'PyTorch场景的分级可视化构图比对',
      build_info_mindspore_link_text: 'MindSpore场景的分级可视化构图比对',

      step1_title: 'Step1: 采集模型数据文件',
      step1_desc: 'msprobe工具主要通过在训练脚本内添加dump接口、启动训练的方式采集精度数据。',
      pytorch_link_text: 'pytorch场景的数据采集指南',
      mindspore_link_text: 'mindspore场景的数据采集指南',
      step2_title: 'Step 2: 构建图结构文件(.vis.db)',
      step2_desc: '若您已拥有模型数据文件，可通过下方构建工具，构建图结构文件。',

      build_graph_file_title: '构建图结构文件',
      build_graph_file_desc: '将模型数据文件构建图结构文件。',

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
      noData: '暂无数据',
      comparisonDetails: '比对详情',
      nodeInfo: '节点信息',
      nodeInfoPanel: {
        copySuccessful: '复制成功！',
        copyFailed: '复制失败：',
        stackInfo: '堆栈信息',
        parallelMergedInfo: '数据并行合并详情',
        debug: '调试侧：',
        bench: '标杆侧：',
        matchedParams: '已匹配参数',
        unmatchedParams: '未匹配参数',
      },
      debugNode: '调试侧节点：',
      benchNode: '标杆侧节点：',
      nodeList: '节点列表（{{count}}）',
      matched: '已匹配',
      unmatched: '未匹配',
      noOverflowData: '当前数据未使用溢出检测模式生成',
      searchName: '按名称搜索',
      loading: '加载中...',
      copy: '复制',
      matchNodes: '建立匹配',
      unmatchNodes: '取消匹配',
      configFile: '匹配配置文件',
      selectConfigFile: '选择匹配配置文件',
      selectConfigFileTooltip: '选择对应配置文件，会读取匹配节点信息，并将对应节点进行匹配',
      generateConfigFile: '生成匹配配置文件',
      generateConfigFileTooltipBefore: '将手动匹配的节点对应关系保存到同级目录下名称为',
      generateConfigFileTooltipAfter: '的配置文件中，如果不存在则会创建',
      notSelected: '未选择',
      file_generate_success: '操作成功:文件已生成到当前目录下，文件名称为',
      manageDesendant: '同步操作后代节点',
      bottomOfList: '已到达列表底部',
      topOfList: '已到达列表顶部',
      selectDebugMatchNode: '请选择调试侧未匹配节点！',
      selectBenchMatchNode: '请选择标杆侧未匹配节点！',
      selectMatchedNodes: '请选择待取消匹配的节点！',
      matchSuccessByConfig:
        '通过配置文件匹配成功，匹配节点数量为{{total}}，其中成功数量为：{{success}}，失败数量为：{{failed}}',
      matchSuccess: '匹配成功，对应节点状态已更新',
      unmatchSuccess: '取消匹配成功，对应节点状态已更新',
      updateHierarchyFailed: '更新图数据失败',
      emptyNodeList: '节点列表为空！',
      singleDetails: '输入输出详情',
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
