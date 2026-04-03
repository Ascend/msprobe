# 项目目录

详细项目介绍如下：

```text
MindStudio-probe
├── ccsrc                         # C/C++源码目录
│   ├── CMakeLists.txt            # C/C++编译总入口
│   ├── aclgraph_dump            # aclgraph_dump C/C++源码
│   ├── adump                    # adump C/C++源码
│   └── atb_probe                # atb_probe C/C++源码
├── cmake                        # 存放解析C化部分cmake文件
├── docs                         # 文档目录
│   └── zh                       # 中文文档
├── examples                     # 工具配置样例存放目录
├── output                       # 交付件生成目录
├── plugins                      # 插件类代码总入口
│   └── tb_graph_ascend          # tb_graph_ascend插件代码目录入口
├── python                       # Python源码目录
│   ├── msprobe                  # msProbe Python源码
│   │   ├── core                 # 工具核心功能模块
│   │   ├── infer                # 推理工具模块
│   │   ├── mindspore            # MindSpore工具模块
│   │   ├── msaccucmp            # msaccucmp工具模块
│   │   ├── overflow_check       # 溢出检测模块
│   │   ├── pytorch              # PyTorch工具模块
│   │   └── visualization        # 可视化模块
├── scripts                      # 存放安装卸载升级脚本
├── test                         # 测试代码目录
├── setup.py                     # 端到端打包构建脚本
├── README.md                    # 整体仓代码说明
└── LICENSE                      # LICENSE文件
