# Project Directory

Details of the project structure are as follows:

```text
MindStudio-probe
├── ccsrc                         # C/C++ source code directory
│   ├── CMakeLists.txt            # Main entry for C/C++ compilation
│   ├── aclgraph_dump            # aclgraph_dump C/C++ source code
│   ├── adump                    # adump C/C++ source code
│   └── atb_probe                # atb_probe C/C++ source code
└── cmake                     # CMake files for the C-based components
├── docs                         # Documentation directory
│   └── zh                       # Chinese documents
├── examples                   # Directory for tool configuration examples
├── output                       # Directory for generated deliverables
├── plugins                    # Entry to plugin code
│   └── tb_graph_ascend          # Entry to the tb_graph_ascend plugin code directory
├── python                      # Python source code directory
│   ├── msprobe                  # msProbe Python source code
│   │   ├── core                 # Core functional modules of the tool
│   │   ├── infer                # Inference tool module
│   │   ├── mindspore            # MindSpore module
│   │   ├── msaccucmp            # msaccucmp module
│   │   ├── overflow_check       # Overflow detection module
│   │   ├── pytorch              # PyTorch module
│   │   └── visualization        # Visualization module
├── scripts                      # Directory for storing installation, uninstallation, and upgrade scripts
├── test                         # Test code directory 
├── build.py                     # E2E packaging and build script
├── README.md                  # Repository description
├── LICENSE                       # License file
