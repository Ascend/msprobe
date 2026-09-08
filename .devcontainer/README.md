# Dev Container 快速入门指南

> **真正开箱即用**：零手动配置！**首次 2 分钟** 全自动构建，**后续 10 秒** 极速开工。

## 🛠️ 极简前置准备

无需手动配置复杂工具链，请根据场景选择以下任一基础环境：

| 方案 | VS Code 安装                                                                       | Docker 服务                                                                   | 适用场景 |
| :--- |:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------| :--- |
| **远程服务器（推荐）** | [VS Code](https://code.visualstudio.com/) + `Dev Containers` + `Remote - SSH` 插件 | Linux 服务器已启用 Docker 服务                                                       | 高性能计算、释放本地资源 |
| **本地 PC** | [VS Code](https://code.visualstudio.com/) + `Dev Containers` 插件                  | [Docker Desktop](https://www.docker.com/products/docker-desktop/)（Linux 模式） | 单机离线开发 |

> ⚠️ *注意：默认配置启用了 Host 网络模式及高权限，请务必在可信环境中使用。*

## 🚀 3 步闪电开工

1. **打开项目**：在 VS Code 中打开本项目代码目录。
2. **加载容器**：点击右下角弹出的 **`Reopen in Container`** 提示（或通过 `F1` 执行同名命令）。
3. **进入开发**：待容器环境自动初始化完成后，即可直接进行编码、编译、单元测试及调试。

## 🔨 编译与单元测试

环境就绪后，通过 VS Code 菜单栏 **`Terminal`** > **`Run Task`** 即可调用预设的自动化任务：

| 任务名称 | 功能说明 |
| :--- | :--- |
| `Build: Debug Mode` | 构建 Debug 版本（仅 C++ 项目支持，Python 项目请忽略） |
| `Test: Run Unit Tests` | 执行全量单元测试 |
| `Clean: All Workspace` | 清理工作区内的所有构建缓存与临时文件 |

> *也可直接在终端执行 `python3 build.py` 命令，其功能与上述任务一致。*

## ⏱️ 自动化流程与耗时说明

启动 Dev Container 后，系统将**全自动完成以下环境配置**：

| 阶段            | 自动化任务                            | 首次耗时 | 后续启动 | 体验 |
|:--------------|:---------------------------------| :--- | :--- | :--- |
| **1. 环境拉取**   | 拉取预置镜像并部署 VS Code Server       | ~1 分钟 | 3 秒 | 全程无感 |
| **2. 身份与挂载**  | 挂载代码目录（`/workspace`）并同步 Git 权限 | ~10 秒 | 3 秒 | 全程无感 |
| **3. 工具链加载** | 并行安装 Python 插件及 Clangd 等开发工具   | ~20 秒 | 3 秒 | 开箱即用 |
| **总计**        | **零人工干预·全自动就绪**                  | **⏱️ ~2 分钟** | **⚡ ~10 秒** | **一次配置，持续高效** |

> **镜像说明**：因 MindStudio 镜像制作流程复杂且耗时，本方案**内置预构建镜像**。若需了解镜像细节，可参考 [《MindStudio 统一构建镜像制作指南》](https://gitcode.com/Ascend/msot/blob/master/docs/zh/common/docker_image_build_guide.md)。

## 💡 效率优化与故障恢复

### 1. 配置 SSH 免密登录（10 秒完成）

为避免频繁输入密码，可在 Windows PowerShell 中粘贴执行以下脚本，按提示操作即可自动完成配置：

```powershell
# 1. 交互式输入用户名和IP地址
$ip = Read-Host "请输入远程服务器的IP地址"
$user = Read-Host "请输入远程服务器的用户名"

# 2. 定义本地SSH相关路径
$sshDir = "$env:USERPROFILE\.ssh"
$pubKeyPath = "$sshDir\id_ed25519.pub"

# 3. 检查本地是否存在公钥，若不存在则自动生成
if (-not (Test-Path $pubKeyPath)) {
    Write-Host "未检测到本地公钥，正在生成 ed25519 密钥对..." -ForegroundColor Yellow
    ssh-keygen -t ed25519 -C "mindstudio_devcontainer" -f "$sshDir\id_ed25519" -N '""'
    Write-Host "密钥对生成完毕。" -ForegroundColor Green
}

# 4. 上传公钥至远程服务器
Write-Host "正在将公钥上传至 ${user}@${ip} ..." -ForegroundColor Cyan
Get-Content $pubKeyPath | ssh "${user}@${ip}" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
Write-Host "公钥上传完成，免密登录配置成功！" -ForegroundColor Green
```

### 2. 毁坏无忧：一键复原环境

若开发过程中容器环境搞乱或损坏，无需重新搭建：只需按 `F1` 键选择 **Dev Containers: Rebuild Container**，即可瞬间获得一个全新的纯净环境！

## ❓ FAQ

### 1. VS Code 远程连接卡在“Waiting for port forwarding...”？

**原因分析**：
VS Code 远程开发依赖 SSH 端口转发。若服务端 `sshd_config` 限制过严，或远程 VS Code Server 组件异常，均会导致连接挂起。

**解决方案**：

1. **检查服务端 SSH 配置**

    - 编辑 `/etc/ssh/sshd_config`（需 root 权限），确保以下参数已启用：

        ```bash
        AllowTcpForwarding yes
        GatewayPorts yes
        X11Forwarding yes
        ```

    - **关键检查**：确认不存在 `PermitOpen none` 配置，若有请注释掉（`#PermitOpen none`），否则将禁用所有端口转发。
    - 重启 SSH 服务：`sudo systemctl restart sshd`

2. **清理远程 VS Code Server**
    若配置无误仍无法连接，可能是服务端组件损坏或版本不匹配。
    - 在远程服务器执行：`rm -rf ~/.vscode-server`
    - 重新发起连接，VS Code 将自动重新部署匹配的 Server 组件。

### 2. 代码提交响应缓慢或无反馈？

**原因分析**：
项目默认启用 `pre-commit` 钩子。首次提交时需下载并初始化检查工具，耗时约 30~60 秒。后续提交将直接执行检查，响应通常为秒级。

### 3. 修改 `.vscode/settings.json` 后 `git pull` 冲突且无法更新？

**原因分析**：
为支持个性化配置，该文件被标记为 `skip-worktree`，本地修改不会显示在 `git status` 中。当远端同步更新该文件时，Git 会拒绝覆盖本地内容以防止丢失。

**解决方案**：
请使用封装命令更新代码（**注意**：此操作将以远端版本覆盖本地，请提前备份）：

```bash
git safe-pull
```

该命令会自动处理 `skip-worktree` 标记与本地暂存：拉取成功后应用远端版本并恢复标记；若拉取失败，本地修改将保留在 stash 中，确保数据安全。
