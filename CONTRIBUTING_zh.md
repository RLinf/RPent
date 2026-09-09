# 为 RPent 做贡献

[English](CONTRIBUTING.md)

感谢你关注 RPent！

RPent 欢迎社区贡献。无论是提交 bug、修复问题、完善测试和文档、优化性能，还是新增
planner、robot 或 tool 集成，都能帮助项目持续发展。

集成新功能时，可以先阅读现有的[自定义 planner](docs/source-zh/rst_source/usage/configure_planner.rst)、
[添加 robot](docs/source-zh/rst_source/development/add_robot.rst) 和
[添加 primitive](docs/source-zh/rst_source/development/add_primitive.rst) 指南。

## 1. 贡献流程

### 第 1 步：Fork 仓库并创建分支

1. 在 GitHub 上 fork [RLinf/RPent](https://github.com/RLinf/RPent)。
2. clone 自己的 fork，并将主仓库添加为 `upstream`：

   ```bash
   git clone https://github.com/<your-username>/RPent.git
   cd RPent
   git remote add upstream https://github.com/RLinf/RPent.git
   ```

3. 从最新的 `main` 创建主题分支：

   ```bash
   git fetch upstream
   git checkout -b feat/<short-description> upstream/main
   ```

分支名应体现改动类型，例如 `feat/new-planner`、`fix/tool-cancellation`、
`test/cli-contract` 或 `docs/robotwin-setup`。

### 第 2 步：搭建开发环境

```bash
# 创建并启用隔离环境。
uv venv --python 3.11
source .venv/bin/activate

# 安装源码检出和轻量测试依赖。
uv pip install -e ".[test]"

# 开发 robot 时，改为选择对应的测试环境。
uv pip install -e ".[test,libero-pro]"  # LIBERO
uv pip install -e ".[test,robocasa]"    # RoboCasa
uv pip install -e ".[test,robotwin]"    # RoboTwin

# 安装并启用仓库 hooks。
uv pip install pre-commit==4.6.2
pre-commit install
```

以上命令是不同选择；只安装当前改动涉及的 robot extra。各环境的具体配置见
[安装指南](docs/source-zh/rst_source/installation.rst)。

### 第 3 步：开发

开发时请遵守以下约定：

- **保持改动聚焦。** 新功能、无关重构和全仓库清理通常应该拆成不同的 Pull
  Request。
- **可选依赖只在使用处导入。** 例如，只在实际使用 RoboCasa 的代码中导入
  `robosuite`，不要在模块加载时导入。这样即使没有安装所有模拟器，
  `import rpent`、robot 发现和 CLI help 仍然可用。集成专用依赖应放在可选 extra
  中。
- **保持现有契约。** 如果 API 或用户可见行为发生变化，请说明变化以及用户需要如何
  调整。
- **改动应配套测试。** 新行为和 bug 修复应包含测试。refactor 应保持已有测试通过；
  如果行为发生变化，请同步更新测试并说明原因。单测必须能在普通 CPU 机器上离线
  跑通；需要模型或服务输出时使用 fake。
- **遵循项目代码风格。** pre-commit 负责 Ruff 格式化和 lint。公共 Python API 应有
  类型标注和 Google 风格 docstring，运行时信息应使用项目 logger，而不是 `print`。

### 第 4 步：运行检查并更新文档

创建 Pull Request 前，请运行与 CI 相同的必需检查：

```bash
# 格式化并检查整个仓库。
pre-commit run --all-files

# 运行全部单元测试。
pytest tests/unit_tests -v

# 开发时运行单个文件中的相关测试。
pytest tests/unit_tests/rpent/cli/test_main_contracts.py -k xxx -x
```

如果检查失败或修改了文件，请检查这些变化并重新运行。

用户可见行为发生变化时，请同步更新文档：

- 中英文文档都覆盖该功能时，保持 `docs/source-en/` 和 `docs/source-zh/` 一致；
- 公共 API 发生变化时，更新 docstring 和示例；
- 安装、快速开始或对外展示的功能发生变化时，更新 README。

### 第 5 步：提交并创建 Pull Request

Commit message 遵循
[Conventional Commits](https://www.conventionalcommits.org/)：

```text
<type>(<optional-scope>): <description>
```

常用 type 包括：

- `feat`：用户可见的新功能；
- `fix`：bug 修复；
- `docs`：仅文档改动；
- `test`：仅测试改动；
- `ci`：CI 或 workflow 改动；
- `refactor`：不改变行为的代码重构；
- `perf`：性能优化；
- `style`：不改变语义的格式调整；
- `build`：打包或构建改动；
- `chore`：维护工作。

示例：

```text
feat(planner): add an example provider adapter
fix(toolkit): retain the original handler error
test(cli): cover an offline planner run
docs(robotwin): clarify runtime setup
```

Pull Request 标题使用相同格式。scope 保持小写，description 应简短且具体。

将分支推送到自己的 fork，然后向 `RLinf/RPent:main` 创建 Pull Request。建议在
Pull Request 描述中包含：

- 问题及为什么需要该改动；
- 实现方案的简要总结；
- 存在相关 issue 时添加链接（`Fixes #123` 或 `Refs #123`）；
- 兼容性变化、迁移方式、新依赖和运行时影响；
- 实际运行的自动化检查；
- 手动完成的 GPU、模拟器、服务或真实机器人验证。

标记为可供 review 前，请先自行检查完整 diff。

## 2. 项目红线

- 不要仅仅为了让 CI 通过而弱化或删除失败的测试。
- 不要提交凭据、日志、checkpoint、下载资产或生成结果。
- 不要让必需 CI 依赖网络、模型服务、模拟器或真机。

## 3. 负责任地使用 AI

欢迎使用 AI 编码工具，但贡献者本人仍是改动的作者，并对提交的所有内容负责。

- 发布前逐行阅读并理解完整 diff。
- 亲自运行代码和相关测试。
- 删除不适合项目的生成代码、注释和抽象。
- 保持 AI 辅助的 Pull Request 聚焦、可评审，不要提交未经检查的大规模改动。
- 能够在 review 中解释设计和实现。

AI 辅助代码与其他贡献遵守相同的兼容性、文档、测试和质量要求。

## 4. 获取帮助

需要协助或有问题，可以：

- 在 [Issues](https://github.com/RLinf/RPent/issues) 中报告 bug；
- 通过微信群联系维护者，入口见 [README](README.zh-CN.md)。

感谢你帮助 RPent 变得更加可靠、实用。
