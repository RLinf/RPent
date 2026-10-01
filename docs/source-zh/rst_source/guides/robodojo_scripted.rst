RoboDojo 脚本化规控
========================================

这条独立执行路径将可信 Python 配方交给独占的 RoboDojo 开发环境服务执行，
由服务端 CuRobo 提供 IK。入口是在源码仓库中运行的 Python 模块，不注册到
planner CLI 或 Dashboard，也不启动 VLA 服务。

安装
----

仿真环境需要 Linux、Python 3.11 和兼容的 NVIDIA 驱动。
在 RPent 仓库根目录创建专用环境：

.. code-block:: bash

   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[robodojo-sim]" --extra-index-url https://pypi.nvidia.com

该 extra 从 Git 安装 RLinf 环境适配器和 ``rlinf-robodojo-runtime``。
运行时包需为 0.3.0 或更新版本，由它提供仿真桥、随包分发的 RoboDojo 源码，
并管理 Isaac Sim / IsaacLab 的依赖版本。RPent 保留 cuRobo 的 Git 引用和
uv 构建依赖配置，因此需要从仓库根目录运行 uv。不同机器人栈应使用独立环境。
Git 分支引用可能变化，请随评测结果保存实际解析到的提交和安装版本。

依赖覆盖用于协调仿真器元数据与 RPent、RLinf 的要求；解析或导入成功不代表
仿真兼容性已经验证。``uv pip check`` 仍会报告被显式覆盖的 IsaacLab Starlette
约束，以及 Isaac Sim 的 typing-extensions、uvicorn、wrapt 约束。
这些上游元数据冲突尚未消除，GPU 兼容性仍需另行验证。
当前 IsaacLab wheel 遗漏了包初始化时必读的
``config/extension.toml``。启动服务前，请将已解析到的同一版本改为 editable
安装，并在使用该环境期间保留源码目录。以下命令从 Git 安装元数据中读取版本，
需要在替换安装之前执行：

.. code-block:: bash

   ISAACLAB_REV=$(python -c 'import importlib.metadata as m, json; print(json.loads(m.distribution("isaaclab").read_text("direct_url.json"))["vcs_info"]["commit_id"])')
   git clone --branch main --single-branch https://github.com/yuechen0614/IsaacLab.git /path/to/IsaacLab
   git -C /path/to/IsaacLab checkout --detach "$ISAACLAB_REV"
   uv pip install --no-deps \
     -e /path/to/IsaacLab/source/isaaclab \
     -e /path/to/IsaacLab/source/isaaclab_assets \
     -e /path/to/IsaacLab/source/isaaclab_tasks

这里的 ``--no-deps`` 用于保留 extra 已解析的依赖，不能代替前面的完整仿真栈安装。

场景资产需单独准备。从
`RoboDojo 数据集 <https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo>`_
下载 ``Assets/**``，将 ``ROBODOJO_ASSETS_ROOT`` 设为包含 ``Assets/`` 的目录。
确认机器人、物体、材质和保存布局文件均存在，而非 LFS 指针。
IsaacLab 引用的 NVIDIA USD/材质资产也需按仿真器说明准备。离线使用时，
``ROBODOJO_USD_ASSET_PREFIX`` 用于 RoboDojo 的 ``Assets/Isaac/5.0`` URL
重写，必须对应匹配的资产包；其他 IsaacLab 资产 URL 需要另行配置。

启动独占 dev 服务
-----------------

在专用终端中，从仓库根目录运行：

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/scene-data
   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000
   python -m robots.robodojo.env_server \
     --task general_pickup --layout 0 --mode dev \
     --max-episode-steps 200 --headless --enable_cameras \
     --host localhost --port 18765 --transport http \
     --save-dir /path/to/server-output --video-dir /path/to/videos

默认源码目录来自 ``robodojo_runtime.source_root()``。只有使用其他源码树时才
传入 ``--source-root``，后续 freeze 必须选择同一目录。
settling 环境变量会改变初始化阶段的物理步数；历史成功记录使用 1000 步，
报告结果时应披露这个设置。

等待服务就绪，并保持独占：不要同时连接 Dashboard、第二个配方或其他动作客户端。
使用固定布局，不传 ``--random``。runner 要求当前步数为零，且 task、layout、
step limit 完全匹配，然后执行一次 reset。runner 不验证服务实际加载的源码，
不强制实施独占，也不负责启动、停止或重新配置服务。
评测结束后由操作者停止服务，以完成各相机视频的写入。

freeze、verify、run
-------------------

先编写并审阅一个定义了 ``main(env, output)`` 的 Python 配方。
下面的 recipe 路径指向你自己的文件，并非仓库自带的成功策略。
在使用同一环境的第二个终端中，从仓库根目录执行：

.. code-block:: bash

   SIM_SOURCE=$(python -c 'from robodojo_runtime import source_root; print(source_root())')
   python -m robots.robodojo.scripted.eval freeze \
     --output /path/to/new-release --source-root "$SIM_SOURCE" \
     --recipe /path/to/recipe.py --task general_pickup --layout 0 \
     --step-limit 200 --action-budget 180
   python -m robots.robodojo.scripted.eval verify --output /path/to/new-release
   python -m robots.robodojo.scripted.eval run \
     --output /path/to/new-release --endpoint http://localhost:18765

``freeze`` 将配方和数值运动辅助函数嵌入 worker，执行奖励边界静态检查，
并对本地源码、解释器及依赖版本清单计算哈希。``verify`` 检查这些文件的内容及
仿真源码文件清单。这是本地指纹，不是可迁移的软件包、依赖重装或在线服务证明。
外部场景资产及已安装包的内容并未全部纳入哈希；请保留原有路径。

``run`` 在连接服务前领取一次尝试，即使前置检查失败也会消耗该次机会。
已领取的 release 不能重新 freeze 或重试。领取前重新 freeze 会归档旧 manifest。
新实验需要新的 release 目录和全新 episode；统计尝试次数时应保留先前结果。

worker 只能获得公开的 RGB-D、标定、本体状态、任务指令及步数预算。
``env.step`` 的 reward 和 done 槽位均为 ``None``。worker 退出且动作管道关闭后，
独立 evaluator 才读取 ``env.is_success()`` 和 reward。
请检查 ``trial/result.json``：``valid`` 表示执行与哈希检查有效，
``official_success`` 表示原生任务成功谓词。仅凭 CLI 零退出码不能证明这两者。
日志与审计位于 ``trial/``，配方产物位于 ``trial/agent/``。
该协议用于可信代码，不是操作系统安全沙箱，也不构成排行榜提交。

配方
----

``robots/robodojo/recipes/README.md`` 记录了两份历史成功 JSON 配方的证据位置和
哈希：``general_pickup`` layout 1（94/200 步）与 ``pour_by_language`` layout 1
（690/800 步）。配方保存在 ``recipes/historical/``。
它们使用的旧控制器依赖姿态 IK 和不同的动作接口，不能将这些 JSON 文件直接传给
当前 runner。仓库未附带在当前接口上取得可靠成功证据的 Python 配方，
本次提取也未运行仿真 rollout 来验证成功。

可将自己的 Python 配方放在 ``recipes/`` 或仓库外，并显式传入路径。
配方可调用 ``env.get_obs()``、``env.get_status()``、
``env.solve_ik_position(arm, xyz)`` 和 ``env.step(action)``。
动作使用原生关节或末端执行器字段。worker 无法调用 ``solve_ik_pose``、reset
或 evaluator RPC。stdout 专用于桥协议；请向 stderr 输出日志，或将产物写入
``output``。允许导入的模块见 ``scripted/reward_audit.py``。

``move_to`` 和 ``set_gripper`` 从 ``scripted/primitives.py`` 嵌入，配方无需也不应
导入仓库模块。它们的第一个参数是具有 ``env``、``_last_obs`` 和
``_check_cancelled()`` 的对象；第二个参数未使用，可传 ``None``。
以下示例只演示接口、打开右夹爪，不声称任务成功：

.. code-block:: python

   class Motion:
       def __init__(self, env):
           self.env = env
           self._last_obs = None

       def _check_cancelled(self):
           status = self.env.get_status()
           if status["step"] >= status["step_limit"]:
               raise RuntimeError("episode budget exhausted")

   def main(env, output):
       motion = Motion(env)
       set_gripper(motion, None, arm="right", gripper=-1)

``move_to`` 的 gripper 参数为正时关闭、为负时打开、为零时保持当前位置；
``set_gripper`` 的零值也表示打开。无论配方是否自行检查，外层桥都会执行冻结的
动作预算限制。添加成功声明前，应保存对应 task/layout 的 release、评估结果、
动作 trace、初始化设置及视频证据。

离线检查
--------

.. code-block:: bash

   uv pip install -e ".[test]"
   pytest tests/unit_tests/robots/robodojo -v
   ruff check --preview robots/robodojo tests/unit_tests/robots/robodojo

这些 CPU 测试覆盖源码冻结、奖励过滤、单次尝试、worker/evaluator 时序，
以及使用 fake 的环境适配器契约；不会启动 Isaac Sim，也不证明 GPU rollout 成功。
