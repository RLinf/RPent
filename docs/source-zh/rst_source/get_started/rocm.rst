.. _rocm-install-zh:

在 AMD ROCm 上运行
==================

RPent 无需修改代码即可在 AMD Instinct GPU 上运行。本页记录一套已验证的环境：
MI300X 上的 ROCm 6.4,采用 LIBERO-PRO 快速开始路径（Pi0.5 与 SAM3 运行在 GPU 上，
OSMesa 在 CPU 上渲染）。该环境已成功跑通一个完整 LIBERO-PRO 任务；其余路径
（Molmo、XPolicyLab、RoboCasa、RoboTwin）尚未在 ROCm 上验证。

只有安装步骤与 :doc:`quickstart` 不同。安装完成后的资源下载、planner 配置和任务
执行与 NVIDIA 上的命令完全一致。

环境要求
--------

- Linux,一块 AMD Instinct GPU（如 MI300X）,ROCm 6.4 驱动。
- ``docker``,可访问 ``/dev/kfd`` 和 ``/dev/dri``。
- 容器内的 `uv <https://docs.astral.sh/uv/getting-started/installation/>`_。
- 与快速开始相同的 Anthropic API key 和网络要求。

1. 启动容器
-----------

使用 ROCm 6.4 LIBERO 镜像,并通过 ``HIP_VISIBLE_DEVICES`` 暴露一块 GPU：

.. code-block:: bash

   docker run -d --name rpent --network host --ipc host --shm-size 32g \
     --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
     -e HIP_VISIBLE_DEVICES=0 -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa -e LIBERO_TYPE=pro \
     rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4 sleep infinity

``MUJOCO_GL=osmesa`` 和 ``PYOPENGL_PLATFORM=osmesa`` 让 LIBERO-PRO 环境在 CPU 上
渲染。OSMesa 可用但受限于 CPU 性能；ROCm 上的 EGL 尚未尝试。

2. 安装 ROCm 版 PyTorch 的 RPent
--------------------------------

进入容器并克隆仓库：

.. code-block:: bash

   docker exec -it rpent bash
   git clone https://github.com/RLinf/RPent.git
   cd RPent

如果不加 override,``libero-pro`` extras 会从 PyPI 解析出 CUDA 版 ``torch``。
另外,PyPI 的 ``triton``（CUDA）会与 ``pytorch-triton-rocm`` 安装到同一
``triton/`` 目录并在导入时段错误,因此需将其排除。创建 override 文件并安装：

.. code-block:: bash

   printf 'torch==2.8.0+rocm6.4\ntorchvision==0.23.0+rocm6.4\ntriton ; sys_platform == "never"\n' > torch-override.txt
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[libero-pro]" --override torch-override.txt \
     --extra-index-url https://download.pytorch.org/whl/rocm6.4 --index-strategy unsafe-best-match

如果环境中已装入 CUDA 版 ``triton``,先卸载再重装 ROCm 版：

.. code-block:: bash

   uv pip uninstall triton
   uv pip install -e ".[libero-pro]" --override torch-override.txt \
     --extra-index-url https://download.pytorch.org/whl/rocm6.4 --index-strategy unsafe-best-match \
     --reinstall-package pytorch-triton-rocm

验证安装：

.. code-block:: bash

   python -c "import torch; print(torch.version.hip)"; ls .venv/lib/python3.11/site-packages/triton/backends

``torch.version.hip`` 应输出 ROCm 版本号,且 ``triton/backends/`` 中包含 ``amd``。
两项检查都通过后再继续。
后续命令都在已激活的 ``.venv`` 中执行；新开终端时先 ``source .venv/bin/activate`` 再继续，与快速开始保持一致。

3. 下载资源与模型
-----------------

按快速开始的第 2–4 步操作。资源下载前需先用 ``--link-mode copy --no-deps``
重装 ``rpent-liberopro``,之后 ``liberopro-download-assets --skip-existing``
才能正常工作。SAM3 检查点用以下命令下载：

.. code-block:: bash

   modelscope download --model facebook/sam3 sam3.pt --local_dir ./checkpoints/sam3

4. 运行任务
-----------

与 NVIDIA 相同方式启动环境、VLA 和 SAM3 服务,然后在另一台机器或终端运行
planner,通过 ``--env-endpoint``、``--vla-endpoint`` 和 ``--sam3-endpoint``
连接。在 MI300X 上,三个服务均在启动后约 45 秒内通过 ``healthz``。

注意事项
--------

- 仅客户端安装缺少 ``imageio[ffmpeg]``,不会生成 ``episode.mp4``。安装
  ``imageio[ffmpeg]`` 即可启用视频导出。
- ROCm 上目前只验证了使用 Pi0.5 与 SAM3 的 LIBERO-PRO 路径；其他本体路径可能
  还需要额外工作。
- 已验证的检查点为以 bf16 加载的 ``RLinf-Pi05-LIBERO-130-fullshot-SFT``
  （3.35 B 参数）,SAM3 使用 ``sam3.pt`` 提供定位服务。
