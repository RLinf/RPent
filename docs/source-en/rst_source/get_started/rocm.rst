.. _rocm-install:

Running on AMD ROCm
===================

RPent runs unmodified on AMD Instinct GPUs. This page documents a verified
setup: ROCm 6.4 on an `MI300X <https://www.amd.com/en/products/accelerators/instinct/mi300.html>`_
with the LIBERO-PRO quickstart path (Pi0.5 and SAM3 on the GPU, OSMesa
rendering on the CPU). One full LIBERO-PRO episode ran to success on this
setup; other paths (Molmo, XPolicyLab, RoboCasa, RoboTwin) are not yet
verified on ROCm.

Only the install step differs from :doc:`quickstart`. Everything after the
install — asset download, planner configuration, and task execution — follows
the same commands as on NVIDIA.

Requirements
------------

- Linux with an AMD Instinct GPU (for example MI300X) and ROCm 6.4 drivers.
- ``docker`` with access to ``/dev/kfd`` and ``/dev/dri``.
- `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ inside the container.
- An Anthropic API key and network access, as in :doc:`quickstart`.

1. Start the Container
----------------------

Use the ROCm 6.4 LIBERO image and expose one GPU through ``HIP_VISIBLE_DEVICES``:

.. code-block:: bash

   docker run -d --name rpent --network host --ipc host --shm-size 32g \
     --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
     -e HIP_VISIBLE_DEVICES=0 -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa -e LIBERO_TYPE=pro \
     rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4 sleep infinity

``MUJOCO_GL=osmesa`` and ``PYOPENGL_PLATFORM=osmesa`` make the LIBERO-PRO
environment render on the CPU. OSMesa works but is CPU-bound; EGL on ROCm has
not been tried.

2. Install RPent with ROCm PyTorch
----------------------------------

Enter the container and clone the repository:

.. code-block:: bash

   docker exec -it rpent bash
   git clone https://github.com/RLinf/RPent.git
   cd RPent

Without an override, the ``libero-pro`` extras resolve CUDA ``torch`` wheels
from PyPI. Also, PyPI ``triton`` (CUDA) installs into the same ``triton/``
directory as ``pytorch-triton-rocm`` and segfaults on import, so it is
excluded. Create the override file and install:

.. code-block:: bash

   printf 'torch==2.8.0+rocm6.4\ntorchvision==0.23.0+rocm6.4\ntriton ; sys_platform == "never"\n' > torch-override.txt
   uv venv venv --python 3.11
   uv pip install --python venv/bin/python -e "RPent[libero-pro]" --override torch-override.txt \
     --extra-index-url https://download.pytorch.org/whl/rocm6.4 --index-strategy unsafe-best-match

If a CUDA ``triton`` was already installed in the environment, remove it and
reinstall the ROCm one:

.. code-block:: bash

   uv pip uninstall --python venv/bin/python triton
   uv pip install --python venv/bin/python -e "RPent[libero-pro]" --override torch-override.txt \
     --extra-index-url https://download.pytorch.org/whl/rocm6.4 --index-strategy unsafe-best-match \
     --reinstall-package pytorch-triton-rocm

Verify the install:

.. code-block:: bash

   python -c "import torch; print(torch.version.hip)"; ls venv/lib/python3.11/site-packages/triton/backends

``torch.version.hip`` prints a ROCm version and ``triton/backends/`` contains
``amd``. Proceed only after both checks pass.

3. Download Assets and Models
-----------------------------

Follow steps 2–4 of :doc:`quickstart`. The asset download needs
``rpent-liberopro`` reinstalled with ``--link-mode copy --no-deps`` before
``liberopro-download-assets --skip-existing`` works.
Download the SAM3 checkpoint with:

.. code-block:: bash

   modelscope download --model facebook/sam3 sam3.pt --local_dir ./checkpoints/sam3

4. Run a Task
-------------

Start the environment, VLA, and SAM3 services the same way as on NVIDIA, then
run the planner from a separate machine or terminal, connecting through
``--env-endpoint``, ``--vla-endpoint``, and ``--sam3-endpoint``. All three
services pass ``healthz`` within about 45 seconds of start on MI300X.

Caveats
-------

- A client-only install lacks ``imageio[ffmpeg]``, so ``episode.mp4`` is not
  written. Install ``imageio[ffmpeg]`` to enable the video export.
- Only the LIBERO-PRO path with Pi0.5 and SAM3 is verified on ROCm; other
  embodiment paths may need additional work.
- The verified checkpoint is ``RLinf-Pi05-LIBERO-130-fullshot-SFT`` loaded in
  bf16 (3.35 B parameters), with SAM3 ``sam3.pt`` serving grounding requests.
