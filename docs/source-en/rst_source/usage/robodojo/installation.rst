RoboDojo Backend Installation
=============================

Use separate Python environments for RPent/SAM3, RoboDojo's Isaac Sim stack,
and Pi_05. Follow the official RoboDojo and XPolicyLab installation instructions
for compatible simulator, CUDA, and policy dependencies. GPU execution requires
the corresponding assets and checkpoints; the RPent integration does not
download them automatically.

Sources and assets
------------------

Install Git LFS before cloning the official repository and its submodules:

.. code-block:: bash

   git lfs install
   git clone --recurse-submodules https://github.com/RoboDojo-Benchmark/RoboDojo.git
   cd RoboDojo
   git lfs pull
   git submodule foreach --recursive 'git lfs pull'
   git lfs fsck

Download the asset/checkpoint repositories linked by the official RoboDojo
release with the same Git LFS workflow: clone, ``git lfs pull``, and
``git lfs fsck``. Follow that release's placement instructions. Confirm that
required files contain real data rather than LFS pointer text before starting
the simulator. If a release has incomplete LFS attributes, resolve that with
the dataset publisher; RPent does not provide a custom materializer.

RPent configuration
-------------------

In the RPent environment:

.. code-block:: bash

   uv pip install -e ".[sam3]"

Configure SAM3's checkpoint using ``SAM3_CHECKPOINT_PATH``. Supply the RoboDojo
checkout and the Python executables explicitly:

.. code-block:: bash

   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --source-root /path/to/RoboDojo \
     --sim-python /path/to/sim-env/bin/python \
     --pi05-python /path/to/pi05-env/bin/python

``--xpolicylab-root`` defaults to ``SOURCE_ROOT/XPolicyLab``; set it for
a separate checkout. Both Python flags default to the current interpreter,
so separate runtimes must pass their executable paths. The CLI constructs
child import paths without reading a workspace's ``config/runtime.env`` or
changing the parent environment. Existing shell environment variables are
inherited by child processes.

Use ``--env-endpoint``, ``--vla-endpoint``, and ``--sam3-endpoint`` to attach
to already running services. A borrowed service requires no local source or
Python path for that component. When starting ``vla_server.py`` directly,
set ``ROBODOJO_PI05_POLICY_ROOT`` to ``XPolicyLab/policy/Pi_05``.
