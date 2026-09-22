RoboDojo Backend Installation
=============================

This page covers what the RPent integration adds on top of the upstream
repositories. Follow the official RoboDojo and XPolicyLab instructions for the
simulator, CUDA and policy dependencies; GPU execution needs the matching
assets and checkpoints, and RPent does not download them.

Python environments
-------------------

The backend drives three interpreters and they must stay separate. Isaac Sim
pins ``websockets==12.0``, ``numpy==1.26.0``, ``packaging==23.0``,
``filelock==3.13.1`` and ``typing_extensions==4.12.2``; the RPent environment
runs a newer ``websockets`` with its own ``torch`` build, and the Pi_05
environment runs JAX and ``openpi``. Installing them into one interpreter
breaks the Isaac Sim pins.

**1. RPent.** Install the repository, then add the perception extra this
backend uses:

.. code-block:: bash

   uv pip install -e ".[sam3]"

**2. RoboDojo simulator.** Use the upstream installer, which builds the Isaac
Sim environment and the vendored CuRobo. That script is the supported path and
RPent does not re-implement it:

.. code-block:: bash

   cd /path/to/RoboDojo
   bash scripts/install.sh

The combination this backend is validated against is Python 3.11 with
``isaacsim 5.1.0.0``, ``torch 2.7.0+cu128``, ``numpy 1.26.0``,
``websockets 12.0``, ``viser 0.1.34``, ``tyro 0.9.0`` and ``warp-lang 1.11.0``,
with CuRobo installed from ``third_party/curobo``. Pass that environment's
interpreter as ``--sim-python``, and do not install RPent or Pi_05 packages into
it.

**3. Pi_05 policy.** Build the uv environment named by XPolicyLab's deploy
config (``policy_uv_env_path: openpi``):

.. code-block:: bash

   cd /path/to/RoboDojo/XPolicyLab/policy/Pi_05
   bash install.sh

The script requires ``uv`` and creates ``openpi/.venv``. The RoboDojo launcher
activates that environment and needs a conda installation plus an interpreter it
can import YAML from, so set ``ROBODOJO_CONDA_ROOT`` when the default one cannot.
Pass the matching interpreter as ``--pi05-python``. RPent is not installed into
this environment: the CLI composes ``PYTHONPATH`` for every child service from
the RPent repository root, ``--source-root`` and ``--xpolicylab-root``, so no
RPent package has to live in the policy environment.

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

The default ``--policy-backend rlinf`` requires a policy interpreter that
provides the ``pi05_robodojo_arx_x5`` config and its openpi dependencies;
RLinf main does not include that config yet. Set ``PI05_CHECKPOINT_PATH`` to a
compatible RLinf checkpoint and pass that interpreter as ``--pi05-python``.
Use ``--policy-backend xpolicylab`` for the XPolicyLab environment described above.

Configure SAM3's checkpoint using ``SAM3_CHECKPOINT_PATH``, and export the
placement settling budget. The default leaves objects unstable in official
mode; the variable is read by the RoboDojo checkout, not by RPent, and the CLI
passes it on to the child services it starts:

.. code-block:: bash

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000

Supply the RoboDojo checkout and the Python executables explicitly:

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
Python path for that component. The CLI starts the shared
``rpent.robots.components.pi05_vla_server --embodiment robodojo`` by default.
With ``--policy-backend xpolicylab``, it starts ``xpolicylab_vla_server`` with
``--policy-root`` pointing to ``XPolicyLab/policy/Pi_05`` instead.
Select the matching backend when borrowing a VLA endpoint. Changing backend
does not convert checkpoints; the RLinf client encodes native observations
into openpi's wire format.

Every owned service logs and writes into the run's output directory: the CLI
passes it as ``--save-dir`` to the environment server and as ``--output-dir``
to the optional XPolicyLab entry point, so concurrent runs do not share state.

Verify the installation
-----------------------

Run one bounded development episode and confirm the services come up before the
planner takes over:

.. code-block:: bash

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000
   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --planner codex --model <planner-model> --max-turns 1 \
     --source-root /path/to/RoboDojo \
     --sim-python /path/to/sim-env/bin/python \
     --pi05-python /path/to/pi05-env/bin/python \
     --output-dir /path/to/run-output

Expected behaviour:

* ``/path/to/run-output`` contains ``robodojo_env_server.log``,
  ``sam3_server.log``, ``robodojo_vla_server.log`` and, once the policy server is
  spawned with XPolicyLab, ``vla_server.log``.
* The environment server reports ready, and the first observation carries
  ``cam_head``, ``cam_left_wrist`` and ``cam_right_wrist`` with intrinsics and
  extrinsics, plus joint and gripper state.
* The run writes one MP4 per camera under ``/path/to/run-output/videos``.
* Shutdown leaves no owned child process behind and the GPUs return to idle.

A service that exits during startup is the usual failure mode; read its log in
the run output directory first. Isaac Sim start-up takes tens of seconds and the
first run also compiles shaders.
