Robot episodes
==============

Run one model against LIBERO, RoboTwin, RoboCasa or RoboDojo. All four use the
same episode loop and the tools ``view_env``, ``segment``, ``back_project`` and
``execute_actions``. The simulator measures state, applies actions, and reports
task success. Only ``view_env`` sends images to the model.

Install the simulator
---------------------

Check out RPent and the paired RLinf revision. Run the installer on the host
that owns the simulator. It creates a separate virtual environment and uses
the paired RLinf simulator adapter from its checkout.

.. code-block:: bash

   export RLINF_ROOT=/work/RLinf
   bash tools/rl/install_simulator.sh libero   .venv-libero   "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robotwin .venv-robotwin "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robocasa .venv-robocasa "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robodojo .venv-robodojo "$RLINF_ROOT"

Choose the line for the environment you will run.

.. list-table::
   :header-rows: 1

   * - Simulator
     - Prepare on its host
   * - LIBERO
     - Python 3.12, EGL and MuJoCo. Set ``LIBERO_TYPE=pro`` and ``LIBERO_CONFIG_PATH``. The configuration needs the asset, BDDL and native state paths. See :doc:`../simulators/libero`.
   * - RoboTwin
     - Python 3.11, NVIDIA Vulkan and a CUDA toolkit matching Torch for cuRobo. Run ``robotwin-download-assets`` in its environment and set ``ROBOTWIN_ASSETS_PATH`` to the snapshot root. See :doc:`../simulators/robotwin`.
   * - RoboCasa
     - Python 3.11 and EGL. Run ``robocasa-download-assets --assets-path /work/data/robocasa --no-macros -y`` in its environment. See :doc:`../simulators/robocasa`.
   * - RoboDojo
     - Python 3.11, Isaac Sim 5.1 and an RTX GPU. Prepare the RoboDojo assets and evaluation layouts under ``ROBODOJO_ASSETS_ROOT``. See :doc:`../simulators/robodojo`.

Provide an OpenAI-compatible model endpoint and a SAM3 service. Install the
``sam3`` extra in the Python environment that serves SAM3 and set
``SAM3_CHECKPOINT_PATH`` to the local checkpoint. Run SAM3 on its assigned GPU.

Download the SAM3 weights from the configuration instructions in :doc:`../simulators/libero`.

.. code-block:: bash

   uv venv --python 3.12 /work/envs/sam3
   export SAM3_PYTHON=/work/envs/sam3/bin/python
   uv pip install --python "$SAM3_PYTHON" -e '.[sam3]'
   export SAM3_CHECKPOINT_PATH=/work/models/sam3/sam3.pt
   "$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
     --host 127.0.0.1 --port 28424 --cuda-device 1

Run an episode
--------------

In your model environment, start an OpenAI-compatible server on the model GPUs.

.. code-block:: bash

   MODEL_PATH=/work/models/Qwen3.8-27B SERVE_GPUS=2,3 \
     bash tools/rl/serve.sh

In a new terminal, activate the simulator environment. Use the SAM3 endpoint set up above.

.. code-block:: bash

   source .venv-libero/bin/activate
   export LIBERO_TYPE=pro
   export LIBERO_CONFIG_PATH=/work/data/libero-config
   export RPENT_MODEL=openai-chat:qwen3.8-27b
   export RPENT_BASE_URL=http://127.0.0.1:30080/v1
   export RPENT_SAM3_ENDPOINT=http://127.0.0.1:28424
   export RPENT_OUTPUT_DIR=/work/runs/libero-first
   bash tools/rl/run.sh --simulator libero --suite libero_spatial --task 0 --seed 0

RoboTwin, RoboCasa, and RoboDojo use ``--simulator`` and ``--task-name``. Use ``--layout`` to choose a RoboDojo evaluation layout. The output contains images, measured states, tool results, and the simulator's native outcome. For remote services, forward the loopback listeners over the private network and use the forwarded addresses.

For a batch, save this manifest as ``/work/data/jobs.json``.

.. code-block:: json

   [{"environment":"libero","suite":"libero_spatial","task":0,"seed":0}]

.. code-block:: bash

   python tools/rl/run_batch.py run /work/runs/libero-batch \
     --jobs /work/data/jobs.json --gpus 0 --per-gpu 1

Each completed rollout is sealed independently. Repeat the same command with the same configuration to skip sealed entries.

Serve training tools
--------------------

Copy ``tools/rl/deployment-local.yaml`` or ``tools/rl/deployment-384.yaml``. Fill in the interpreters, assets, and addresses and give each host the same YAML file. From the RPent checkout, start its role.

Run the node command from a Python environment with RPent and PyYAML installed. It stays in the foreground.

.. code-block:: bash

   python -m robots.rl.services --config /work/deployment.yaml --node local

The local example uses GPU 0 for SAM3 and GPU 1 for four slots each of LIBERO and RoboTwin. For 384 slots, run the command with the corresponding name on ``sim``, ``model0`` through ``model3``, and ``twin``. Point the RLinf configuration at the two routers on ``model0``.

Set each backend's ``gpu_slots`` to its physical allocation, and ``endpoint`` to the address the router can reach. Set SAM3 ``replicas`` per GPU and ``max_pending`` for the perception queue. For routed networks, use restricted forwarding and put the forwarded addresses in the corresponding fields. Choose a fresh ``output_dir`` for each launch. It holds the service logs and simulator traces. Stopping the node command releases its own services and simulator processes.

Use the same RPent revision for trainers and simulator hosts. RLinf manages the model during training. Continue with the `RLinf mixed training guide <https://github.com/RLinf/RLinf/blob/main/docs/source-en/rst_source/examples/agentic/rpent_episode_sync.rst>`_ to start Ray and training.

Coordinates and actions
-----------------------

Positions and lengths use centimetres at the model interface. Linear velocities use centimetres per second. Orientations use fixed-axis XYZ roll, pitch and yaw in radians.
The frame origin is the measured arm mount, or the midpoint of the two
mounts. Forward, robot left and up are positive X, Y and Z. The back projection
and segmentation tools report visible geometry in this frame. Raw depth and
simulator point maps use metres.

A batch contains up to three ordered actions. Each item controls one
component. A RoboCasa arm target uses the frame measured when its item begins,
including after a preceding base or lift action. Observation and geometry
receipts carry the capture and result frame versions. These versions describe
the measurement and are absent from the action input.
