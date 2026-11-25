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
the simulator adapter from the paired RLinf checkout. Run from the RPent root with both checkouts in the same parent directory.

.. code-block:: bash

   export PROJECT_DIR="$(dirname "$PWD")"
   export RLINF_ROOT="$PROJECT_DIR/RLinf"
   bash tools/rl/install_simulator.sh libero   "$PROJECT_DIR/envs/libero"   "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robotwin "$PROJECT_DIR/envs/robotwin" "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robocasa "$PROJECT_DIR/envs/robocasa" "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robodojo "$PROJECT_DIR/envs/robodojo" "$RLINF_ROOT"

Choose the line for the environment you will run.

.. list-table::
   :header-rows: 1

   * - Simulator
     - Prepare on its host
   * - LIBERO
     - Python 3.12, EGL and MuJoCo. Use the LIBERO-PRO asset command below.
   * - RoboTwin
     - Python 3.11, NVIDIA Vulkan and a CUDA toolkit matching Torch for cuRobo. Use the RoboTwin asset command below.
   * - RoboCasa
     - Python 3.11 and EGL. Run ``robocasa-download-assets --assets-path /work/data/robocasa --no-macros -y`` in its environment. See :doc:`../simulators/robocasa`.
   * - RoboDojo
     - Python 3.11, Isaac Sim 5.1 and an RTX GPU. Prepare the RoboDojo assets and evaluation layouts under ``ROBODOJO_ASSETS_ROOT``. See :doc:`../simulators/robodojo`.

Download assets and SAM3
------------------------

Use the asset downloader on each host that runs the corresponding simulator. LIBERO-PRO's package supplies the BDDL files and native initial states. The downloader sets up the scene assets from `RLinf/LIBERO-PRO-assets <https://huggingface.co/datasets/RLinf/LIBERO-PRO-assets>`_ and the package creates ``config.yaml`` under ``$PROJECT_DIR/data/libero-config``.

.. code-block:: bash

   mkdir -p "$PROJECT_DIR/data/libero-config"
   LIBERO_CONFIG_PATH="$PROJECT_DIR/data/libero-config" \
     "$PROJECT_DIR/envs/libero/bin/liberopro-download-assets" --skip-existing

The RoboTwin downloader from the installed runtime prepares the asset snapshot under ``$PROJECT_DIR/data/robotwin``.

.. code-block:: bash

   "$PROJECT_DIR/envs/robotwin/bin/robotwin-download-assets" \
     --output "$PROJECT_DIR/data/robotwin"

Request access to the `SAM3 weights <https://huggingface.co/facebook/sam3>`_ and download ``sam3.pt`` on the host that serves SAM3. These commands run the Hugging Face CLI through uv.

.. code-block:: bash

   uv venv --python 3.12 "$PROJECT_DIR/envs/sam3"
   export SAM3_PYTHON="$PROJECT_DIR/envs/sam3/bin/python"
   uv pip install --python "$SAM3_PYTHON" -e '.[sam3]'
   uvx --from huggingface_hub hf auth login
   uvx --from huggingface_hub hf download facebook/sam3 sam3.pt --local-dir "$PROJECT_DIR/models/sam3"

For standalone rollout, start SAM3 on the assigned GPU. For training, the node command below starts SAM3 from its YAML.

.. code-block:: bash

   export SAM3_CHECKPOINT_PATH="$PROJECT_DIR/models/sam3/sam3.pt"
   "$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
     --host 127.0.0.1 --port 28424 --cuda-device 1

Run an episode
--------------

From the RPent checkout, activate your model environment and start an OpenAI-compatible server on the model GPUs.

.. code-block:: bash

   export PROJECT_DIR="$(dirname "$PWD")"
   MODEL_PATH="$PROJECT_DIR/models/Qwen3.8-27B" SERVE_GPUS=2,3 \
     bash tools/rl/serve.sh

In a new terminal, return to the RPent checkout and activate the simulator environment. Use the SAM3 endpoint set up above.

.. code-block:: bash

   export PROJECT_DIR="$(dirname "$PWD")"
   source "$PROJECT_DIR/envs/libero/bin/activate"
   export LIBERO_TYPE=pro
   export LIBERO_CONFIG_PATH="$PROJECT_DIR/data/libero-config"
   export RPENT_MODEL=openai-chat:qwen3.8-27b
   export RPENT_BASE_URL=http://127.0.0.1:30080/v1
   export RPENT_SAM3_ENDPOINT=http://127.0.0.1:28424
   export RPENT_OUTPUT_DIR="$PROJECT_DIR/runs/libero-first"
   bash tools/rl/run.sh --simulator libero --suite libero_spatial --task 0 --seed 0

RoboTwin, RoboCasa, and RoboDojo use ``--simulator`` and ``--task-name``. Use ``--layout`` to choose a RoboDojo evaluation layout. The output contains images, measured states, tool results, and the simulator's native outcome. For remote services, forward the loopback listeners over the private network and use the forwarded addresses.

For a batch, save this manifest as ``$PROJECT_DIR/data/jobs.json``.

.. code-block:: json

   [{"environment":"libero","suite":"libero_spatial","task":0,"seed":0}]

.. code-block:: bash

   python tools/rl/run_batch.py run "$PROJECT_DIR/runs/libero-batch" \
     --jobs "$PROJECT_DIR/data/jobs.json" --gpus 0 --per-gpu 1

Each completed rollout is sealed independently. Repeat the same command with the same configuration to skip sealed entries.

Serve training tools
--------------------

Copy ``tools/rl/deployment-local.yaml`` or ``tools/rl/deployment-384.yaml``. Replace ``/work`` in the sample with your project directory and set the addresses and give each host the same YAML file. From the RPent checkout, start its role.

The node command stays in the foreground. Use the SAM3 environment on the local host and a simulator environment on nodes that only serve simulators.

.. code-block:: bash

   "$PROJECT_DIR/envs/sam3/bin/python" -m robots.rl.services \
     --config "$PROJECT_DIR/deployment-local.yaml" --node local

The local example uses GPU 0 for SAM3 and GPU 1 for four slots each of LIBERO and RoboTwin. For 384 slots, run the command with the corresponding name on ``sim``, ``model0`` through ``model3``, and ``twin``. Point the RLinf configuration at the two routers on ``model0``.

Set each backend's ``gpu_slots`` to its physical allocation, and ``endpoint`` to the address the router can reach. Set SAM3 ``replicas`` per GPU and ``max_pending`` for the perception queue. For routed networks, use restricted forwarding and put the forwarded addresses in the corresponding fields. Choose a fresh ``output_dir`` for each launch. It holds the service logs and simulator traces. Stopping the node command releases its own services and simulator processes.

Use the same RPent revision for trainers and simulator hosts. RLinf manages the model during training. Open ``docs/source-en/rst_source/examples/agentic/rpent_episode_sync.rst`` in the paired RLinf checkout for task preparation and training commands.

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
