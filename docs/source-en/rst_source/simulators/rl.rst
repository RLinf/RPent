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
     - Python 3.12, EGL and MuJoCo. Set ``LIBERO_TYPE=pro`` and ``LIBERO_CONFIG_PATH``. The configuration needs the asset, BDDL and native state paths. See :doc:`libero`.
   * - RoboTwin
     - Python 3.11, NVIDIA Vulkan and a CUDA toolkit matching Torch for cuRobo. Run ``robotwin-download-assets`` in its environment and set ``ROBOTWIN_ASSETS_PATH`` to the snapshot root. See :doc:`robotwin`.
   * - RoboCasa
     - Python 3.11 and EGL. Run ``python -m robocasa.scripts.download_kitchen_assets --type all`` in its environment. See :doc:`robocasa`.
   * - RoboDojo
     - Python 3.11, Isaac Sim 5.1 and an RTX GPU. Prepare the RoboDojo assets and evaluation layouts under ``ROBODOJO_ASSETS_ROOT``. See :doc:`robodojo`.

Provide an OpenAI-compatible model endpoint and a SAM3 service. Install the
``sam3`` extra in the Python environment that serves SAM3 and set
``SAM3_CHECKPOINT_PATH`` to the local checkpoint. Run SAM3 on its assigned GPU.

.. code-block:: bash

   "$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
     --host 127.0.0.1 --port 28424 --cuda-device 1

Run an episode
--------------

Activate the simulator's virtual environment. This example runs LIBERO with
the local SAM3 endpoint and an existing model server.

.. code-block:: bash

   source .venv-libero/bin/activate
   export RPENT_MODEL=openai-chat:Qwen3.5-4B
   export RPENT_BASE_URL=http://127.0.0.1:30000/v1
   export RPENT_SAM3_ENDPOINT=http://127.0.0.1:28424
   export RPENT_OUTPUT_DIR="$PWD/runs/libero-first"
   bash tools/rl/run.sh --simulator libero --suite libero_spatial --task 0 --seed 0

For RoboTwin, RoboCasa and RoboDojo use ``--simulator`` and ``--task-name``.
Set ``--layout`` for a RoboDojo evaluation layout. The run directory contains
the images, measured states, tool results and native outcome. Use
``tools/rl/run_batch.py`` with a JSON list of task descriptions for collection
with per-episode completion records and resume.

Serve training tools
--------------------

Use the same simulator environment and assets for the RPC server. The
backend owns its allocated GPU slots and children. This LIBERO example serves
four concurrent episodes.

.. code-block:: bash

   python -m robots.rl.remote_manager --simulator libero --gpu-slots 0:4 \
     --sam3-endpoint http://127.0.0.1:28424 --port 28561 \
     --output-dir "$PWD/runs/libero-toolkit"

Use one endpoint per simulator. The listener defaults to loopback. Use a
restricted tunnel for another host. Local paths and simulator assets stay
on the backend, and RPC transfers the model-facing observations and results.
Use the same RPent revision in the order-scheduling planner and all toolkits.
See the `RLinf training guide <https://github.com/zhexuanxu/RLinf/blob/rpent-ppo/docs/source-en/rst_source/examples/agentic/rpent_episode_sync.rst>`_.

Coordinates and actions
-----------------------

Positions, lengths and linear velocities use centimetres at the model
interface. Orientations use fixed-axis XYZ roll, pitch and yaw in radians.
The frame origin is the measured arm mount, or the midpoint of the two
mounts. Forward, robot left and up are positive X, Y and Z. The back projection
and segmentation tools report visible geometry in this frame. Raw depth and
simulator point maps use metres.

A batch contains up to three ordered actions. Each item controls one
component. A RoboCasa arm target uses the frame measured when its item begins,
including after a preceding base or lift action. Observation and geometry
receipts carry the capture and result frame versions. These versions describe
the measurement and are absent from the action input.
