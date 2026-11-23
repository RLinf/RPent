Robot Episodes
==============

Run a model against LIBERO, RoboTwin, RoboCasa or RoboDojo through the same
four tools. The simulator supplies the task and its native success verdict.

Installation
------------

Use a Python environment for each simulator. Install RPent in that environment
and install the simulator libraries and assets described in the simulator's
guide. The requirements under ``requirements/libero-rl.txt``,
``requirements/rl-robotwin.txt`` and ``requirements/rl-robocasa.txt`` cover the
corresponding episode runtimes. For RoboDojo, use its Isaac environment and
the ``robodojo-sim`` extra. Provide a SAM3 checkpoint or an existing SAM3
endpoint. Keep the model server in its own environment.

For an RLinf deployment, install ``requirements/episode-planner.txt`` in the
RL runtime from the paired RPent checkout. The file installs the planner and
RPC client. Install the simulator requirements on the hosts that own the
simulation processes.

.. code-block:: bash

   uv pip install --python "$RL_PYTHON" -r requirements/episode-planner.txt
   uv pip install --python "$LIBERO_PYTHON" -r requirements/libero-rl.txt
   uv pip install --python "$ROBOTWIN_PYTHON" -r requirements/rl-robotwin.txt
   uv pip install --python "$ROBOCASA_PYTHON" -r requirements/rl-robocasa.txt
   uv pip install --python "$ROBODOJO_PYTHON" -e '.[robodojo-sim]' \
     --extra-index-url https://pypi.nvidia.com \
     --override requirements/robodojo-override.txt

Set each variable to the Python executable in its own environment. Run the
simulator install and asset setup on its backend host.

.. list-table::
   :header-rows: 1

   * - Simulator
     - Validated runtime
   * - LIBERO-PRO
     - Python 3.12, MuJoCo 3.3.0, robosuite 1.5.2 and the LIBERO-PRO assets
   * - RoboTwin
     - Python 3.11, SAPIEN 3.0.0b1, RoboTwin runtime 0.1.3 and the matching cuRobo build
   * - RoboCasa
     - Python 3.11, MuJoCo 3.3.0, RoboCasa365 1.0.1 and the pinned robosuite revision
   * - RoboDojo
     - Python 3.11, Isaac Sim 5.1, matching IsaacLab and cuRobo builds, and native evaluation layouts

Run an Evaluation
-----------------

Activate the simulator environment and run from the RPent checkout. This
example uses an existing SAM3 pool and an OpenAI-compatible model server.
To use the checkpoint processor for prompt budgeting in a direct run, install
the matching Transformers, Torch and Torchvision versions in that interpreter
and set ``RPENT_TOKENIZER_PATH`` to the local checkpoint. With a remote toolkit,
only the planner interpreter loads the checkpoint processor.

.. code-block:: bash

   export RPENT_MODEL=openai-chat:Qwen3.8-27B
   export RPENT_BASE_URL=http://127.0.0.1:30000/v1
   export RPENT_SAM3_ENDPOINT=http://127.0.0.1:28424
   bash tools/rl/run.sh --simulator libero --libero-type pro \
     --suite libero_spatial --task 2 --seed 0 --cuda-device 0 \
     --world-cord general --position-unit cm \
     --max-turns 100 --max-tokens 32768 --reasoning-effort xhigh

For the other simulators, select ``robotwin``, ``robocasa`` or ``robodojo``
and use ``--task-name``. RoboDojo also takes ``--layout``. To collect a fixed
task list, pass its JSON file to ``python tools/rl/run_batch.py run OUTPUT --jobs MANIFEST``.
An optional ``--pilot-jobs`` subset runs first and counts toward the same list.
The collector writes audited outcomes after each episode and resumes at the
remaining task and seed entries.

Connect an RLinf Planner
------------------------

Start the backend in the simulator environment and assign its simulator, GPU
slots and SAM3 endpoint. For RoboTwin, export ``ROBOTWIN_ASSETS_PATH`` in the
backend's environment before starting the manager.

.. code-block:: bash

   python -m robots.rl.remote_manager --simulator libero \
     --gpu-slots 0:1,1:1 --sam3-endpoint http://127.0.0.1:28424 \
     --output-dir /path/to/backend --port 28490

Set RLinf's ``tools.rpent.python`` to the planner interpreter and
``tools.rpent.remote_toolkit_endpoint`` to the backend address reachable by its
agent workers. Use the same RPent version on both sides. The backend owns
episode slots and verifies the requested simulator, input contract and
coordinates. Direct evaluation and remote execution use the same simulator
adapter.

Actions and Perception
----------------------

- ``view_env`` returns the latest RGB with camera names and dimensions.
  It is the only tool that sends RGB to the model.
- ``segment`` takes either text or a point for SAM3 and reports the visible
  surface. ``back_project`` takes a pixel, a box or a saved mask and uses the
  recorded depth. Pixels are ``[row,col]``. Boxes are half-open
  ``[row_min,col_min,row_max,col_max]``. Point and box inputs use the camera
  dimensions at their observation step. A mask fixes its source.
- Perception receipts identify the camera, capture step, position unit and
  both coordinate versions. History keeps the original RGB-D and calibration.
  Its geometric results use the current result coordinates.
- ``execute_actions`` executes up to three ordered items. On RoboTwin and
  RoboDojo each arm item names ``left`` or ``right``. A RoboCasa batch may
  include ``move_base``, ``move_lift`` and arm items. The interpreter measures
  the frame before each item and reads the arm target exactly as supplied in
  that frame. Each item returns measured state and residuals. Partial motion,
  an error or a terminal condition stops the remaining items.

``--world-cord base`` uses the simulator world. ``general`` uses the arm
mount or the midpoint of two mounts, with +x forward, +y left and +z up.
``--position-unit m|cm`` controls linear inputs and results. Angles use
radians. Camera prompts describe approximate picture directions for finding
objects. Use the depth and state tools for action coordinates. Native success
and budgets end the episode automatically. A model's final text ends an
unfinished attempt according to the native verdict.
