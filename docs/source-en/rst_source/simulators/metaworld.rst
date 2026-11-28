MetaWorld
=========

Run tabletop manipulation tasks with RPent in `MetaWorld <https://github.com/Farama-Foundation/Metaworld>`_. A visual planner drives a MuJoCo Sawyer arm with RGB-D observations and scripted Cartesian motion tools, so no VLA checkpoint is needed.

Overview
--------

Check the supported action models, planners, tasks, and hardware before installing.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      Scripted Cartesian motion tools by default; optionally a compatible external WAM worker.

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``, ``gemini_cli``, ``opencode``

   .. grid-item-card:: Tasks

      Ten MetaWorld 3.0.0 MT10 task types, each run as one fixed MT1 episode.

   .. grid-item-card:: Hardware

      Linux, Python 3.11; an EGL-capable GPU, or CPU rendering with OSMesa.

Tasks
~~~~~

Each run is one episode of one task type with a fixed seed; results are not MT10 or MT50 benchmark scores.

.. list-table::
   :header-rows: 1

   * - Task
     - Instruction given to the planner
   * - ``reach-v3``
     - Move the gripper to the visible red goal marker.
   * - ``push-v3``
     - Push the puck to the visible goal marker on the table.
   * - ``pick-place-v3``
     - Pick up the puck and place it at the visible goal marker.
   * - ``door-open-v3``
     - Pull the handle to open the door.
   * - ``drawer-open-v3``
     - Pull the drawer open using its handle.
   * - ``drawer-close-v3``
     - Push the drawer closed.
   * - ``button-press-topdown-v3``
     - Press the button down from above.
   * - ``peg-insert-side-v3``
     - Insert the peg into the side-facing hole.
   * - ``window-open-v3``
     - Slide the window open using its handle.
   * - ``window-close-v3``
     - Slide the window closed using its handle.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The table separates what the planner sees, which tools step the simulator, and how the task result is decided.

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - ``view_env_state`` returns the latest 480 x 480 RGB image, end-effector position, gripper opening, task instruction, simulator step count (``steps``), and recorded snapshot index (``step``). ``back_project`` reads world coordinates from the recorded depth image. Object and goal coordinates are not exposed.
   * - Action
     - ``move_to`` moves the end effector toward a world XYZ target, and ``set_gripper`` opens or closes the gripper in place. The wrist orientation is fixed. With ``--am-endpoint``, ``wam_act`` executes action chunks from an external WAM worker.
   * - Reward / success
     - MetaWorld's native success signal, saved as ``success`` in ``result.json``. An episode ends at the step limit (500 by default) or when native success triggers. Planner ``finish`` and ``move_to`` returning ``reached`` are not task success.
   * - Task prompt
     - The fixed instruction for the selected task, listed in the table above.

.. dropdown:: Motion and Projection Details

   ``move_to(target_xyz, gripper, max_steps=50)`` applies closed-loop XYZ corrections at MetaWorld's native action scale of 0.01 m per step. A call runs at most 100 steps and returns ``reached``, ``distance_m``, and ``steps``. It stops when the end effector is within 5 mm of the target, when the episode ends, or at ``max_steps``. Motions are not collision-aware.

   The gripper value applies only on executed steps: +1 closes and -1 opens. If the end effector already lies within 5 mm of the target, ``move_to`` takes zero steps and leaves the gripper unchanged. Use ``set_gripper(gripper, steps=10)`` to open or close in place; it runs at most 20 steps.

   MetaWorld's success tolerance can be looser than 5 mm, so an episode may succeed and end before ``move_to`` reports ``reached``.

   Call ``back_project`` with camera ``primary``, a selection such as ``{"image_shape": [480, 480], "points_rc": [[240, 240]]}``, and the ``step`` returned by ``view_env_state``. ``samples_xyz`` lists world coordinates in metres in the input order. A pixel with invalid or far-plane depth keeps a ``null`` slot, and if no selected pixel has valid depth the result has ``status: "need_observation"``. Pixel coordinates outside the image fail validation. Observation tools do not advance the simulation.

Installation
------------

Create an isolated Python 3.11 environment and install the ``metaworld`` extra. MetaWorld ships its task assets in the package, so nothing else needs to be downloaded:

.. code-block:: bash

   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[metaworld]"

.. note::

   The extra pins MuJoCo 3.14.0, which conflicts with MuJoCo 3.3.0 in the ``libero``, ``libero-pro``, and ``libero-plus`` extras.

Run a Task
----------

Configure your planner credentials (see :doc:`../guides/configure_planner`), set the model name, and run one task:

.. code-block:: bash

   export PLANNER_MODEL=openai/your-model
   rpent --robot metaworld --task reach-v3 --seed 0 \
     --planner api --model "$PLANNER_MODEL" \
     --output-dir ./runs/metaworld-reach

``rpent`` starts the environment, runs the planner, and stops the environment when the run ends. Common options:

- ``--task``: one of the ten task IDs above; ``reach-v3`` by default.
- ``--seed``: an integer from 0 to 4294967295; 0 by default.
- ``--camera``: ``corner2`` by default, or ``corner``, ``corner3``, ``topview``, ``behindGripper``, and ``gripperPOV``.
- ``--max-episode-steps``: the episode step limit; 500 by default.
- ``--cuda-device``: the physical CUDA GPU for the EGL renderer, for example ``--cuda-device 0``.

MetaWorld uses local task memory, which starts empty; pass ``--memory-dir`` to choose its directory. ``--memory-profile hf``, exploration mode, and Flash Mode are not supported.

For a Dashboard session, replace ``--task reach-v3 --seed 0`` with ``--dashboard``, then claim a task with ``/rpent-task reach-v3 0``. The Dashboard starts a fresh environment for each task, so it cannot be combined with ``--env-endpoint``. See :doc:`../guides/dashboard`.

.. dropdown:: Separate Simulator Environment and Standalone Server

   To run the simulator in another Python environment, pass ``--sim-python /path/to/python``. That interpreter must have ``.[metaworld]`` installed from this repository.

   To reuse a running environment server, start it with the task, seed, camera, and step limit of the run:

   .. code-block:: bash

      python -m robots.metaworld.env_server --task reach-v3 --seed 0 \
        --port 18765 --video-dir ./runs/standalone/videos

   Then connect a run to it with the same task and seed:

   .. code-block:: bash

      rpent --robot metaworld --task reach-v3 --seed 0 \
        --env-endpoint http://127.0.0.1:18765 \
        --planner api --model "$PLANNER_MODEL"

   Connecting resets the episode, so give each run its own server. ``rpent`` does not stop a server it connects to; the server's owner sets its renderer device and video directory.

View Results
------------

Each run writes its records to ``--output-dir``:

- ``result.json``: ``success`` holds MetaWorld's native success, and ``planner_finish`` holds the planner's ``finish`` report. ``success`` is ``null`` when the environment result is unavailable. The file also records the task, seed, camera, step limit, and planner budget.
- ``states.json`` with the RGB and depth images saved after each tool call.
- ``videos/episode_0000.mp4``: recorded when ``rpent`` starts the environment, with the initial frame and one frame per executed simulation step.
- ``environment.json``: the simulator's dependency versions, step limit, action scale, and rendering settings.
- ``env_server.log``: the environment log.

External Action Model
---------------------

To let the planner call a running WAM worker that supports MetaWorld, add ``--am-endpoint``:

.. code-block:: bash

   rpent --robot metaworld --task reach-v3 --seed 0 \
     --planner api --model "$PLANNER_MODEL" \
     --am-endpoint http://HOST:PORT \
     --output-dir ./runs/metaworld-wam

The planner gets ``wam_act(max_chunks=1)``, which executes up to ``max_chunks`` predicted action chunks (at most 20). ``move_to`` and ``set_gripper`` remain available. RPent checks the worker before it starts the environment, and a Dashboard session shows the worker's status as ``WAM`` next to ``ENV``.

Start and stop a MetaWorld-trained worker yourself; RPent ships none, and LIBERO or RoboTwin checkpoints do not fit this action space.

.. dropdown:: WAM Worker Requirements

   The worker implements the WAM interface described in :doc:`../development/add_action_model` and reports embodiment ``metaworld_sawyer_4d`` with action space ``metaworld.cartesian.delta.v1``. It may request only the ``primary`` camera, ``eef_position`` (3 values), and ``gripper_opening`` (1 value).

   Each request carries the top-left-origin RGB image under ``images.primary``, the end-effector position in world metres under ``state.eef_position``, the normalized opening under ``state.gripper_opening`` (shape ``[1]``), and the task instruction. The worker returns finite ``[N, 4]`` actions in ``[-1, 1]``: XYZ world deltas scaled by 0.01 m, then the gripper (+1 closes, -1 opens), at 80 Hz with fixed wrist orientation.

   An incompatible worker fails before the environment starts, and an invalid action chunk fails before any step. Cancellation and the end of the episode stop further actions. Each run or Dashboard task binds its own worker session and releases it when the task ends.

Troubleshooting
---------------

Check ``env_server.log`` in the output directory first.

- **The environment fails to start on a machine without an EGL-capable GPU.** Install the OSMesa libraries and run ``export MUJOCO_GL=osmesa`` before ``rpent``; RPent keeps the backend you set.
- **The renderer uses the wrong GPU.** Pass ``--cuda-device 0`` with the physical GPU ordinal. If ``MUJOCO_EGL_DEVICE_ID`` is already set in the environment, it takes precedence and ``--cuda-device`` is ignored.
- **A run with** ``--env-endpoint`` **fails to connect.** Start the server with the same ``--task``, ``--seed``, ``--camera``, and ``--max-episode-steps`` as the run.
- **A run with** ``--am-endpoint`` **fails before the environment starts.** The worker did not answer within 30 seconds or reported a different embodiment or action space; see WAM Worker Requirements above.
