Quick Start
===========

Install RPent, run one LIBERO-PRO task, and inspect its result. This example uses the Claude Code planner, Pi0.5 for robot actions, and SAM3 for visual segmentation.

For another platform, follow :doc:`usage/robocasa`, :doc:`usage/robotwin`, or :doc:`real-world robot deployment <usage/real_robots>` and use that platform’s requirements.

Requirements
------------

- Linux, an NVIDIA GPU, and a CUDA 12-compatible driver.
- ``git``, ``bash``, and a C/C++ toolchain.
- `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ to create an isolated Python environment. These commands use Python 3.11.
- An Anthropic API key with access to the example model, and network access to download assets and weights. Model calls incur API charges.

1. Install RPent
----------------

Create an isolated Python environment and install the LIBERO-PRO dependencies.

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[libero-pro]"

Run the remaining commands in this terminal from the RPent repository root. In a new terminal, activate ``.venv`` and set the checkpoint paths and API key again.

2. Download Assets and Models
-----------------------------

Download the LIBERO-PRO scene assets. ``--skip-existing`` reuses files already downloaded:

.. code-block:: bash

   liberopro-download-assets --skip-existing

Download Pi0.5 and SAM3, then set their paths:

.. code-block:: bash

   uv pip install "huggingface_hub>=0.34,<1.0" modelscope
   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
     --exclude optimizer.pt \
     --local-dir ./checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT
   modelscope download --model facebook/sam3 sam3.pt \
     --local_dir ./checkpoints/sam3

   export PI05_CHECKPOINT_PATH="$PWD/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT"
   export SAM3_CHECKPOINT_PATH="$PWD/checkpoints/sam3/sam3.pt"

Pi0.5 executes robot actions; SAM3 locates objects in images. SAM3 is also available on Hugging Face; see :doc:`usage/libero` for access and alternative download instructions.

3. Configure and Check the Planner
----------------------------------

Replace ``YOUR_API_KEY`` with your key:

.. code-block:: bash

   export ANTHROPIC_API_KEY="YOUR_API_KEY"
   rpent-check-llm --planner claude_code --model claude-opus-4-8

The official Anthropic endpoint needs no ``ANTHROPIC_BASE_URL`` setting. See :doc:`usage/configure_planner` for custom endpoints and other planners. Proceed after the check passes; it verifies model-service authentication and connectivity.

4. Run Your First Task
----------------------

Run task ``2`` in ``libero_object_swap`` with scene seed ``0``:

.. code-block:: bash

   rpent --robot libero --libero-type pro \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8 \
     --output-dir ./logs/first-libero-pro

The terminal shows environment, VLA, and SAM3 service startup, followed by planner messages and tool calls. Inspect ``logs/first-libero-pro/`` when the run ends. Choose a new output directory for later runs to preserve this record.

5. Inspect the Result
---------------------

- ``episode.mp4``: replay the robot’s actions.
- ``transcript_*.json``: inspect planner messages, tool calls, and the finish state.
- ``run.log``: inspect runtime messages and errors.

For LIBERO, success is the top-level ``terminated`` value in the final environment state, available in the result of ``view_env_state(step=-1)``. The planner’s ``finish`` status alone does not establish environment success.

See :ref:`run-output-files` for default directory names, exported action sequences, and step artifacts.

See :doc:`usage/dashboard` to watch cameras and actions live. For more tasks, exploration, and experiment reproduction, continue with :doc:`usage/libero`.

If a Step Fails
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Symptom
     - Action
   * - Model connection fails
     - Check the key, model access, and endpoint; rerun ``rpent-check-llm``.
   * - Asset download fails
     - Rerun the download. If Hugging Face is slow, optionally set ``HF_ENDPOINT=https://hf-mirror.com`` for that command.
   * - Environment or model fails to start
     - Check both checkpoint paths, then inspect ``env_server.log``, ``vla_server.log``, or ``sam3_server.log``. For out-of-memory errors, check other GPU processes.
   * - Task runs but does not succeed
     - Review the video and final state to distinguish action failures from service errors. One task does not establish benchmark performance.
