Flash Mode
==========

A **Flash plan** stores the action sequence for a LIBERO task and marks the key
objects or locations needed by those actions. During replay, RPent finds their
current coordinates in the camera images, updates the action coordinates, and
executes the recorded actions in order.

This keeps planning and perception separate:

1. The Flash plan decides **what to do**.
2. SAM3 or the point locator finds **where to do it** in the current scene.
3. The LIBERO toolkit executes the actions at the updated coordinates.

``--planner flash`` therefore does not call an LLM to make planning
decisions. The locator is used only for visual localization: it points to a requested
object or location in a camera image so RPent can recover its current
coordinates.

LIBERO-PRO performance and execution time
-----------------------------------------

Across the complete 800-case LIBERO-PRO matrix (Spatial, Object, Goal, and Long;
task/swap; 10 seeds per task), Flash Mode with Molmo2-8B solved 581 episodes (72.63%).
The two tasks without a successful source trace and therefore no Flash plan
are conservatively counted as 0/10. The eight suite scores are listed in
:doc:`Performance <../leaderboard/performance>`.

The :doc:`Time & Token Costs <../leaderboard/time-token-costs>` table reports
60.19 seconds per evaluation episode and 0 output tokens for Flash Mode.
The mean covers successful, failed, and timed-out evaluation episodes; tokens
refer to evaluation-stage output only.

\* RPent Flash Mode uses directly downloaded, officially released GPT-5.5
exploration memory; Molmo2-8B is used for visual localization.
These results use the best-performing seed from s0-s9.

How replay works
----------------

Each plan contains an action plan and a set of anchors. An anchor describes a
task-relevant object or location, such as the object to pick or the destination
for a placement. Actions that depend on an anchor store their offset from that
anchor instead of relying only on an absolute coordinate.

At run time, RPent extracts the anchors required by the plan and locates each
one with the interface recorded for it:

* **SAM3** locates segmentation anchors and returns an object mask and its
  position.
* **The locator service** locates point anchors by pointing to the requested object or
  location in the camera image.

RPent then combines each live anchor position with the offset stored in the
plan and executes the resulting waypoint. This lets the same plan run when
objects appear at different positions.

Flash plan files
----------------

Plans are distributed through the `GPT-5.5 memory directory
<https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/libero/GPT_5.5_xhigh>`_
on Hugging Face rather than tracked in Git. Flash defaults to
``--memory-version GPT_5.5_xhigh`` and uses that version's isolated cache.
The published ``GPT_5.5_xhigh/flash/`` directory contains 78 plan/anchor pairs:
20 Spatial, 20 Object, 19 Goal and 19 Long tasks. ``goal_swap_t0`` and
``10_swap_t9`` have no successful source plan and remain absent. These files
come unchanged from the merged Hugging Face Flash release; publishing them in
the versioned directory does not constitute a new simulator evaluation.

With ``--memory-profile local --memory-dir /path/to/memory/libero/GPT_5.5_xhigh``,
replay reads ``flash/`` plans under that selected root without downloading.
A locally generated corpus can likewise use any root containing ``flash/``.
Missing plan or anchor files cause an error. Astra memory has no replay assets;
it cannot be used for Flash. See :ref:`Memory Management <memory-management>`.

.. code-block:: text

   memory/libero/GPT_5.5_xhigh/flash/
     object_swap_t3_anchors.json   objects and locations to locate at run time
     object_swap_t3_plan.json      actions and their anchor-relative coordinates

The task selects the plan. The seed changes the environment layout, not the
plan used for the task.

Generate a Flash plan
---------------------

The generator creates one plan from one simulator-verified successful episode.
Its two required inputs are the episode audit JSON and the matching primitive
recipe JSONL:

.. code-block:: bash

   python -m robots.libero.flash.generate \
     --audit results/goal_swap_t3_s7.json \
     --recipe results/goal_swap_t3_s7_recipe.jsonl \
     --destination memory/libero/flash

The audit must contain a non-empty ``task_language`` (or
``perturbed_task_language``) and ``libero_terminated: true``. The audit and
recipe filenames, plus the audit suite/task/seed fields when present, must
identify the same episode.

If ``segment_*.json`` readings were saved for the episode, pass their directory
with ``--segments``. Otherwise the generator derives semantic point anchors from
the instruction and the recipe's ordered pick/release or articulation
transactions. Nearby ``move_to`` and ``move_pose`` coordinates are stored as XY
offsets from those anchors, in the format consumed by Flash replay.

The relation parser supports all 80 LIBERO-PRO tasks: Spatial, Object, Goal, and
Long (``10``), across both task and swap suites. Long instructions are preserved
as ordered transactions, including dependent actions such as turning on the
stove before placement or closing an appliance after insertion.

To download only the Flash plans manually, run:

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
     --include "libero/GPT_5.5_xhigh/flash/**" --local-dir /path/to/download

Use ``--memory-profile local --memory-dir /path/to/download/libero/GPT_5.5_xhigh``
with the downloaded plans.

Run Flash Mode
--------------

Flash Mode is for evaluation only and cannot be combined with ``--explore``.
It replays a prepared plan from memory. Start a locator service first, then pass its
endpoint to RPent:

.. code-block:: bash

   rpent --robot libero --planner flash \
     --suite libero_object_swap --task 3 --seed 0 \
     --locator-endpoint http://127.0.0.1:20703

Flash replay supports the task and swap suites for LIBERO-PRO Spatial,
Object, Goal, and Long (``10``), for 80 task identities in total.

The VLA and SAM3 services use the normal LIBERO runtime configuration. You can
also connect to services that are already running with ``--vla-endpoint`` and
``--sam3-endpoint``.

Locator setup
-------------

The locator service accepts one RGB image and a query, and returns one pixel
``(x, y)`` in the input image, or ``None`` if the target cannot be located.
Choose its backend when starting the server; Flash uses the same endpoint for
all backends. The opening survey, wrist refinement, depth projection, and held
object correction use the same replay logic for every backend.

The manipulation prompt is shared with Molmo. API and Codex requests add a JSON
format instruction: ``{"point": [x, y]}`` with coordinates normalized to 0–1000,
or ``{"point": null}``. The server converts coordinates to original-image
pixels. Malformed JSON, invalid coordinates, and provider failures are errors.
Molmo retains its native point-markup parser, which returns a miss when no
valid point can be parsed.

Molmo
~~~~~

Molmo requires a newer ``transformers`` version than the LIBERO policy
environment, so run it in a separate Python environment:

.. code-block:: bash

   uv venv --python 3.11 /path/to/molmo-venv
   uv pip install --python /path/to/molmo-venv/bin/python -e ".[molmo]"

Download ``allenai/Molmo2-8B`` from `Hugging Face
<https://huggingface.co/allenai/Molmo2-8B>`_ or `ModelScope
<https://modelscope.cn/models/allenai/Molmo2-8B>`_, then start the service:

.. code-block:: bash

   export MOLMO_CHECKPOINT_PATH=/path/to/Molmo2-8B
   PYTHONPATH=/path/to/RPent /path/to/molmo-venv/bin/python \
     -m rpent.robots.components.locator_server --backend molmo \
     --transport http --host 127.0.0.1 --port 20703

Cloud API
~~~~~~~~~

The API backend uses RPent's existing provider-prefixed model names and provider
credential environment variables. Install RPent in the server environment and
select a vision-capable model; the Molmo extra and CUDA are not required:

.. code-block:: bash

   python -m rpent.robots.components.locator_server \
     --backend api --model "<provider:model>" \
     --host 127.0.0.1 --port 20703

Use ``--base-url`` to override the provider endpoint. Credentials are read on the
server, just as for the API planner; see :doc:`configure_planner` for provider
configuration. Each localization sends one image with no conversation history.

For Claude, set ``ANTHROPIC_API_KEY`` in the locator server's environment and
use an ``anthropic:`` model name. Replace ``<claude-model-id>`` with a
vision-capable Claude model available to your API account:

.. code-block:: bash

   python -m rpent.robots.components.locator_server \
     --backend api --model "anthropic:<claude-model-id>" \
     --host 127.0.0.1 --port 20703

The provider reads ``ANTHROPIC_BASE_URL`` when set; ``--base-url`` takes
precedence. This uses Claude API credentials, not a Claude Code login.

Codex login
~~~~~~~~~~~

Use the Codex installation and login on the locator server's machine. Configure
``CODEX_BIN`` if needed, then start:

.. code-block:: bash

   codex login
   python -m rpent.robots.components.locator_server \
     --backend codex --model "<model>" --reasoning-effort low \
     --host 127.0.0.1 --port 20703

Omitting ``--model`` uses Codex's configured default. Each localization uses a
fresh ephemeral thread in a temporary working directory and closes its Codex
process afterward. The image is attached directly;
the request asks for JSON coordinates and does not attach RPent tools. This
backend uses the server's Codex authentication, including an existing ChatGPT
login, rather than the RPent client's credentials. Set ``TMPDIR`` when temporary
files must stay under a particular working directory.

Timeouts, logs, and migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Flash retains the original 180-second client RPC timeout. The locator server
adds no request deadline; provider SDK defaults still apply.
The server logs its selected backend/model at startup, and the query prompt,
raw answer, image dimensions, pixel result, and elapsed time for each successful
request. A target miss is also logged. API/Codex append the JSON instruction
described above to the logged manipulation prompt.

Start the service through ``rpent.robots.components.locator_server``; it replaces
``molmo_server``. Python callers use ``LocatorClient.locate(image, query)`` from
``locator_client`` in place of ``MolmoClient.ground``. The RPC method is now
``locator.locate``, so upgrade client and server together and use
``--locator-endpoint``. Plans use ``locator: "point"`` for the selected point
backend and ``segment`` for SAM3. Existing plans using ``locator: "molmo"`` are
loaded as ``point`` and use the selected backend without editing the plan files.
The old endpoint flag is unsupported. Report model-specific results separately: the benchmark above used
Molmo2-8B and does not establish API or Codex localization performance.

Replay multiple layouts
-----------------------

Run the same Flash plan with different seeds to evaluate it on different
layouts. Reusing existing VLA, SAM3, and locator services avoids loading the
models again for every run:

.. code-block:: bash

   for seed in $(seq 0 9); do
     rpent --robot libero --planner flash \
       --suite libero_object_swap --task 3 --seed "$seed" \
       --output-dir logs/sweep/swap_t3_s$seed \
       --vla-endpoint http://127.0.0.1:20701 \
       --sam3-endpoint http://127.0.0.1:20702 \
       --locator-endpoint http://127.0.0.1:20703
   done
