RoboDojo A/B Protocol: Bare Pi_05 vs RPent Harness
===================================================

Purpose
-------

Quantify the increment of the RPent harness (agent loop + toolchain + safety
monitoring + memory) over the bare Pi_05 policy on RoboDojo tasks. The bare
Pi_05 baseline is the official evaluation flow: Isaac Sim sends observations
to the Pi_05 policy service over WebSocket, the policy returns action chunks,
and the loop closes until episode end — no LLM, no tools, no safety
monitoring.

Scene alignment (key mechanism)
-------------------------------

Both sides use the same official layout files, so scenes match episode by
episode:

* Official layout files: ``<robodojo-modelscope>/Assets/Eval_Layout/RoboDojo/
  <env_cfg>/<seed>/<task>_<N>.json`` (N is the layout id, 0..54);
* Bare side: ``eval_env.py`` honors the ``ROBODOJO_LAYOUT_IDS="N1,N2,..."``
  allowlist;
* Harness side: ``env_server.py --layout N`` →
  ``env.reset(seed=N)`` → ``seed_manager.get_seed_scene_info(N)`` → the same
  layout file.

Thus ``ROBODOJO_LAYOUT_IDS="0,1,2"`` bare runs are equivalent to harness runs
with ``--layout 0/1/2``.

Standard commands
-----------------

Bare Pi_05 baseline (official eval client):

.. code-block:: bash

   cd <RoboDojo source>
   ROBODOJO_LAYOUT_IDS="0,1,2,3,4,5,6,7,8,9" \
     bash scripts/robodojo.sh eval \
       --policy-dir XPolicyLab/policy/Pi_05 \
       --task <TASK> --ckpt RoboDojo-sim-arx_x5-joint-0 \
       --policy-env uv --eval-env <sim env> \
       --action-type joint --seed 0 --policy-gpu 0 --env-gpu 0 --eval-num <N>

Results land in ``eval_result/RoboDojo/<TASK>/Pi_05/arx_x5/.../_result.json``
(success_rate / score / per-episode layout_id + score) plus per-episode
three-camera mp4s.

Harness (RPent agent):

.. code-block:: bash

   cd <rpent checkout>
   export PYTHONPATH="$PWD" PATH="$PWD/.venv/bin:$PATH" \
     SAM3_CHECKPOINT_PATH=$PWD/checkpoints/sam3/sam3.pt \
     HF_HUB_DISABLE_XET=1 CELL_TIMEOUT_S=3600
   rpent --env robodojo --task <TASK> --layout <N> \
     --sim-device 0 --planner codex --model deepseek-v4-flash --max-turns 30 \
     --output-dir <ws>/evidence/<run_tag>

Results land under ``--output-dir``: ``audit*.json`` (reward_details / score),
``videos/*.mp4``, and service logs.

Reporting conventions
---------------------

One table per task (at least 10 bare episodes; the harness covers at least a
documented subset of the same layouts):

.. code-block:: text

   | layout | bare Pi_05 score | harness score | bare Pi_05 failure mode | harness increment |

Rules:

* scores are compared episode by episode (official reward tiers, e.g.
  put_bottles 0/25/40/100);
* success is determined by the official reward (``is_success`` / score>=1.0),
  never by the agent's self-report;
* small samples (<=10) are indicative only; official conclusions require the
  50-episode protocol.

Known baselines (local official bare runs)
-------------------------------------------

``put_bottles_into_dustbin`` (standard, seed 0):

.. list-table::
   :header-rows: 1

   * - N
     - SR
     - score
   * - 45
     - 64.4%
     - 75.1
   * - 16
     - 43.8%
     - 64.4
   * - 9
     - 66.7%
     - 76.7

``stack_bowls_random``:

.. list-table::
   :header-rows: 1

   * - N
     - SR
     - score
   * - 300
     - 6.7%
     - 15.9
   * - 30
     - 16.7%
     - 24.2

Note: bare small-sample variance is large (10 episodes can be 0 or 100), so
comparisons must lock the same layout list.

Current progress
----------------

See :doc:`integration_log` (A/B validation records).
