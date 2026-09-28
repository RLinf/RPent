Remote Services
===============

By default, RPent starts and stops the environment, VLA, and SAM3 services
with each LIBERO run. Keep that default for single-machine use. Configure
external endpoints only when services live on different hosts, or when you
want to reuse VLA and SAM3 models across tasks.

Three flags set the endpoints: ``--env-endpoint`` for the LIBERO
environment, ``--vla-endpoint`` for the Pi0.5 VLA, and ``--sam3-endpoint``
for SAM3. Each takes ``[protocol://]HOST:PORT`` — HTTP when the protocol is
omitted, or ``socket://`` for socket RPC.

Dashboard Sessions do not support ``--env-endpoint`` because every TaskRun
uses a fresh environment service. ``--vla-endpoint`` and ``--sam3-endpoint``
remain available in Dashboard mode.

LIBERO environment service
--------------------------

One environment service is pinned to a suite, task, seed, and max episode
steps; those values must match the RPent client exactly. On the env host:

.. code-block:: bash

   export LIBERO_TYPE=pro
   python -m robots.libero.env_server \
     --suite libero_object_swap --task 2 --seed 0 \
     --max-episode-steps 10000 \
     --cuda-device 0 \
     --transport http --host 0.0.0.0 --port ENV_PORT

The environment service is task-bound. To change any of those parameters,
stop the old service and start a new one.

Pi0.5 VLA service
-----------------

On the VLA host, set the checkpoint path and start the HTTP service:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   python -m rpent.robots.components.pi05_vla_server \
     --embodiment libero \
     --cuda-device 0 \
     --transport http --host 0.0.0.0 --port VLA_PORT

The VLA service loads the model once and can be reused by multiple RPent runs.

SAM3 service
------------

On the SAM3 host, set the local checkpoint path and start the HTTP service:

.. code-block:: bash

   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt
   python -m rpent.robots.components.sam3_server \
     --cuda-device 0 \
     --transport http --host 0.0.0.0 --port SAM3_PORT

The SAM3 service loads the model once and can be reused by multiple RPent runs.

Connect RPent
-------------

On the machine that runs RPent, point at the three endpoints. Suite, task,
seed, and max episode steps must match the environment service:

.. code-block:: bash

   rpent \
     --robot libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --libero-type pro --max-episode-steps 10000 \
     --env-endpoint http://ENV_HOST:ENV_PORT \
     --vla-endpoint http://VLA_HOST:VLA_PORT \
     --sam3-endpoint http://SAM3_HOST:SAM3_PORT \
     --planner claude_code --model claude-opus-4-8

Replace each ``*_HOST`` with a reachable address of the machine that runs
that service, and each ``*_PORT`` with the free port you chose at startup.
Any of the three endpoint flags can be omitted; when one is unset, RPent
spawns that service locally on a free port. All three default to HTTP when
the protocol is omitted, and all three accept ``socket://HOST:PORT``.

Custom RLinf checkout
---------------------

The environment and VLA services import ``rlinf``. When starting them
manually against a development RLinf checkout instead of the installed
package, point ``PYTHONPATH`` at the checkout first:

.. code-block:: bash

   export PYTHONPATH=/path/to/rlinf:$PYTHONPATH

Servers spawned by RPent get this automatically: the checkout is resolved
from ``RPENT_RLINF_ROOT`` (or ``RLINF_REPO_PATH``), falling back to the
``rlinf`` directory next to the RPent checkout. A resolved path that does
not exist is harmless — Python ignores invalid ``PYTHONPATH`` entries —
so the servers import the installed ``rlinf`` package.

.. _libero-parallel-eval:

Parallel evaluation
-------------------

The following example demonstrates parallel evaluation with LIBERO using
Pi0.5 VLA and SAM3. Other robots or evaluation setups may use different
services and endpoints.

To run the same LIBERO task concurrently, first start one Pi0.5 VLA service
and one SAM3 service as described above. After both services report the log line
``RPC server listening on ...``, pass the same endpoints —
``http://VLA_HOST:VLA_PORT`` and ``http://SAM3_HOST:SAM3_PORT`` — to every
concurrent ``rpent`` process, replacing the placeholders with the host addresses
and ports of the corresponding services.

When the services and RPent run on the same machine, the hosts can be
``127.0.0.1``. All processes then use the same VLA and SAM3 services. Leave
``--env-endpoint`` unset: each process starts its own ``env_server`` and keeps its
evaluation environment independent. The VLA and SAM3 models are loaded once, so
their services do not need to be started again for each run.

.. code-block:: bash

   pids=()

   for i in $(seq 1 10); do
     rpent --robot libero --libero-type pro \
       --suite libero_object_swap --task 2 --seed 0 \
       --planner claude_code --model claude-opus-4-8 \
       --vla-endpoint http://VLA_HOST:VLA_PORT \
       --sam3-endpoint http://SAM3_HOST:SAM3_PORT \
       --output-dir logs/parallel_object_swap_t2_s0/run_$i &
     pids+=($!)
   done

   wait "${pids[@]}"

.. note::

   For long-running evaluations over SSH, start the shared services with
   ``nohup`` or in a ``tmux`` / ``screen`` session; a bare ``&`` may stop them
   when the SSH shell exits.
