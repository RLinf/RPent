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
