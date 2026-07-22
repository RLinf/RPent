Advanced Deployment
===================

RPent normally starts and stops the LIBERO environment, VLA, and SAM3
services with each run. Keep that default for a single-machine setup. Use
external endpoints when the services run on different hosts or when the VLA
and SAM3 models should be reused across runs.

The endpoint options support the following transports:

.. list-table::
   :header-rows: 1

   * - Service
     - RPent option
     - Endpoint format
   * - LIBERO environment
     - ``--env-endpoint``
     - HTTP or socket RPC, ``[protocol://]HOST:PORT``
   * - Pi0.5 VLA
     - ``--vla-endpoint``
     - HTTP or socket RPC, ``[protocol://]HOST:PORT``
   * - SAM3
     - ``--sam3-endpoint``
     - HTTP or socket RPC, ``[protocol://]HOST:PORT``

LIBERO environment server
-------------------------

An environment server is tied to one suite, task, seed, and episode-step
limit. These values must exactly match the RPent client command. On the
environment host, run:

.. code-block:: bash

   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0
   python -m robots.libero.env_server \
     --suite libero_object_swap --task 2 --seed 0 \
     --max-episode-steps 10000 \
     --transport http --host 0.0.0.0 --port ENV_PORT

The server is task-specific. Stop it and start a new one before changing any
of the matching arguments.

Pi0.5 VLA server
----------------

On the VLA host, set the checkpoint path and start the HTTP service:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export CUDA_VISIBLE_DEVICES=0
   python -m robots.libero.vla_server \
     --transport http --host 0.0.0.0 --port VLA_PORT

The VLA server loads the model once and can be shared by multiple RPent runs.

SAM3 server
-----------

On the SAM3 host, set the local checkpoint path and start the HTTP service:

.. code-block:: bash

   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt
   export CUDA_VISIBLE_DEVICES=0
   python -m robots.libero.sam3_server \
     --transport http --host 0.0.0.0 --port SAM3_PORT

The SAM3 server loads the model once and can be shared by multiple RPent runs.

Connect RPent
-------------

On the machine running RPent, use the three endpoint options to connect to the
services above. The suite, task, seed, and maximum episode steps must match the
environment server command:

.. code-block:: bash

   rpent \
     --suite libero_object_swap --task 2 --seed 0 \
     --libero-type pro --max-episode-steps 10000 \
     --env-endpoint http://ENV_HOST:ENV_PORT \
     --vla-endpoint http://VLA_HOST:VLA_PORT \
     --sam3-endpoint http://SAM3_HOST:SAM3_PORT \
     --planner claude_code --model claude-opus-4-7

Replace each ``*_HOST`` with the address of the machine running that service,
and make sure it is reachable from the machine running RPent. Replace each
``*_PORT`` with the available port selected when starting the service. The
three endpoint options are independent; when one is omitted, RPent starts that
service on the current machine and selects an available port automatically.
For all three services, omit the protocol to use HTTP, or use
``socket://HOST:PORT`` for socket RPC.
