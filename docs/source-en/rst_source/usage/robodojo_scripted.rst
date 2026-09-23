Scripted RoboDojo Control
========================================

This standalone runner executes a trusted Python recipe against an exclusively
owned RoboDojo development server. CuRobo provides server-side IK. The entry
points are Python modules run from a source checkout; this integration is not
registered with the planner CLI or Dashboard and does not start a VLA service.

Installation
------------

Use Linux, Python 3.11 and a compatible NVIDIA driver for the simulator.
From the RPent checkout, create a dedicated environment:

.. code-block:: bash

   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[robodojo-sim]" --extra-index-url https://pypi.nvidia.com

The extra installs the RLinf environment adapter and
``rlinf-robodojo-runtime`` from Git. Runtime 0.3.0 or later supplies the
simulation bridge and vendored RoboDojo code; it owns the Isaac Sim / IsaacLab
dependency versions. RPent supplies the cuRobo Git reference and uv build
dependency settings. Run uv from the checkout so those settings apply.
Install each robot stack in its own environment. Git branch references can
change; keep the resolved commits and installed dependency versions with results.

Dependency overrides reconcile simulator metadata with RPent and RLinf.
Resolution or import checks do not establish simulator compatibility.
``uv pip check`` still reports the deliberately overridden IsaacLab Starlette
pin and Isaac Sim's typing-extensions, uvicorn and wrapt pins. These upstream
metadata conflicts remain; GPU compatibility needs separate validation.
The current IsaacLab wheel omits ``config/extension.toml``, which its package
initializer requires. Before starting the server, install the same resolved
revision in editable mode. Keep this checkout available for the environment's
lifetime. The commands below read the revision from the Git installation metadata;
run them before replacing that installation:

.. code-block:: bash

   ISAACLAB_REV=$(python -c 'import importlib.metadata as m, json; print(json.loads(m.distribution("isaaclab").read_text("direct_url.json"))["vcs_info"]["commit_id"])')
   git clone --branch main --single-branch https://github.com/yuechen0614/IsaacLab.git /path/to/IsaacLab
   git -C /path/to/IsaacLab checkout --detach "$ISAACLAB_REV"
   uv pip install --no-deps \
     -e /path/to/IsaacLab/source/isaaclab \
     -e /path/to/IsaacLab/source/isaaclab_assets \
     -e /path/to/IsaacLab/source/isaaclab_tasks

``--no-deps`` here preserves the dependencies already resolved by the extra;
it is not a replacement for installing the simulator stack above.

Scene assets are separate. Download ``Assets/**`` from the
`RoboDojo dataset <https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo>`_
and set ``ROBODOJO_ASSETS_ROOT`` to the directory containing ``Assets/``.
Check robot, object, material and saved-layout files are present, not LFS
pointers. NVIDIA USD/material assets referenced by IsaacLab also need to be
available; follow the simulator's asset setup instructions. For offline use,
``ROBODOJO_USD_ASSET_PREFIX`` handles RoboDojo's ``Assets/Isaac/5.0`` URL
rewrites and must point to a matching asset pack; configure other IsaacLab
asset URLs separately.

Start an exclusive dev server
----------------------------------------

In a dedicated terminal, from the checkout:

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/scene-data
   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000
   python -m robots.robodojo.env_server \
     --task general_pickup --layout 0 --mode dev \
     --max-episode-steps 200 --headless --enable_cameras \
     --host localhost --port 18765 --transport http \
     --save-dir /path/to/server-output --video-dir /path/to/videos

The default code root is ``robodojo_runtime.source_root()``. Use
``--source-root`` only to select an explicit alternative, and freeze that same
tree below. The settling variable changes initialization physics; historical
success records used 1000 steps. Disclose this setting with results.

Wait for the server to become ready. Keep it exclusive: no Dashboard, second
recipe or other client may act on it. Use a fixed layout, without ``--random``.
The runner requires step zero and an exactly matching task, layout and step
limit, then performs one reset. It does not authenticate the loaded server
code, enforce exclusive ownership, start, stop or reconfigure the server.
Stop the server after evaluation to flush its camera videos.

Freeze, verify and run
----------------------

Write and review a Python recipe defining ``main(env, output)`` first.
The recipe path below is your own file, not a bundled successful policy.
In a second terminal using the same environment and checkout:

.. code-block:: bash

   SIM_SOURCE=$(python -c 'from robodojo_runtime import source_root; print(source_root())')
   python -m robots.robodojo.scripted.eval freeze \
     --output /path/to/new-release --source-root "$SIM_SOURCE" \
     --recipe /path/to/recipe.py --task general_pickup --layout 0 \
     --step-limit 200 --action-budget 180
   python -m robots.robodojo.scripted.eval verify --output /path/to/new-release
   python -m robots.robodojo.scripted.eval run \
     --output /path/to/new-release --endpoint http://localhost:18765

``freeze`` embeds the recipe and numerical primitives, applies the static
reward-boundary audit and hashes local sources, the interpreter and a dependency
inventory. ``verify`` checks those bytes and the simulator source inventory.
These are local fingerprints, not a portable package, dependency reinstall or
attestation of the live server. External scene assets and installed package
contents are not comprehensively hashed. Keep the original paths available.

``run`` claims one attempt before contacting the server; even a failed preflight
consumes the claim. A claimed release cannot be refrozen or retried. Pre-claim
refreezing archives the old manifest. A new experiment requires a new release
directory and a fresh episode; retain earlier results when reporting attempts.

The worker sees only public RGB-D, calibration, proprioception, instruction and
step budgets. Reward and done slots from ``env.step`` are ``None``. After worker
exit and action-pipe closure, a separate evaluator reads ``env.is_success()``
and reward. Inspect ``trial/result.json``: ``valid`` describes execution and
hash checks, while ``official_success`` is the native task predicate. A zero
CLI exit status alone does not establish either. Logs and audits are under
``trial/``; recipe artifacts are under ``trial/agent/``. This protocol is for
trusted code and is not an OS security sandbox or a leaderboard submission.

Recipes
-------

``robots/robodojo/recipes/README.md`` records evidence locations and hashes for
two historical successful JSON plans: ``general_pickup`` layout 1 (94/200
actions) and ``pour_by_language`` layout 1 (690/800). The plans are preserved
under ``recipes/historical/``. Their old controller required pose IK and a
different action interface, so these JSON files cannot be passed to this
runner. No Python recipe with demonstrated success on the current interface
is bundled. No simulator rollout was performed to validate this extraction.

Add your own reviewed Python file under ``recipes/`` or at an external path.
It may call ``env.get_obs()``, ``env.get_status()``,
``env.solve_ik_position(arm, xyz)`` and ``env.step(action)``. Actions use native
joint or end-effector keys. ``solve_ik_pose``, reset and evaluator RPCs are not
available to the worker. Reserve stdout for the bridge; log to stderr or write
under ``output``. The static import allowlist is in ``scripted/reward_audit.py``.

``move_to`` and ``set_gripper`` are embedded from ``scripted/primitives.py``;
do not import repository modules in a recipe. Their first argument is an object
with ``env``, ``_last_obs`` and ``_check_cancelled()``; their second argument is
unused and may be ``None``. For example, this interface-only recipe opens the
right gripper and does not claim task success:

.. code-block:: python

   class Motion:
       def __init__(self, env):
           self.env = env
           self._last_obs = None

       def _check_cancelled(self):
           status = self.env.get_status()
           if status["step"] >= status["step_limit"]:
               raise RuntimeError("episode budget exhausted")

   def main(env, output):
       motion = Motion(env)
       set_gripper(motion, None, arm="right", gripper=-1)

For ``move_to``, positive gripper values close, negative values open and zero
preserves the current position. For ``set_gripper``, zero also opens.
The outer bridge enforces the frozen action budget regardless of recipe checks.
Before adding a success claim, preserve the release, evaluator result, action
trace, initialization settings and video evidence for the exact task/layout.

Offline checks
--------------

.. code-block:: bash

   uv pip install -e ".[test]"
   pytest tests/unit_tests/robots/robodojo -v
   ruff check --preview robots/robodojo tests/unit_tests/robots/robodojo

These CPU tests cover source freezing, reward filtering, single-attempt claims,
worker/evaluator ordering and environment adapters with fakes. They do not
start Isaac Sim or establish GPU rollout success.
