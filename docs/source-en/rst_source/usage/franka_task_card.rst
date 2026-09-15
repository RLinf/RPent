Franka Task Cards
=================

Single and dual Franka share a step-based generator and use the existing
``--planner task_card`` / ``RobotSpec.replay_card`` interface. The LIBERO
executor and its XY/grasp heuristics are not used: Franka needs calibrated
3-D targets and explicit per-arm coordinates. Cards are separate versioned JSON
files, not interchangeable with LIBERO's plan/anchor files.

Generation
----------

Record one successful attempt with the normal Franka planner. Its output
directory must contain a complete ``states.json`` and the original RGB-D and
camera metadata artifacts. Keep the robot YAML and calibration used for that run.
For every ``move_delta`` and ``rotate_delta`` source step, supply an annotation
JSON. The recording agent can author this from its visible reasoning and images;
review its interpretation before generation. Example (use actual step numbers):

.. code-block:: json

   {
     "3": {"intent": "approach cup", "phrase": "the red cup", "camera": "base"},
     "5": {"intent": "improve wrist view", "rotation_mode": "relative"}
   }

Dual-Franka projection cameras are ``base`` and ``d455``; single-Franka cameras
are ``wrist`` and ``third_person``. Each move is localized in the source image
immediately before the motion. The generator stores the achieved TCP offset
from that point. During replay Molmo locates the phrase again on a fresh image;
existing depth/calibration helpers recover the 3-D target. Both arms use the
current shared ``right_base`` motion contract; left-arm workspace checks transform
the target into ``left_base``. Legacy unlabelled arm-frame records are rejected.
Rotation mode ``relative`` retains the recorded delta and checks the starting
orientation (0.15 rad tolerance); ``fixed`` reconstructs the recorded target
quaternion using rotation composition. Object-relative rotation is not inferred;
such steps need an appropriate VLA primitive or a future direction estimator.

Generate offline (only the Molmo service is contacted):

.. code-block:: bash

   python -m robots.franka.task_card.generate \
     --robot dual_franka --task dual_franka_t1 \
     --run-dir logs/SOURCE_RUN --annotations annotations.json \
     --robot-config /path/to/robot.yaml \
     --calibration-path /path/to/hand_eye_calibration.json \
     --molmo-endpoint http://MOLMO_HOST:PORT \
     --destination memory/dual_franka/task_card/cup.json

The terminal asks whether the source attempt succeeded. Failure/stuck agent
verdicts, failed replay outcomes, unsupported actions, gaps in step indices and
reported primitive failures are rejected. Contiguous indices alone cannot prove
that an unrecorded hardware action never occurred; review the transcript too.
Annotations are required rather than guessed from motion coordinates.

Replay
------

.. code-block:: bash

   rpent --robot dual_franka --task-id 1 --planner task_card \
     --task-card memory/dual_franka/task_card/cup.json \
     --robot-config /path/to/robot.yaml \
     --calibration-path /path/to/hand_eye_calibration.json \
     --molmo-endpoint http://MOLMO_HOST:PORT \
     --vla-endpoint http://VLA_HOST:PORT

For single-arm generation use ``--robot franka --task franka_t1``; replay with
``--robot franka --task-id 1``. Single-arm VLA calls require the external VLA
endpoint. Normal dual-arm local VLA configuration also works. Existing hardware
setup applies.

An operator terminal is required. First confirm scene restoration and permission
to initialize/reset the environment, then confirm execution after initialization.
At completion, only the human answer ``success`` marks the card solved. Any
primitive/projection error ends the sequence as a failure without further motion.
The executor preserves serial primitive order; a dual-arm VLA call remains one
primitive and predicts fresh actions. It does not introduce parallel scheduling.

Replay saves ``task_card_outcome.json`` (human verdict and source/replay step
mapping), ``task_card_recipe.jsonl`` (commands actually attempted), and the usual
``states.json``, images and planner transcript. Failed runs also retain these
artifacts; their JSONL is not a successful recipe. Source files are left intact.
Configuration/calibration hashes must match. Analytic translations are limited
to 0.20 m per step and configured workspace bounds, rotations to 0.35 rad.
These checks do not implement collision detection or a hardware emergency stop.

This implementation uses terminal confirmations, not Dashboard confirmation
widgets; omit ``--dashboard`` and ``--interactive``. Upstream dual-Franka exploration
remains available separately; task-card replay does not automatically publish memories.
Current dual-arm VLA steps are ``vla_right_grasp``, ``vla_handoff`` and
``vla_left_place``; single-arm uses ``vla_grasp``. Hardware validation is required
before relying on a generated card; automated tests use mocked robots/grounding.
