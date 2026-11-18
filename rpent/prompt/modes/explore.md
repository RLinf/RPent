## Exploration

This is session {{session_number}} of {{session_max}}.
Attempt limit for this session: {{explore_attempts_per_session}}.

Learn a strategy through observed attempts. Prefer safe in-place recovery that
preserves useful progress. Before resetting or handing off, record the attempt's
evidence and next change. Each new attempt must change a named factor: order,
target, approach geometry, controller or tested parameter. An unchanged plan
does not test a new hypothesis. Inspect action results and re-localize after
scene restoration. A failed approach is evidence about that approach, not proof
that the task is impossible.

## Attempts and restoration

In simulation, the initial episode counts toward the attempt limit. Follow the
environment's interpretation of that limit and its other run limits.
For a finite attempt budget, an unsolved finish is refused while attempts remain.
Archive the failed attempt, reset with a changed plan and re-observe. The robot's
reset contract determines which task and scene settings are preserved. Once
success is confirmed, stop manipulation and use remaining turns for artifacts
and completion.

Before success, if repeated failed variations have revealed a concrete fix,
consider whether a fresh simulated attempt would test it more clearly. Weigh
the remaining attempt budget and useful progress a reset would discard. Record
that choice and close out the evidence before resetting; there is no fixed
failure count that requires a reset.

For a physical robot, a new planner or toolkit does not restore the
physical scene. Follow the robot's operator readiness and restoration protocol,
then inspect fresh observations before further motion. Operator abort, hardware
faults and safety stops override the attempt budget. Never fabricate readiness
receipts or a success verdict.

Do not assume task solvability unless the benchmark explicitly guarantees it.
When preparing a handoff, preserve exhausted approaches and next proposed changes
in useful notes. The runtime's attempt and finish guards determine when the
session can end; running out of ideas does not bypass them.

## Completion

Follow the robot's authoritative success and stop rules. On confirmed success,
stop task motion and prepare the audit and eligible memory from recorded evidence.
If unsolved, report it honestly without claiming a winning trajectory. Save
required artifacts and handoff notes before finishing.
