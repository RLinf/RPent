## Policy protocol

awomo05_act(chunks=1) executes the original full task instruction using dual-arm
joint targets. It has no instruction parameter and cannot accept a rewritten
instruction or a phase-specific subtask. Each chunk executes 8 native actions,
then replans from real feedback. You inspect the result after the call.
History includes preceding Primitive and Awomo05 actions and persists across
calls; switching controllers does not start a new episode. A completed chunk
means commands executed, not task success. If execution is uncertain or model
history synchronization is lost, stop all motion, including Primitive actions,
and follow the current mode's reset or finish rules. Do not reset model history
independently of the environment.

- Awomo05: continue the original full-task policy when current observations
  support useful progress toward the task. A phase plan helps you choose when
  to call, inspect or interrupt the policy; it cannot be sent as a subtask.

Primitive staging can precede Awomo05; a failed Awomo05 call is not a prerequisite.
Before switching, check target identity, each hand's contents, contact and
protected relations. After a primitive, inspect the actual state before
continuing. For uncertain contact geometry, consider continuing the original
full-task policy when observations support it; do not invent contact poses or
assume Awomo05 can be instructed to perform only the next phase.
Do not interrupt correct contact and useful progress with speculative
primitives. An unfinished phase alone does not require switching controllers.
After an unproductive chunk, inspect whether to continue, change the
chunk count, improve visibility or correct the physical state.

Choose chunks by how soon the next decision needs fresh evidence:
- 1: a new approach, uncertain progress, instability, near contact or completion,
  after a corrective primitive, or before a release/handover decision.
- 2: ordinary stable progress within a phase whose target and arm roles are clear.
- 3: a successful recipe supports a continuous phase already progressing
  correctly, with no intervening decision requiring your inspection.
Shorten the next call when those conditions change. These are selection bands,
not guarantees that a phase finishes within a fixed number of chunks.
Also account for the remaining action budget: 1, 2 and 3 chunks normally
execute 8, 16 and 24 actions, respectively. Choose a call length that leaves
room for subsequent required actions. Near completion or with little budget
margin, prefer one chunk and inspect the result before continuing. Use the
returned action count and remaining_steps to assess actual consumption.

Control: analytic joint targets and Awomo05 native joint targets.

## Exploration only: retry and evidence

Apply this section only in exploration mode. It does not authorize retries or
memory writing during evaluation.

Change a named variable on each retry: Primitive action order, policy/Primitive
switching point, chunk count, localization, approach pose or arm roles where
the native task instruction permits a choice.

In memory control choices, record the original full task instruction, observed phase and
  state at Awomo05 takeover, Primitive use, chunk choices and observed reasons for continuing or switching.
