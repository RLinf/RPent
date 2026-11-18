## Policy protocol

vpp2_act(instruction, chunks=1) executes the supplied instruction. Describe the
visible object, action, destination or goal relation, preferably in English.
Include required arm roles, dependencies and what the other hand must preserve.
Keep the current phase consistent with the complete task; do not add unrelated
actions. VPP2 predicts 32 dual-arm targets and executes the first 24 per chunk,
then replans for each additional chunk. You inspect the result after the call.
History includes preceding Primitive and VPP2 actions; changing instructions
does not start a new episode. If history is unavailable after uncertain
execution, do not retry on stale feedback: inspect the scene and use fresh
analytic actions or follow the current mode's reset rules.

- VPP2: grasping, re-grasping, receiver grasp, bimanual coordination, insertion,
  articulation, tool use and motion requiring sustained contact.

Primitive staging can precede VPP2; a failed VPP2 call is not a prerequisite.
Before switching, check target identity, each hand's contents, contact and
protected relations. After a primitive, inspect the actual state before
continuing. Use VPP2 for uncertain contact geometry rather than inventing poses.
Do not interrupt correct contact and useful progress with speculative
primitives. An unfinished phase alone does not require switching controllers.
After an unproductive chunk, inspect whether to continue, clarify the
instruction, improve visibility or correct the physical state.

Choose chunks by how soon the next decision needs fresh evidence:
- 1: a new approach, uncertain progress, instability, near contact or completion,
  after a corrective primitive, or before a release/handover decision.
- 2: ordinary stable progress within a phase whose target and arm roles are clear.
- 3: a successful recipe supports a continuous phase already progressing
  correctly, with no intervening decision requiring your inspection.
Shorten the next call when those conditions change. These are selection bands,
not guarantees that a phase finishes within a fixed number of chunks.
Also account for the remaining action budget: 1, 2 and 3 chunks normally
execute 24, 48 and 72 actions, respectively. Choose a call length that leaves
room for subsequent required actions. Near completion or with little budget
margin, prefer one chunk and inspect the result before continuing. Use the
returned action count and remaining_steps to assess actual consumption.

Control: analytic joint targets and VPP2 native EE16 targets.

## Exploration only: retry and evidence

Apply this section only in exploration mode. It does not authorize retries or
memory writing during evaluation.

Change a named variable on each retry: instruction, arm roles, action order,
chunk count, localization or approach pose.

In memory control choices, record VPP2 instructions, Primitive use, chunk choices and observed reasons for continuing or switching.
