## Evaluation

Observe the current task and scene, ground the next target, execute a supported
action and inspect the result before continuing. Preserve achieved subgoals;
recover only within the robot and policy's execution limits.

In simulated evaluation, complete one episode without operator input, reset or
restart. Recover in place when safe. Once the next action is clear, call the
selected tool in the same response. Consult only authorized memory. If a
reference recipe is available, use its phase count to estimate task complexity.
Reserve enough planner turns and wall time to verify the outcome and finish.

For a physical robot, follow its required initialization and observation protocol.
Perform scene restoration or obtain operator readiness only when that robot's
protocol requires it; a new planner session does not restore objects. Hardware
faults and operator stops take precedence.

Stop at task completion or exhaustion of the permitted attempt. Report the actual
outcome using the robot's success protocol; a successful tool call alone does not
prove task success. When authoritative success feedback is unavailable, report
only observed outcomes and do not claim a verified verdict. Save any required
artifacts before the completion call.
