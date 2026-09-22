Real-World Demos
================

From Digital Agents to the Physical World
-----------------------------------------

Coding agents such as Codex and Claude Code have made a productive pattern
familiar: a model understands the goal, breaks it into steps, calls tools,
checks what happened, and keeps adjusting. A robot cannot simply roll back a
bad action. Its scene keeps changing, its observations are incomplete, and a
grasp or contact changes the world for real. Planning is only one part of the
problem; fast and precise specialist controllers are still needed for the
physical execution.

RPent connects these pieces. It combines the task understanding and planning
of a general model with the fine-grained manipulation of VLA specialists,
memory, tools, and robot interfaces. The result is a loop that can observe,
decide, act, and correct instead of replaying one fixed action sequence. RPent
is an open-source embodied-agent infrastructure jointly initiated by Tsinghua
University, Wuwen Xinqiong, and Zhengxing Innovation, and is built by the core
team behind the RLinf embodied reinforcement-learning framework. See the
`RPent repository <https://github.com/RLinf/RPent>`_ for the code.

See the Shape of an Embodied Agent
----------------------------------

Start with the three clips below. A blue plate is sent to a new destination, a
robot reads both a bottle and its target bag, and a learned placing skill is
reused to pour steel beads. The point is not that the robot has memorized
three more motions. The point is that the task can change while the robot is
working, and the agent can reorganize what it knows around the new goal.

The videos are edited showcase clips rather than benchmark results. For setup
and reproduction details, see the real-robot guides.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 1.5rem 0;">
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4">Download the video</a>.
       </video>
       <h3>Same plate, a new destination</h3>
       <p><strong>Goal:</strong> Move the clean blue plate into the cardboard box.</p>
       <p>The same blue plate now has a new destination. RPent looks again, chooses the cardboard box, checks the outcome, and corrects its placement instead of replaying the frozen VLA route.</p>
     </div>
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4">Download the video</a>.
       </video>
       <h3>Read the bottle. Read the bag.</h3>
       <p><strong>Goal:</strong> Sort a Pepsi bottle into the bag labelled Pepsi.</p>
       <p>RPent reads the labels, gives the bottle a small test lift to verify the grasp, and changes the wrist posture when the original path is blocked.</p>
     </div>
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4">Download the video</a>.
       </video>
       <h3>From placing to pouring</h3>
       <p><strong>Goal:</strong> Pour steel beads from a container into a bowl.</p>
       <p>RPent recombines a learned placing skill for a new pouring task, then coordinates the arms around the shared workspace instead of requiring a new task-specific policy.</p>
     </div>
   </div>

From a Successful Run to a Reusable Skill
------------------------------------------

When an exploration succeeds, RPent can preserve the verified task logic as a
Task Card. On a later run, Flash Mode can replay that card while re-localizing
its anchors from the current visual state. The agent does not need to call the
large model for every small movement; it reuses a proven flow and only asks
for help when the scene requires a change. This turns one successful run into
an experience that can be used again.

That is the larger question behind these demos: when new objects, goals, and
obstacles appear outside the training distribution, can a robot reorganize
existing skills instead of waiting for a new policy? RPent is designed to move
from executing a fixed instruction toward understanding the goal, calling the
right capability, adapting to the scene, and turning success into reusable
experience.
