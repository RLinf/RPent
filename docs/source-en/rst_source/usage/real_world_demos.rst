Real-World Demos
================

See the Shape of an Embodied Agent
----------------------------------

These three edited real-robot demos show RPent re-observing the scene and
changing its action when the goal changes, a path is blocked, or a skill must
be recombined: placing the same blue plate in a new cardboard box, reading a
bottle and bag label while checking the grasp, and reusing a placing skill to
pour steel beads. They show a loop of observation, decision, action, and
correction rather than a replay of fixed motions, and how a successful run can
become a reusable capability.

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
