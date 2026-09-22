Real-World Demos
================

See the Shape of an Embodied Agent
----------------------------------

Across this set of videos, RPent handles open-ended tasks: pouring steel beads into a bowl, coordinating both arms to wipe and store a plate, and moving an occluding object so a hidden target becomes visible and reachable.

These are not separate policies trained for every scene. The important change is that the robot moves from executing a learned motion to understanding the task, calling a capability, and changing its action as the scene changes.

When the task changes to “put the clean dishes in the cardboard box,” a frozen VLA may replay the learned route and place the same blue plate in the metal basket. RPent first observes the current scene, selects the new destination, checks the result, and re-localizes the plate when visual positioning drifts.

In another task, the robot reads the brand labels on a bottle and a bag, performs a small test lift to verify the grasp, and changes its wrist posture when the original path is unavailable. When a spoon is hidden under bowls, it moves the bowls away before attempting the grasp. This is an observation, decision, action, and correction loop rather than a pre-written sequence.

RPent can also recombine a skill originally used to pick up and place a container for pouring steel beads, and combine grasping with dual-arm contact for plate wiping and storage.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 1.5rem 0;">
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4">Download the video</a>.
       </video>
       <h3>Same plate, a new destination</h3>
       <p>When the task changes to putting clean dishes in the cardboard box, a frozen VLA may replay its learned route and place the blue plate in the metal basket. RPent observes the new scene, selects the cardboard box, checks the result, and re-localizes the plate when visual positioning drifts.</p>
     </div>
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4">Download the video</a>.
       </video>
       <h3>Read the bottle. Read the bag.</h3>
       <p>The robot reads the brand labels on the bottle and bag, performs a small test lift to verify the grasp, and changes its wrist posture when the original path is unavailable.</p>
     </div>
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4">Download the video</a>.
       </video>
       <h3>From placing to pouring</h3>
       <p>RPent recombines a skill originally used to pick up and place a container for pouring steel beads, and can combine grasping with dual-arm contact for plate wiping and storage.</p>
     </div>
   </div>

From a Successful Run to a Reusable Skill
------------------------------------------

After an exploration succeeds, RPent can preserve the verified task logic as a
Task Card. Flash Mode can replay that card on a later run, re-localizing its
anchors from the current visual state and asking the large model for help only
when the scene changes. Reusing the proven flow avoids a model call for every
small movement and reduces execution latency.

These demos are not about teaching a robot a few more motions. They ask a
larger question: when new objects, goals, and obstacles appear outside the
training distribution, can a robot reorganize existing skills instead of
waiting for a new policy? RPent is designed to move from executing a fixed
instruction toward understanding the goal, calling the right capability,
adapting to the scene, and turning success into reusable experience.
