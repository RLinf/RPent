Real-World Demos
================

See the Shape of an Embodied Agent
----------------------------------

The three videos below show a changing goal, a blocked path, and the reuse of an existing skill. Each explanation sits directly below the task it describes.

.. raw:: html

   <div style="display: flex; flex-direction: column; gap: 2rem; margin: 1.5rem 0;">
     <article style="border: 1px solid var(--color-foreground-border); border-radius: 0.6rem; overflow: hidden;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4">Download the video</a>.
       </video>
       <div style="padding: 1rem 1.25rem;">
         <h3 style="font-size: 1.15rem;">When the goal changes, the plate finds a new home</h3>
         <p>When the task changes to putting clean dishes in the cardboard box, a frozen VLA may replay its learned route and place the same blue plate in the metal basket. RPent first observes the current scene, selects the new destination, checks the result, and re-localizes the plate when visual positioning drifts: when the task changes, the robot changes its action.</p>
       </div>
     </article>
     <article style="border: 1px solid var(--color-foreground-border); border-radius: 0.6rem; overflow: hidden;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4">Download the video</a>.
       </video>
       <div style="padding: 1rem 1.25rem;">
         <h3 style="font-size: 1.15rem;">Read the labels, place it right</h3>
         <p>In another task, the robot reads the brand labels on a bottle and a bag, sorts each drink into the matching bag, and performs a small test lift to verify the grasp. When the original path is unavailable, it changes its wrist posture; when a spoon is hidden under bowls, it moves the bowls away before attempting the grasp.</p>
         <p>This is an observation, decision, action, and correction loop rather than a pre-written sequence.</p>
       </div>
     </article>
     <article style="border: 1px solid var(--color-foreground-border); border-radius: 0.6rem; overflow: hidden;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4">Download the video</a>.
       </video>
       <div style="padding: 1rem 1.25rem;">
         <h3 style="font-size: 1.15rem;">From placing to pouring: reuse the skill</h3>
         <p>The tasks in this video further show how RPent reuses existing capabilities. A skill originally used to pick up and place a container can be recombined for pouring steel beads; grasping can also be combined with dual-arm coordination and sustained contact for plate wiping and storage.</p>
         <p>After exploration succeeds, RPent stores the verified task logic as a Task Card. Flash Mode can replay the card on a later run, making only the visual adjustments that the current scene requires instead of calling the large model for every step, which reduces execution latency.</p>
         <p>These videos are not about teaching a robot a few more motions. They ask whether a robot entering the open world can reorganize existing skills as new tasks, objects, and obstacles appear outside the training distribution. RPent moves the robot from executing a fixed instruction toward understanding the goal, calling the right capability, adapting to change, and turning one success into reusable experience.</p>
       </div>
     </article>
   </div>
