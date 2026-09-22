Real-World Demos
================

These edited clips show how RPent closes the loop between observation,
decision, action, and correction on real robots. The videos are showcase
clips rather than benchmark results; use the real-robot guides for setup and
reproduction details.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 1.5rem 0;">
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4">Download the video</a>.
       </video>
       <h3>Same plate, a new destination</h3>
       <p><strong>Goal:</strong> Move the clean blue plate into the cardboard box.</p>
       <p>When the destination changes, RPent re-observes the scene and adjusts the placement instead of replaying the frozen VLA route.</p>
     </div>
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4">Download the video</a>.
       </video>
       <h3>Read the bottle. Read the bag.</h3>
       <p><strong>Goal:</strong> Sort a Pepsi bottle into the bag labelled Pepsi.</p>
       <p>RPent reads labels, verifies the grasp with a test lift, and adjusts the wrist when the original path is unavailable.</p>
     </div>
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4">Download the video</a>.
       </video>
       <h3>From placing to pouring</h3>
       <p><strong>Goal:</strong> Pour steel beads from a container into a bowl.</p>
       <p>RPent reuses a learned placing skill for a new pouring task and coordinates both arms around the shared workspace.</p>
     </div>
   </div>
