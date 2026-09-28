YAM
===

The video below shows YAM reading labels, adapting its motion, and revealing an occluded target.

.. raw:: html

   <div style="display: flex; flex-direction: column; gap: 2rem; margin: 1.5rem 0;">
     <article style="border: 1px solid var(--color-foreground-border); border-radius: 0.6rem; overflow: hidden;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4" type="video/mp4">
         Your browser does not support embedded video. <a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4">Download the video</a>.
       </video>
       <div style="padding: 1rem 1.25rem;">
         <h3 style="font-size: 1.15rem;">Read the labels, place it right</h3>
         <p>The robot reads the brand labels on a bottle and a bag, then places each drink in the matching bag. After grasping a bottle, it performs a small test lift to verify the grasp; when the original path is unavailable, it changes its wrist posture and continues. When a spoon is hidden under bowls, it moves the bowls away before attempting the grasp.</p>
         <p>This is an observation, decision, action, and correction loop rather than a pre-written sequence.</p>
       </div>
     </article>
   </div>
