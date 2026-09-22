真实世界演示
============

下面三个视频分别展示任务改变、路径受阻和技能复用。每个视频下方的说明对应视频中的具体任务。

.. raw:: html

   <div style="display: flex; flex-direction: column; gap: 2rem; margin: 1.5rem 0;">
     <article style="border: 1px solid var(--color-foreground-border); border-radius: 0.6rem; overflow: hidden;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4" type="video/mp4">
         您的浏览器不支持嵌入式视频。<a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4">下载视频</a>。
       </video>
       <div style="padding: 1rem 1.25rem;">
         <h3 style="font-size: 1.15rem;">目标变了，盘子也换个去处</h3>
         <p>例如，当任务变化为“将干净餐具放入纸箱”时，面对同一只蓝色盘子，冻结 VLA 可能会沿用训练时学到的动作，将它放入原来的金属篮中；而在 RPent 中，智能体会先观察当前环境，再根据新的目标选择放置位置。执行过程中，如果视觉定位出现偏差，它还会检查结果、排除错误目标，并重新定位手中的盘子：任务变了，机器人也能随之改变行动。</p>
       </div>
     </article>
     <article style="border: 1px solid var(--color-foreground-border); border-radius: 0.6rem; overflow: hidden;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4" type="video/mp4">
         您的浏览器不支持嵌入式视频。<a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4">下载视频</a>。
       </video>
       <div style="padding: 1rem 1.25rem;">
         <h3 style="font-size: 1.15rem;">读懂标签，把饮料放对位置</h3>
         <p>另一个任务中，机器人需要读取瓶身和袋子上的品牌标签，将不同饮料放入对应的袋子。抓起瓶子后，它会先进行小幅试抬，确认夹持稳定；当原有运动路径不可行时，再调整腕部姿态继续执行。面对被碗遮挡的勺子，机器人也不会直接尝试抓取，而是先移开两个碗，让目标重新变得可见、可达。</p>
         <p>这里展示的已经不再是一条预先写死的动作序列，而是一个完整的“观察—判断—执行—纠错”闭环。</p>
       </div>
     </article>
     <article style="border: 1px solid var(--color-foreground-border); border-radius: 0.6rem; overflow: hidden;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4" type="video/mp4">
         您的浏览器不支持嵌入式视频。<a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4">下载视频</a>。
       </video>
       <div style="padding: 1rem 1.25rem;">
         <h3 style="font-size: 1.15rem;">从放置到倾倒：技能复用</h3>
         <p>这个视频中的任务进一步展示了 RPent 对已有能力的复用。机器人可以将原本用于“拿起并放置容器”的技能重新组合，用于倾倒钢珠；又把抓取动作与双臂协同、持续接触组合起来，完成持盘、擦拭和收纳。</p>
         <p>探索成功后，RPent 会把经过验证的任务逻辑沉淀为任务卡（Task Card）；再次执行时，RPent 可以启用 Flash Mode，通过任务卡直接复用这套流程，只根据当前视觉状态做必要调整，避免每一步都重新调用大模型，降低具身智能体的执行延时。</p>
         <p>这几段视频想展示的并不是“机器人又学会了几个动作”，而是一个更重要的问题：当机器人走向开放世界，新的任务、物体和障碍不断出现时，能否将已有技能重新组织起来，完成训练分布之外的任务？RPent 使机器人从执行一条固定指令，进一步走向理解目标、调用能力、应对变化，并把一次成功变成下一次可以复用的经验，这正是具身智能体的雏形。</p>
       </div>
     </article>
   </div>
