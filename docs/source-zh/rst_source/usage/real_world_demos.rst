真实世界演示
============

先看效果：具身智能体的雏形
----------------------------

下面三段真机演示展示 RPent 如何在任务改变、路径受阻和操作组合变化时重新观察场景并调整行动：将同一只蓝色盘子放入新的纸箱，读取瓶身和袋子标签并试抬确认抓取，再把已有的放置技能复用于钢珠倾倒。它们展示的不是固定动作的重复，而是从观察、判断、执行到纠错的闭环，以及将成功经验沉淀为可复用能力。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 1.5rem 0;">
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4" type="video/mp4">
         您的浏览器不支持嵌入式视频。<a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4">下载视频</a>。
       </video>
       <h3>同一只盘子，新的放置目标</h3>
       <p><strong>目标：</strong>将干净的蓝色盘子放入纸箱。</p>
       <p>同一只蓝色盘子换了新的目的地。RPent 重新观察场景，选择纸箱，检查放置结果，并在需要时纠正位置，而不是照搬冻结 VLA 的原路径。</p>
     </div>
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4" type="video/mp4">
         您的浏览器不支持嵌入式视频。<a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4">下载视频</a>。
       </video>
       <h3>读瓶身，也读袋子</h3>
       <p><strong>目标：</strong>将 Pepsi 瓶放入标有 Pepsi 的袋子。</p>
       <p>RPent 读取瓶身和袋子上的标签，先小幅试抬确认夹持稳定；原有路径被挡住时，再调整腕部姿态继续执行。</p>
     </div>
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4" type="video/mp4">
         您的浏览器不支持嵌入式视频。<a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4">下载视频</a>。
       </video>
       <h3>从放置到倾倒</h3>
       <p><strong>目标：</strong>将容器中的钢珠倒入碗中。</p>
       <p>RPent 将已有的放置技能重新组合到倾倒任务中，并围绕共同工作空间协调双臂，不必为每个新任务重新训练一套策略。</p>
     </div>
   </div>

从一次成功到可复用的技能
--------------------------

探索成功后，RPent 可以把经过验证的任务逻辑沉淀为任务卡（Task Card）。再次执行时，Flash Mode 根据当前视觉状态重新定位任务卡中的锚点，直接复用已经验证过的流程。它不需要让大模型为每一个细小动作重新决策，只在场景发生变化时请求调整，于是一次成功就能变成下一次可以复用的经验。

这些演示真正想追问的是：当训练分布之外的新物体、新目标和新障碍不断出现时，机器人能不能重新组织已有技能，而不是等待一条新的固定策略？RPent 让机器人从执行一条写死的指令，进一步走向理解目标、调用能力、应对变化，并把一次成功沉淀为下一次可以复用的经验。
