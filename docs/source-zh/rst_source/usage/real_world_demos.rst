真实世界演示
============

下面的剪辑展示 RPent 如何在真机上把观察、判断、执行和纠错连成闭环。
这些视频用于展示系统行为，不是评测成绩；安装配置和复现步骤请参考对应的真实机器人教程。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 1.5rem 0;">
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4" type="video/mp4">
         您的浏览器不支持嵌入式视频。<a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo01.mp4">下载视频</a>。
       </video>
       <h3>同一只盘子，新的放置目标</h3>
       <p><strong>目标：</strong>将干净的蓝色盘子放入纸箱。</p>
       <p>目标变化后，RPent 重新观察场景并调整放置位置，而不是照搬冻结 VLA 的原路径。</p>
     </div>
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4" type="video/mp4">
         您的浏览器不支持嵌入式视频。<a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4">下载视频</a>。
       </video>
       <h3>读瓶身，也读袋子</h3>
       <p><strong>目标：</strong>将 Pepsi 瓶放入标有 Pepsi 的袋子。</p>
       <p>RPent 读取标签，先试抬确认夹持稳定，并在路径不可行时调整腕部姿态。</p>
     </div>
     <div style="border: 1px solid var(--color-foreground-border); border-radius: 0.5rem; overflow: hidden; padding: 0.75rem;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4" type="video/mp4">
         您的浏览器不支持嵌入式视频。<a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo03.mp4">下载视频</a>。
       </video>
       <h3>从放置到倾倒</h3>
       <p><strong>目标：</strong>将容器中的钢珠倒入碗中。</p>
       <p>RPent 将已有的放置技能复用于新的倾倒任务，并围绕共同工作空间完成双臂协同。</p>
     </div>
   </div>
