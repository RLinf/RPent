YAM
===

下面的视频展示 YAM 读取标签、调整动作并处理遮挡目标的过程。

.. raw:: html

   <div style="display: flex; flex-direction: column; gap: 2rem; margin: 1.5rem 0;">
     <article style="border: 1px solid var(--color-foreground-border); border-radius: 0.6rem; overflow: hidden;">
       <video controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02-poster.jpg" style="width: 100%; height: auto; display: block;">
         <source src="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4" type="video/mp4">
         您的浏览器不支持嵌入式视频。<a href="https://raw.githubusercontent.com/RLinf/misc/c6f629129ac1a889d764dcd93defc213a4c31ae5/rpent/demo/real-world-demo02.mp4">下载视频</a>。
       </video>
       <div style="padding: 1rem 1.25rem;">
         <h3 style="font-size: 1.15rem;">读懂标签，把饮料放对位置</h3>
         <p>机器人需要读取瓶身和袋子上的品牌标签，将不同饮料放入对应的袋子。抓起瓶子后，它会先进行小幅试抬，确认夹持稳定；当原有运动路径不可行时，再调整腕部姿态继续执行。面对被碗遮挡的勺子，机器人也不会直接尝试抓取，而是先移开两个碗，让目标重新变得可见、可达。</p>
         <p>这里展示的已经不再是一条预先写死的动作序列，而是一个完整的“观察—判断—执行—纠错”闭环。</p>
       </div>
     </article>
   </div>
