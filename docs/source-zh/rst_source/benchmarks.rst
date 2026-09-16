:html_theme.sidebar_secondary.remove:

.. _benchmark-results:
.. _benchmark-leaderboard:
.. _leaderboard:

RPent 排行榜
=================

.. raw:: html

   <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/RLinf/misc@07c25bdb9c036ecc437f8e83c8117a080e4a6d2e/rpent/benchmarks/docs.css">
   <script defer src="https://cdn.jsdelivr.net/gh/RLinf/misc@07c25bdb9c036ecc437f8e83c8117a080e4a6d2e/rpent/benchmarks/table-sort.js"></script>
   <script defer src="https://cdn.jsdelivr.net/gh/RLinf/misc@07c25bdb9c036ecc437f8e83c8117a080e4a6d2e/rpent/benchmarks/leaderboard.js"></script>
   <script defer src="https://cdn.jsdelivr.net/gh/RLinf/misc@07c25bdb9c036ecc437f8e83c8117a080e4a6d2e/rpent/benchmarks/embed.js"></script>
   <div id="rpent-interactive-leaderboard" data-language="zh"
        data-results-url="https://raw.githubusercontent.com/RLinf/misc/07c25bdb9c036ecc437f8e83c8117a080e4a6d2e/rpent/benchmarks/results.json">
     <div class="rpent-static-leaderboard">
       <h2>LIBERO-PRO</h2><table><thead><tr><th>方法 / 模型</th><th>成功率</th></tr></thead><tbody><tr><td>Codex / GPT-6 Astra / low / reasoning</td><td>92.63%</td></tr><tr><td>Claude Code / Opus 4.7</td><td>82.4%</td></tr><tr><td>RPent Flash Mode</td><td>72.63%</td></tr><tr><td>Codex / GPT-5.5 / xhigh / reasoning</td><td>72.1%</td></tr><tr><td>π_RLinf</td><td>50.0%</td></tr><tr><td>π0.5</td><td>11.0%</td></tr><tr><td>AtomVLA</td><td>6.3%</td></tr><tr><td>X-VLA</td><td>3.8%</td></tr><tr><td>MolmoAct</td><td>1.5%</td></tr><tr><td>π0</td><td>0.3%</td></tr></tbody></table>
       <h2>LIBERO</h2><table><thead><tr><th>方法 / 模型</th><th>成功率</th></tr></thead><tbody><tr><td>AtomVLA</td><td>97.0%</td></tr><tr><td>Claude Code / Opus-4.7 / max / reasoning</td><td>96.0%</td></tr><tr><td>π_RLinf</td><td>95.3%</td></tr><tr><td>π0</td><td>94.2%</td></tr><tr><td>NORA</td><td>79.5%</td></tr><tr><td>OpenVLA</td><td>76.5%</td></tr></tbody></table>
       <h2>RoboCasa365 · Target50</h2><table><thead><tr><th>方法 / 模型</th><th>成功率</th></tr></thead><tbody><tr><td>Codex / GPT-6 Astra / low / reasoning</td><td>59.20%</td></tr><tr><td>Xiaomi-Robotics-1</td><td>57.4%</td></tr><tr><td>Codex / GPT-5.5 / xhigh / reasoning</td><td>57.1%</td></tr><tr><td>Claude Code / Opus-4.7 / max / reasoning</td><td>48.6%</td></tr><tr><td>WorldDreamer</td><td>35.3%</td></tr><tr><td>RLDX-1</td><td>30.0%</td></tr><tr><td>π0.5</td><td>16.9%</td></tr><tr><td>π0</td><td>14.8%</td></tr></tbody></table>
       <h2>RoboTwin</h2><table><thead><tr><th>方法 / 模型</th><th>成功率</th></tr></thead><tbody><tr><td>Codex / GPT-5.5 / xhigh / reasoning</td><td>62.4%</td></tr><tr><td>Claude Code / Opus-4.7 / max / reasoning</td><td>58.4%</td></tr><tr><td>LingBot-VLA</td><td>50.4%</td></tr><tr><td>π0.5</td><td>47.9%</td></tr><tr><td>GR00T-N1.7</td><td>20.7%</td></tr><tr><td>StarVLA</td><td>10.6%</td></tr></tbody></table>
       <p>GPT-6 Astra: 92.63% (741/800). RPent Flash Mode / Molmo2-8B: 72.63% (581/800).</p>
       <p>ASPIRE：六项常规评测；Long 为 LIBERO-90 技能库零样本迁移，未报告八套件 Overall。Flash Mode 使用 Molmo2-8B 进行视觉定位，不用于规划。</p><h3>ASPIRE</h3><table><thead><tr><th>分项</th><th>成功率</th><th>评测</th></tr></thead><tbody><tr><td>Object Swap</td><td>98.0%</td><td>保留测试 seed 1-50</td></tr><tr><td>Object Task</td><td>95.0%</td><td>保留测试 seed 1-50</td></tr><tr><td>Goal Swap</td><td>81.0%</td><td>保留测试 seed 1-50</td></tr><tr><td>Spatial Task</td><td>60.0%</td><td>保留测试 seed 1-50</td></tr><tr><td>Spatial Swap</td><td>51.0%</td><td>保留测试 seed 1-50</td></tr><tr><td>Goal Task</td><td>45.0%</td><td>保留测试 seed 1-50</td></tr><tr><td>Long Task</td><td>38.3%</td><td>Long 零样本迁移（LIBERO-90 技能库）</td></tr><tr><td>Long Swap</td><td>22.6%</td><td>Long 零样本迁移（LIBERO-90 技能库）</td></tr></tbody></table><h3>RPent Flash Mode · Task + Swap 汇总</h3><p>采用公开图中明确标注的合计次数；未按柱高估算各 Task/Swap 次数，不能将合计作为单个分项。</p><table><thead><tr><th>套件</th><th>成功率</th><th>成功 / 总回合</th></tr></thead><tbody><tr><td>Object</td><td>89.5%</td><td>179/200</td></tr><tr><td>Spatial</td><td>74.5%</td><td>149/200</td></tr><tr><td>Goal</td><td>69.5%</td><td>139/200</td></tr><tr><td>Long</td><td>57.0%</td><td>114/200</td></tr></tbody></table>
     </div>
   </div>
