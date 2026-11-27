# Simulator and Real-World Robot Guides

A simulator or real-world robot guide takes a reader from prerequisites to running a concrete task with RPent, inspecting result artifacts, and reproducing published results. Pages under `simulators/` and `real_world_robots/` follow this file and [base.md](base.md). Templates live in [../templates/simulator.rst](../templates/simulator.rst) and [../templates/simulator.zh.rst](../templates/simulator.zh.rst).

## Simulator page anatomy

1. **Title and optional figure.** H1 title, optional credited overview figure (`:align: center`, `:width: 90%`, descriptive `:alt:`, source credit in caption; describe upstream figures as upstream, not RPent results; keep section anchors outside the figure directive), followed by a 1–2 sentence lead stating what tasks the reader runs with RPent.
2. **Overview (`Overview` / `概览`, H2, required).** Four `sphinx-design` cards (`.. grid:: 2 4 4 4` with `:gutter: 2`, left-aligned text) in fixed order: **Action Models / 动作模型**, **Planners / 规划器**, **Tasks / 任务**, **Hardware / 硬件**.
3. **Tasks (`Tasks` / `任务`, H3 under Overview, required).** A `.. list-table::` summarizing task suites or categories (keep detailed task/seed matrices in Experiment Reproduction).
4. **Observation and Action (`Observation and Action` / `观测与动作`, H3 under Overview, required).** A `.. list-table::` with rows **Observation / 观测**, **Action / 动作**, **Reward / success / 奖励与成功判定**, and **Task prompt / 任务指令**. Distinguish planner tools from model inputs, and state the native success signal (`terminated` or verifier reward)—never treat planner `finish` as task success.
5. **Installation and Resources (`Installation and Resources` / `安装与资源准备`, or `Installation` / `安装`, H2, required).** System/Python requirements, exact `./rpent-install.sh install robot-<name>` command (and any extra component flags if needed), asset and checkpoint downloads.
6. **Run a Task (`Run a Task` / `运行一个任务`, H2, required).** Model/planner config and one complete, copyable command sequence to run a single task, followed by the 3–5 flags the user actually changes.
7. **View Results (`View Results` / `查看结果`, H2, required).** Output directory files and how to read the native success criterion.
8. **Conditional sections (H2, after the first-run path).** Include only when supported or published; do not add empty placeholders:
   - `Task Memory` / `任务记忆` (or combined `Task Memory and Exploration Mode` / `任务记忆与探索模式`)
   - `Exploration Mode` / `探索模式`
   - `Experiment Reproduction` / `实验复现` (code version, resource revisions, task/seed matrix, model settings, limits, retry policy, aggregation; an environment-specific benchmark qualifier such as `Experiment Reproduction (Target50)` belongs only to that environment)
   - Environment-specific options and common failures (symptom first, then fix).

Link to shared guides (`guides/configure_planner.rst`, `guides/memory.rst`, `guides/dashboard.rst`, `guides/cli.rst`) instead of repeating generic planner or memory instructions.

## Real-world robot pages

- Use an introductory robot photo with `:alt:`, caption, and `:figclass: rpent-robot-figure`; keep the source URL in RST and add credit when required.
- Cover hardware/controller requirements, calibration, configuration, motion checks, task execution, result inspection, and stopping.
- State operator responsibilities and physical safety constraints explicitly.
- Preserve coming-soon pages and clearly state which workflows are ready vs. pending.

## Hardware claims

State software and GPU requirements from verified setups. Distinguish measured GPU models/VRAM from minimum estimates, and add a hardware backend subsection only when its installation, device flags, rendering, run command, and limits are verified and supported.

## Checklist

- [ ] A reader can follow Installation → Run a Task → View Results top to bottom without opening source files.
- [ ] Overview uses the four cards in fixed order (`Action Models`, `Planners`, `Tasks`, `Hardware`) plus H3 `Tasks` and `Observation and Action` tables.
- [ ] Native success (`terminated` / verifier reward) is stated clearly and never conflated with planner `finish`.
- [ ] No internal `rpent/`, `robots/`, `components/`, or `toolkits/` code names or tours of implementation files.
- [ ] Published benchmark numbers keep all experimental conditions, denominators, and task/seed scope intact.
