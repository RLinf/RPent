# Index Pages, Navigation, and Redirects

An index page routes readers to the right guide; it does not duplicate the content of its child pages. This file mirrors the navigation structure in [docs/STYLE_GUIDE.md](../../../../docs/STYLE_GUIDE.md#navigation-and-page-ownership).

## Sidebar groups and directories

Keep the same eight sidebar groups, order, and relative page paths in both `docs/source-en/` and `docs/source-zh/`:

| Sidebar group (EN / ZH) | Directory under `rst_source/` | Owns |
|---|---|---|
| Get Started / 开始使用 | `get_started/` | RPent introduction (`overview.rst`) and Quick Start (`quickstart.rst`) |
| Leaderboard / 排行榜 | `leaderboard/` | Published benchmark performance and time/token pages |
| Real-World Demos / 真实世界演示 | `real_world_demos/` | Direct robot demo pages linking to their deployment guides |
| Guides / 使用指南 | `guides/` | Memory, primitives, planners, CLI reference, Dashboard, Flash Mode, flywheel, remote/parallel deployment |
| Simulators / 仿真环境 | `simulators/` | LIBERO, RoboCasa365, RoboTwin, RoboDojo, EFE |
| Real-World Robots / 真实机器人 | `real_world_robots/` | Franka, Dual Franka, YAM, SO-101 (preserve coming-soon pages) |
| Concepts & Development / 原理与开发 | `development/` | Architecture, interfaces, memory design, RGB-D toolkit, extension guides |
| Resources / 项目资源 | `resources/` | Harness VLA paper, contributing, release notes |

## Index page shape

- **Home page (`docs/source-{en,zh}/index.rst`)**: centered logo and heading, short intro, task-oriented cards, and all eight `:hidden:` toctrees driving the sidebar.
- **Directory routing pages (`simulators/index.rst`, `real_world_robots/index.rst`, `leaderboard/index.rst`)**: `:orphan:` routing pages with one brief intro + `sphinx-design` cards or a `.. list-table::` with concise descriptions, linking to pages already owned by the home-page toctrees without adding another toctree. Never use bare bullet link lists as the visible directory.
- Keep `toctrees` identical between `docs/source-en/` and `docs/source-zh/` (`docs/check.py` verifies `docs/parity` on toctree targets).

## Moving, renaming, and restructuring pages

- Keep existing page URLs and named anchors (`.. _anchor:`) wherever practical.
- Whenever a page moves or is renamed, update all `:doc:` / `:ref:` links and add the old-to-new redirect mapping in **both** `docs/source-en/conf.py` and `docs/source-zh/conf.py` (`redirects = { ... }`).
- Preserve published leaderboard data, reproduction conditions, and coming-soon placeholders unless explicitly asked to change them.
