# README Highlights

Update `README.md` and `README.zh-CN.md` together in the same PR when adding a major user-visible feature (a new simulator, real-world robot, action model, planner, perception toolkit, or workflow such as Flash Mode or Data Flywheel). Bug fixes, internal refactors, and tooling changes do not need a README entry.

- Add a dated bullet at the top of **What's NEW!** (`最新动态`) in the form:
  - EN: `- [YYYY/MM] 🔥 RPent supports <feature>. See the [<guide>](https://rpent.readthedocs.io/en/latest/rst_source/<path>.html).`
  - ZH: `- [YYYY/MM] 🔥 RPent 支持 <功能>。详见 [<指南>](https://rpent.readthedocs.io/zh-cn/latest/rst_source/<path>.html)。`
- Write the Chinese bullet natively ([chinese.md](chinese.md)) rather than translating word for word.
- Update the matching column in **Feature Matrix** (`功能矩阵`) — Agentic Planner, Action Primitive (VLA / WAM), Simulator, or Real World — with the rendered Read the Docs link (`en/latest` in `README.md`, `zh-cn/latest` in `README.zh-CN.md`).
- State what the user can do in plain terms, with no sales language and no internal module paths. Keep `README.md` and `README.zh-CN.md` links, numbers, and status markers (`✅`) aligned.
