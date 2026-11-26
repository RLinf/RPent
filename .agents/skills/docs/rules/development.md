# Concepts and Development Pages

Pages under `development/` explain how RPent's layers cooperate (`architecture.rst`, `interfaces.rst`, `memory.rst`) or show contributors how to extend RPent (`add_robot.rst`, `add_planner.rst`, `add_primitive.rst`, `add_action_model.rst`).

## Concepts (`architecture.rst`, `interfaces.rst`, `memory.rst`)

- **Lead with the engineering problem.** Open with 1–2 sentences stating what the subsystem is and what problem it solves, before naming any internal class.
- **Explain how components cooperate, not a class inventory.** Follow the reader's questions from normal execution to extensions and lifecycle ownership rather than walking file by file through `rpent/`.
- **Build a continuous article.** Open each section by connecting what the reader just learned to the next question. Introduce extensions when they become relevant (for example, explain locally spawned services before borrowed external endpoints).
- **The outline test.** Reading only the introduction and the opening paragraph of each section should form a clear, logical outline.
- **Frame examples.** Before a snippet, state the result it shows; afterwards, highlight key inputs, return values, and lifecycle ownership (who creates and cleans up resources). Keep illustrative snippets clearly distinct from executable code.

## Extension guides (`add_robot.rst`, `add_planner.rst`, `add_primitive.rst`, `add_action_model.rst`)

Contributors write code here, so public RPent names belong in the prose—limited to the symbols the contributor creates, calls, implements, or registers (`RobotSpec`, `RunConfig`, `PromptBundle`, `Toolkit`, `Planner`, `BaseEnvClient`, `BaseVLAClient`, `try_spawn_server`, etc.).

1. **What you add and what works afterwards**, in one paragraph.
2. **Prerequisites** (working checkout, upstream packages, assets/checkpoints).
3. **Extension points and files touched**, in the order the contributor edits them.
4. **One minimal end-to-end example** that closes the loop: implementation, registration, CLI/config selection, and runnable command.
5. **Walkthrough of public RPent calls** in caller order: inputs, return value/side effects, and setup/cleanup ownership.
6. **How to test it**: offline unit test placement (`tests/unit_tests/...`) and real-component/e2e command.

Verify every symbol against the codebase—`check_doc_symbols.py` checks code-font identifiers on all `development/` pages.

## Checklist

- [ ] Lead explains the problem and scope before introducing internal identifiers.
- [ ] Every public signature, default, argument type, and lifecycle rule matches current code.
- [ ] Every section opening connects naturally to the previous section (outline test passes).
- [ ] Extension guides close the loop from implementation and registration to runnable test command.
