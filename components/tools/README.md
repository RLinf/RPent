# Tools

Installable tool components live in `components/tools/<name>/`. Each one has its
own `component.toml` and owns the implementation, the input contract, the
internal primitives, and the tools it exposes to the planner. Two shapes exist:

- a process tool (for example `flywheel/`) with a `cli.py` that runs outside a
  robot or model environment;
- a tool capability (for example `rgbd/`) that a robot slice registers, so it
  runs in the RPent process family beside the planner.

A capability may depend on core contracts and on the model components' client
modules, but must not import `robots.*`, `components/sim`, or another tool; the
robot-specific part belongs under `robots/<robot>/` as an observation adapter. A
robot slice installs a tool by naming it in its `requires`, or
`rpent-install install tool-<name>` puts it in an environment of its own.

`rgbd/` is the generic RGB-D perception capability: it back-projects a pixel to
a world position, segments an object, and measures target parts from recorded
camera views, over the `GeometryObservation` contract that each robot adapts.
Add another tool here only when it has an independent consumer; a capability one
robot needs stays in that slice.
