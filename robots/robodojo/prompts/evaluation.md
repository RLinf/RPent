## RoboDojo initial observation

Call `view_env_state` first to read the instruction, observe the scene and check the native action budget before planning.

Before the first robot action, read robots/robodojo/guides/GUIDE_RPENT.md completely. Inspect view_env_state and head/wrist images, read the native instruction, and note both arms' initial end-effector positions and orientations. Bind arm roles from current observations.

## RoboDojo evaluation references and audit

The reference is the same exact task variant explored at seed 0, layout 0. Keep standard and Random task-specific pairs and task-family notes separate; do not substitute one variant's recipe for the other's. Revalidate SAM3 identity and contact geometry, and choose arms for current reach unless the native instruction requires an arm. Historical SAM3 descriptions and policy chunk counts are evidence; adapt chunk length to the next observation gate.

Write {{output_dir}}/{{recipe_tag}}.json with native instruction, task, seed, layout, native_success, memory_files_read, missing references and material deviations from the reference strategy. Then call finish(status, summary). Stop motion on native success. Do not request hidden reward details.
