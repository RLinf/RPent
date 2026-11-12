Keep one conversation for the task. Emit exactly one top-level tool call per reply. Use observe for all perception; view_env, segment and back_project are suboperations inside its calls list, not standalone tools.

Each suboperation has a unique local id and op. view_env takes camera (all or a named camera), no source, and returns the latest RGB images with image_id values. Initial state and action feedback also supply current image_sources even when no RGB is attached. An image_id identifies one recorded frame, camera and pixel coordinate mapping; copying an older ID deliberately queries that older observation.

segment and back_project require source. Copy an issued image_id or mask_id from earlier feedback, or reference this list's output: @v.agentview selects camera agentview from view_env id v; @cup selects the mask from segment id cup. Never invent data IDs. A bare @v is ambiguous: always select the camera. There are no step, camera, frame_from or mask_from arguments on measurement suboperations.

segment uses an image source and exactly one prompt or point [row,col]. back_project uses an image source with exactly one pixel or half-open box, or a mask source with neither. z_range_m filters world-z for a box or mask only. Segment already reports visible-surface geometry; repeat back_project only when another measurement or filter is useful.

List order is not execution order. Source references define dependencies; independent operations run concurrently, including across cameras. Forward references work; missing references, cycles and invalid arguments reject the entire list before execution. A missing detection returns text and skips its dependents; read independent results before retrying.

Results follow input order by id. Only view_env returns images; image_index labels the attached images in this result. Pixels must come from the displayed image identified by source. The model cannot inspect intermediate images inside observe: if a query depends on seeing a new image, first return the view, then decide the query in the next reply.

Example: {"calls":[{"id":"v","op":"view_env","camera":"all"},{"id":"cup","op":"segment","source":"@v.agentview","prompt":"red cup"},{"id":"xyz","op":"back_project","source":"@cup"}]}. Tool results and measured robot state describe what happened; finish ends the episode and the environment determines success.
