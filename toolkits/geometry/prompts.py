# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environment-independent parent instructions and observed-part descriptions."""

GEOMETRY_INSTRUCTIONS = """Use GeometryTool for complex geometry perception through geometry.
Request observed surface positions or fitted part geometry.
Describe the target's appearance, visible part and spatial relation. Supply a
camera name available from the observation provider, a recorded step (-1 for
latest), and either a surface measurement or required_geometry. Do not guess
pixels or XYZ. Calibration and coordinate frames come from the observation
provider, not from this tool call.

surface_point is one visible material point. visible_region gives the
coordinate-wise median and bounds of visible segmented samples, not a fitted
center. For fitted quantities read geometry[field], its definition, units,
world_frame_id, source_step, evidence_status, source_refs, fit_accepted, warnings
and quality. A returned estimate with fit_accepted=false has not passed quality
checks: read its warnings before deciding whether another observation is needed.
Do not treat status=warning as accepted geometry or a calibrated error bound.
radius means rim radius. Full height and body_center require observed rim and
bottom boundaries; inconsistent estimates carry warnings. Fields without a
computable estimate stay unknown. Fit residuals and segmentation scores are
diagnostics, not calibrated position errors.

GeometryTool has no motion tools. On partial, need_observation or error,
inspect the missing fields and reason before choosing a new observation or
recovery action. Avoid repeating unchanged requests. Preserve target identity
and part definitions across views; compare matching coordinate frames and
measure again after target motion.
"""

PART_DESCRIPTIONS = {
    "rim": "the visible circular rim boundary of",
    "body": "the visible curved outer body, excluding handle, rim and interior, of",
    "bottom": "the visible circular bottom face or its boundary of",
    "plane": "the visible planar surface of",
    "inner_rim": "the visible INNER upper opening edge, excluding outer lip, handle and interior floor, of",
    "plate_surface": "the entire central flat circular supporting surface INSIDE the raised rim of",
    "lid": "the entire sealed circular top lid surface, excluding its raised rolled outer lip, of",
    "rectangular_top": "the entire main rectangular TOP face up to its four physical edges, excluding side faces, doors and printed-label-only crops, of",
    "bottle_segment": "a visible material point on the main cylindrical barrel, excluding shoulder, neck, cap and mouth, of",
}
