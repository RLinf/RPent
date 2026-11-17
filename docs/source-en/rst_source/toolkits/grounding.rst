Object Grounding
================

The grounding toolkit adds open-vocabulary object labels to a built map. A
perception service runs Grounding DINO and SAM 2 on the GPU, and the offline
grounder lifts the resulting masks onto the map's depth, associates them into
instances, and writes the annotations back into the map directory. The map
itself comes from :doc:`mapping`.

Prerequisites
-------------

Grounding needs a CUDA GPU and the ``grounding-sam2`` extra:

.. code-block:: bash

   uv pip install -e '.[grounding-sam2]'

It needs two checkpoints. Grounding DINO must use the original release weights
— the transformers-format ``model.safetensors`` on Hugging Face does not load.

.. code-block:: bash

   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
       -P /path/to/checkpoints
   hf download facebook/sam2.1-hiera-large sam2.1_hiera_large.pt \
       --local-dir /path/to/checkpoints

   export GDINO_CHECKPOINT_PATH=/path/to/checkpoints/groundingdino_swint_ogc.pth
   export SAM2_CHECKPOINT_PATH=/path/to/checkpoints/sam2.1_hiera_large.pt

Keep the checkpoint filenames unchanged: the perception server selects the
SAM 2 configuration from the filename. Its first start also downloads the
``bert-base-uncased`` tokenizer, so that start needs network access.

Start the Perception Service
----------------------------

``toolkits.grounding.server`` wraps Grounding DINO (open-vocabulary detection)
and SAM 2 (mask refinement) behind one ``gdino_sam2.detect_segment`` call.

.. code-block:: bash

   python -m toolkits.grounding.server \
       --transport http --host 127.0.0.1 --port 8115

Service Client
--------------

``toolkits.grounding.client`` works against a local or remote service:

.. list-table::
   :header-rows: 1

   * - Call
     - Inputs
     - Returns
   * - ``GdinoSam2Client.detect_segment(image, labels=...)``
     - An RGB frame and at most 16 labels.
     - Scored boxes and one instance mask per detection.

.. _workspace-mapping-grounding:

Ground Objects
--------------

Grounding runs on its own, after any build. ``toolkits.grounding.ground_map``
reads a built map, picks evenly spaced keyframes (or the indices passed with
``--keyframes``), and writes the annotations back into the map:

.. code-block:: bash

   python -m toolkits.grounding.ground_map \
       --map-path /path/to/maps/workspace \
       --labels chair desk monitor \
       --perception-endpoint http://127.0.0.1:8115 \
       --max-keyframes 12

The map path is the directory a build wrote. On a reconstructed map, each mask
is mapped through the server's resize and center crop onto the depth grid,
unprojected with the frame's intrinsics and camera pose, and associated with
same-label observations from other keyframes. A map without depth keeps 2D
annotations only, and ``--allow-2d`` is required to write them.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - What it does
   * - ``--max-keyframes``
     - 12
     - How many evenly spaced keyframes to ground.
   * - ``--keyframes``
     - evenly spaced
     - Explicit frame indices, comma or space separated.
   * - ``--min-score``
     - 0.35
     - Lowest detection score to keep.
   * - ``--min-mask-pixels``
     - 32
     - Smallest mask, in source pixels.
   * - ``--min-confidence``
     - 1.0
     - Lowest depth confidence for a lifted pixel.
   * - ``--merge-factor``
     - 0.75
     - Same-label centroids within this multiple of their typical radius
       become one instance.
   * - ``--output-dir``
     - the map
     - Write the artifacts elsewhere and leave the map untouched.
   * - ``--allow-2d``
     - off
     - Keep 2D annotations on a map without depth.

The command writes ``objects.json``, ``annotations/`` (per-frame mask
overlays), ``semantic_points.npz`` and ``semantic_points.ply`` (the labelled
cloud), and ``evaluation.json``. Grounding results go back through
``save_annotations``, so the client does not need filesystem access to the
machine that built the map.

``evaluation.json`` reports per-label counts and the median distance between
the nearest same-label observations in different frames. That distance
measures cross-frame consistency; precision and recall need independent
ground-truth labels and are not reported.

Limits
------

Two limits follow from the model geometry and the detector. Masks that lie
entirely outside the model's center crop have no depth and stay 2D detections
— a portrait frame loses rows above and below the square crop even though the
archived frame keeps them. Grounding DINO can also return several overlapping
masks for one object; association merges observations only when their 3D
centroids fall within the radius threshold, so a sparse keyframe set can leave
more than one entry for the same object. Label matching is case-insensitive
and slugs must stay distinct, so ``Chair`` and ``chair`` cannot both be
requested.

Inspect the Result
------------------

``python -m toolkits.mapping.view`` renders per-label counts, object tables,
and the detection overlays into a report directory; see
:ref:`View a Built Map <workspace-mapping-view>`. The labelled cloud also opens
in external viewers through ``semantic_points.ply``.
