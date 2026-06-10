"""RLinf backend: LIBERO environment + OpenPI VLA primitives.

This subpackage depends on an external RLinf checkout.  Use
``physicalagent.backends.add_external_rlinf_to_path`` before importing
if RLinf is not already on ``sys.path``.
"""

from physicalagent.backends.rlinf.primitives import (  # noqa: F401
    CHECKPOINT_PATH,
    LiberoPrimitiveDriver,
    PrimitiveResult,
    build_driver,
    build_env_cfg,
    build_model_cfg,
)