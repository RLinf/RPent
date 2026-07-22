Memory Management
=================

The LIBERO agent's global memory lives under ``resources/libero/memory/`` (a
``MEMORY.md`` index plus leaf notes). It is a reviewed, read-only knowledge
base, read at the start of each run.

Hosting
-------

``resources/`` is not vendored in git. It is hosted on the HuggingFace dataset
``RLinf/RPent-memory`` (laid out per environment, e.g. ``libero/memory/`` and
``libero/results_*_pert/``). ``rpent.utils.resources.ensure_resources`` syncs the
env's subtree from the dataset on each run (incremental: only changed files are
downloaded), so the local copy stays up to date. The dataset is public, so a
fresh clone downloads it without a token. Set ``HF_HUB_OFFLINE=1`` to skip the
sync and use the local copy only. Memory is optional: if an env has none on the
dataset, or the sync fails, the run continues with whatever is on disk.

Updating the memory
-------------------

Publishing memory is a controlled step carried out by maintainers with write
access to the ``RLinf`` organisation; the repository ships no self-serve upload
path. To contribute a better memory entry, open an issue with the proposed
content and a maintainer will review and publish it.

Reads need no credentials; writes require ``RLinf`` write access.

