# RPent LIBERO global memory

This directory is the versioned Global Memory used by the RPent LIBERO agent.
It contains general operating lessons, reusable calibration, and reusable
manipulation patterns derived from reference tasks.

`MEMORY.md` is the single entry point. The agent scans that index first, then
opens the most relevant leaf memories before acting. Manipulation files are
indexed by reusable operation pattern; exact task phrases appear only as retrieval
aliases inside the leaf memory. Geometry, failure modes, and control patterns
remain available without using benchmark task identifiers as the primary
organization.

## Content rules

- Preserve verified action order, parameter ranges, prompts, and recovery
  lessons unless a new experiment directly disproves them.
- Describe geometry from the current scene or from reusable relative offsets;
  do not expose benchmark-definition provenance in the runtime narrative.
- Keep paths relative to the RPent repository root.
- Use ordinary `recipe` naming. Do not encode evaluation-layout provenance or
  agent implementation names in operational advice.
- Every cross-reference must resolve to a real file in this directory.

## Updating

1. Edit the relevant leaf memory manually.
2. Preserve the useful technique and change only the wording that has become
   stale or environment-specific.
3. Update `MEMORY.md` when a file is added, renamed, or removed.
4. Run the memory checks for forbidden provenance terms, absolute paths, and
   unresolved cross-references.
5. Review the diff before committing the snapshot.

The canonical runtime path is `resources/libero/memory/`, relative to the
RPent repository root.
