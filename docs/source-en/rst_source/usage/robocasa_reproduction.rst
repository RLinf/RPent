RoboCasa Full Reproduction
==========================

This page defines the fail-closed RoboCasa365 Harness VLA reproduction
protocol. It is intentionally separate from the single-task development flow
in :doc:`robocasa`.

.. warning::

   The currently supported local hybrid runtime is **preliminary**. It is
   useful for integration and smoke testing, but it is not a published,
   immutable reproduction runtime. A formal release must publish and freeze
   the exact runtime tree, Python environment, RLDX checkpoint, VLM metadata,
   RoboCasa365 revision, and robosuite revision. Results from an unversioned
   local tree must not be presented as paper reproduction results.

Frozen evaluation matrix
------------------------

The protocol contains exactly 340 held-out cells:

.. list-table::
   :header-rows: 1

   * - Split
     - Tasks
     - Evaluation seeds
     - Cells
   * - Atomic
     - 18
     - 1--10
     - 180
   * - Composite seen
     - 16
     - 1--5
     - 80
   * - Composite unseen
     - 16
     - 1--5
     - 80
   * - **Total**
     - **50**
     - As above
     - **340**

Seed 0 is never an evaluation cell. The memory pack contains an exact
successful seed-0 audit/command-trace pair for 43 tasks and an intentionally
empty directory for the seven composite-unseen protocol-whitelisted tasks:
``HeatKebabSandwich``, ``PanTransfer``, ``PortionHotDogs``,
``SeparateFreezerRack``, ``WaffleReheat``, ``WashFruitColander``, and
``WeighIngredients``. There is no Global Memory, ``Task.md``, experience
Markdown, cross-task memory, or substitute memory for an empty task. The two
seed-0 files are procedural priors only; current RGB-D observations and
environment success remain authoritative.
The schema-v3 audit JSON is metadata only and cannot embed a command object;
the paired JSONL file is the sole command authority. Packing removes legacy
``command_sequence`` fields and recomputes existing command counters from
JSONL.

Evaluation is strictly no-reset. Each held-out seed is one live episode, and
only these eight physical actions are accepted in canonical command traces:
``navigate_to``, ``move_base``, ``move_to``, ``move_delta``,
``rotate_pitch``, ``set_gripper``, ``release``, and ``vla_act``. Planner exit
does not define success; the simulator and artifact validator do.
Simulator success is a hard physical termination boundary: reset samples the
initial predicate, the first successful step latches success, and every later
step is rejected. Multi-step analytic and VLA helpers stop at that first
successful step.

Versioned navview patch
-----------------------

The runtime requires one base-mounted ``navview`` camera in
``robosuite/models/assets/bases/omron_mobile_base.xml``. RPent ships the exact
patch as ``robots/robocasa/patches/robosuite_navview.patch``. The patch was
audited against ``RLinf/robosuite`` revision
``85abee228d1c43ab1939bce33028099945d453b4`` and adds exactly:

.. code-block:: xml

   <camera name="navview" mode="fixed" pos="0.2 0 1.6"
           xyaxes="0 -1 0 0.643 0 0.766" fovy="75"/>

Locate the packaged asset, check it against a clean checkout, and apply it
once:

.. code-block:: bash

   NAVVIEW_PATCH=$(python -c 'from importlib.resources import files; print(files("robots.robocasa").joinpath("patches/robosuite_navview.patch"))')
   ROBOSUITE_CHECKOUT=/path/to/runtime/external_dependencies/robocasa365/robosuite
   git -C "$ROBOSUITE_CHECKOUT" apply --check "$NAVVIEW_PATCH"
   git -C "$ROBOSUITE_CHECKOUT" apply "$NAVVIEW_PATCH"

If ``--check`` fails, do not force the patch. First use
``git apply --reverse --check`` to determine whether it is already applied;
otherwise the checkout differs from the audited input and must be resolved as
a versioning problem. The ``doctor`` command below parses the installed XML
and requires exactly one camera with the four attributes above, located as a
direct child of ``worldbody/body[@name='base']``.

Memory and planner authentication
---------------------------------

Build the task-isolated memory pack from the audited migration outputs, then
validate its identities, empty-task whitelist, exact filenames, and SHA-256
manifest:

.. code-block:: bash

   rpent-reproduce robocasa memory-pack \
     --migration-root /path/to/runtime/migration \
     --output /path/to/memory/harness_vla_v1
   rpent-reproduce robocasa memory-validate \
     --memory-dir /path/to/memory/harness_vla_v1

The formal runner defaults to the ``RLinf/RPent-memory`` dataset and downloads
``robocasa/harness_vla_v1/**`` automatically. It still requires an immutable
lowercase 40-character commit SHA, supplied with ``--memory-revision`` or
``RPENT_ROBOCASA_MEMORY_REVISION``. A staging dataset can be selected with
``--memory-repo-id`` or ``RPENT_ROBOCASA_MEMORY_REPO_ID``.

RPent materializes that revision into a private, non-symlink cache instead of
executing from Hugging Face's symlink-based blob cache.
``RPENT_ROBOCASA_MEMORY_CACHE`` may select the private cache root. A missing
download, invalid revision, or invalid pack is fatal; there is no implicit
fallback to server-local memory. ``--memory-dir`` remains an explicit
offline/development override and cannot be combined with Hugging Face options.

Planner authentication is explicit and has two mutually exclusive modes. The
frozen model and effort remain ``gpt-5.5`` and ``xhigh`` in both modes; the
authentication mode does not change the 340-cell matrix or scoring protocol.

``api-key`` is provider-neutral. It accepts any audited
Responses-compatible endpoint that supports the frozen model and effort. Both
the credential file and API base are required; there is no built-in provider
or endpoint default:

.. code-block:: bash

   API_AUTH_ARGS="--planner-auth-mode api-key \
   --api-key-file /run/secrets/responses-api-key \
   --base-url https://provider.example/v1"

``chatgpt-subscription`` uses a separately deployed, trusted broker on one
loopback HTTP listener. RPent receives only a one-line client capability for
that broker. The broker owns and refreshes the ChatGPT subscription OAuth
credential outside the evaluation cell:

.. code-block:: bash

   SUBSCRIPTION_AUTH_ARGS="--planner-auth-mode chatgpt-subscription \
   --broker-credential-file /run/secrets/broker-client-capability \
   --broker-base-url http://127.0.0.1:8765/v1 \
   --broker-health-url http://127.0.0.1:8765/health"

The loopback values above illustrate the RPent client contract; this
repository does not deploy or endorse a release-ready subscription broker.
The external deployment must be independently secured, audited, and frozen.
Before a cell starts, ``doctor`` requires the broker health response to attest
``provider_profile=chatgpt_subscription_broker``,
``auth_mode=chatgpt_broker``, ``credential_broker=true``,
``credential_broker_ready=true``,
``credential_broker_protocol=root_oauth_injection_v1``, ``model=gpt-5.5``,
and ``reasoning_effort=xhigh``. The request and health URLs must use the same
``127.0.0.1`` or ``::1`` listener.

In either mode, the RPent-visible credential must be a regular, non-symlink
file owned by the reproduction user with exact mode ``0600`` and exactly one
non-empty line. Never place its value in a command, committed configuration,
result artifact, or log. In subscription mode, OAuth tokens and
``auth.json`` must never enter a cell, result, command-line argument, or RPent
credential file; do not copy, mount, or symlink them into a cell. The broker
client capability is not an OAuth credential.

RPent freezes the same Codex provider retry settings in both authentication
modes: ``request_max_retries=12``, ``stream_max_retries=120``, and
``stream_idle_timeout_ms=330000``. Retries never extend the existing absolute
cell deadline: atomic cells remain bounded by 1800 seconds and composite cells
by 3600 seconds. The broker and network-egress path must still be continuously
supervised. A prolonged provider, broker, or egress outage is an infrastructure
incident, not benchmark evidence.

Preflight and checkpoint hashes
-------------------------------

Keep results private with exact mode ``0700``. The rollout root must be owned
by the reproduction user with exact mode ``0711``: isolated planner UIDs may
traverse a known path, but cannot list or write the root. Every ancestor of the
rollout root must also be traversable by other UIDs; ``doctor`` rejects a path
under a ``0700`` ancestor. Per-cell workdirs remain protected by a unique
planner group and the Landlock boundary. Results and rollout roots must not
overlap.

Run ``doctor`` before launching any cell. ``--verify-checkpoint`` hashes all
three RLDX checkpoint shards and compares them with the frozen expected
SHA-256 values. Doctor also checks the runtime scripts, Codex binary, VLM
metadata, memory manifest, secret-file policy, and navview XML:

.. code-block:: bash

   export RPENT_ROBOCASA_MEMORY_REVISION="$PUBLISHED_MEMORY_COMMIT"

   COMMON_ARGS="--runtime-root /path/to/runtime \
   --results-root /path/to/results \
   --rollout-root /path/to/rollouts \
   --planner-profile codex-gpt55-xhigh \
   --preliminary-local-runtime"

   # Select exactly one of API_AUTH_ARGS or SUBSCRIPTION_AUTH_ARGS.
   PLANNER_AUTH_ARGS="$API_AUTH_ARGS"

   rpent-reproduce robocasa doctor $COMMON_ARGS $PLANNER_AUTH_ARGS \
     --verify-checkpoint --verify-isolation

Treat any doctor problem as a configuration failure. Record its JSON output
with the immutable release manifest; do not bypass a hash or navview mismatch.
The ``--preliminary-local-runtime`` acknowledgement is required by ``doctor``
and ``run`` only for the current local hybrid snapshot. It explicitly labels
these executions as preliminary and must not be used to imply that the
snapshot is a formal release runtime.

The current development container cannot provide the mount propagation needed
by bubblewrap. The outer RLDX launcher therefore installs the authoritative
Landlock filesystem boundary, drops to a cell-specific high UID, and makes
only fixed mailboxes and private scratch directories writable before Codex
starts. Ordinary observation and prompt files are exposed read-only to the
cell group, but ``_deadline_commit.gate`` and ``_deadline_contract.json``
remain ``root:root`` with exact mode ``0600`` throughout workdir preparation
and are never planner-readable. The adapter aborts before Codex starts if this
exact two-file control set, either inode, or its metadata changes. Doctor
accepts exactly ``codex-cli 0.147.0`` with the frozen native
binary SHA-256, rejects non-empty ``/etc/codex`` managed configuration, and
requires every external runtime script to match its frozen SHA-256. The pinned
Codex process uses the named ``rpent_outer_landlock``
profile. Its full-write *Codex view* deliberately prevents Codex from stacking
bubblewrap or a second Landlock layer; it does not widen the already-installed
kernel boundary or Unix ownership checks. ``network.enabled=false`` still makes
Codex apply its restricted-network seccomp filter to every generated command;
direct local tools remain constrained by the explicit capability set and the
outer filesystem boundary.
Direct compatibility tests must prove scratch writes, rejected cell-root and
``/usr`` writes, rejected RPent/runtime/``/proc`` reads, absent planner credentials,
and rejected shell network sockets. This profile identity is part of the run
manifest and must be revalidated when the Codex binary changes. The active
preflight also captures the exact direct-tool capability set; its attestation
binds the Codex and launcher hashes, profile arguments, kernel release,
Landlock ABI, and all compatibility checks into the run manifest.
The default executable is the native binary at
``/usr/local/libexec/rpent/codex-0.147.0`` with SHA-256
``cb0a15567e9a60a5820d54b0f6ae86d504dc3805c1eab21a47f70e3eb7b73a40``.
Install that exact owned, non-group-writable file before running; a JavaScript
``/usr/bin/codex`` wrapper is not an equivalent default.

GPU scheduling, run, and resume
-------------------------------

The full runner assigns one serial worker to each GPU. A two-card launch is:

.. code-block:: bash

   rpent-reproduce robocasa run $COMMON_ARGS \
     $PLANNER_AUTH_ARGS \
     --selection full --gpus 0,1 \
     --max-attempts 3 --retry-backoff-seconds 60

A single card is supported with ``--gpus 0``. For strict phase ordering, use
the same results root and run these commands in sequence:

.. code-block:: bash

   rpent-reproduce robocasa run $COMMON_ARGS $PLANNER_AUTH_ARGS \
     --selection atomic --gpus 0
   rpent-reproduce robocasa run $COMMON_ARGS $PLANNER_AUTH_ARGS \
     --selection composite_seen --gpus 0
   rpent-reproduce robocasa run $COMMON_ARGS $PLANNER_AUTH_ARGS \
     --selection composite_unseen --gpus 0

The ``full`` selection queues those splits in the same atomic,
composite-seen, and composite-unseen order. Separate commands provide a hard
phase boundary when later splits must not start while the previous split has
a tail cell still running on another GPU.

The preliminary parity setup used two RTX 4090 GPUs with one serial
simulator/VLA worker per card. This is a reference configuration, not a
minimum requirement; parallelism changes throughput, not the frozen matrix or
scoring contract. The repository does not yet publish an audited 340-cell
wall-clock total or LLM API cost. Both vary with cell outcomes, retries,
provider pricing, and token usage. Before reporting them, retain the runner
JSON fields ``gpus``, ``started_at``, ``finished_at``, and
``elapsed_seconds``, record the exact GPU/VRAM/driver configuration, and use
the provider billing export. Do not infer cost from request count alone.

The runner holds non-blocking OS locks for the complete results root and every
requested GPU until all workers exit. A concurrent process targeting the same
results root or GPU fails before scheduling any cell, preventing duplicate
held-out episodes and last-writer artifact replacement. Persistent lock files
carry no state; ownership is determined only by the live kernel lock.

Every run, including a single cell or smoke selection, performs both the
frozen checkpoint hash check and active isolation preflight automatically
before scheduling cells. The resulting private attestations and
``_run_manifest.json`` bind checkpoint, memory, planner, isolation, navview
patch, and runtime implementation identities to the results root.
Those identities are recomputed before every cell, and each cell audit is
cross-checked against the root manifest. Runtime, adapter, memory, checkpoint,
or navview drift during a long run therefore stops execution rather than
mixing provenance.

The same command is the resume command. On restart, it uses the validator as
the single source of truth and skips only cells with a canonical completion
manifest, matching audit/trace identities, allowed actions, and valid hashes.
Missing, incomplete, or invalid cells are scheduled again. A persistent
``_FATAL_STOP.json`` requires investigation rather than automatic deletion.
Changing a bound run input requires a new empty results root; mixed-provenance
resume is rejected rather than overwritten.

The RPent-owned deadline supervisor loads the frozen planner wrapper from
verified source bytes and gives the planner and driver one absolute monotonic
deadline. After a primitive has completed and its state, log, and trace are
durable, the driver must acquire a root-only commit gate before publishing
``done_NN``. It records a post-publication monotonic receipt and removes the
marker if publication crossed the deadline; the gate also rejects a command
that reaches it after the deadline. At expiry, the supervisor seals the same
gate, stops the driver, and records a hash-bound freeze of the contiguous
committed prefix. This prevents an in-flight primitive from becoming canonical
during planner descendant cleanup.

The final state in that frozen prefix determines the benchmark outcome. A
deadline-prefix ``success=true`` remains a canonical success even if the Codex
turn does not exit before the deadline; otherwise the timeout is a canonical
benchmark failure. The audit, trace, planner status, deadline contract, and
seal/freeze attestations must all agree. Timeout evidence also includes every
state, done marker, command log, commit receipt, and the raw trace named by the
freeze. Offline validation hashes those files and reconstructs the frozen
prefix rather than trusting summary fields alone. A timeout log may end without
``turn.completed`` and may contain a syntactically recognized reconnect event.
Outside the timeout case, every reconnect must recover to ``turn.completed``;
unknown or explicit failure events remain invalid.

Validation, summary, and publication gate
-----------------------------------------

Validate an individual cell or the complete matrix, then write a stable
summary:

.. code-block:: bash

   rpent-reproduce robocasa validate \
     --results-root /path/to/results \
     --split atomic --task OpenDrawer --seed 1
   rpent-reproduce robocasa validate --results-root /path/to/results
   rpent-reproduce robocasa validate --results-root /path/to/results \
     --require-publication-ready
   rpent-reproduce robocasa summarize \
     --results-root /path/to/results \
     --output /path/to/results/summary.json \
     --require-publication-ready

A zero exit without ``--require-publication-ready`` means integrity/completeness
validation passed; it does not authorize publication. A run is publication-ready
only when the complete validator additionally reports ``publication_ready:
true`` and ``root_release_ready: true``, with ``expected: 340`` and ``complete:
true`` and no invalid or incomplete cells.
The checked-in schema currently accepts only the preliminary external runtime,
so ``root_release_ready`` and ``publication_ready`` are deliberately always
false and ``--require-publication-ready`` must fail. Enabling publication
requires a separately reviewed formal-runtime schema and immutable release
assets; editing a self-hashed manifest cannot enable the gate.
Canonical failures are valid observations and must not be deleted or silently
rerun to improve the score. Only after all 340 cells pass validation may the
task-weighted result be compared with the paper's reported value. Smoke,
partial, resumed-but-incomplete, or preliminary local-hybrid results must be
labelled as such and must not be compared with the paper's task-weighted
RoboCasa365 result of 55.4% described in :doc:`../awesome_works/harnessvla`.
