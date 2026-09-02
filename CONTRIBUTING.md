# Contributing to RPent

[简体中文](CONTRIBUTING_zh.md)

Thank you for your interest in RPent!

RPent welcomes contributions from the community. Bug reports, fixes, tests,
documentation improvements, performance work, and new planner, robot, or tool
integrations all help the project grow.

For integration work, start with the existing guides for
[custom planners](docs/source-en/rst_source/usage/configure_planner.rst),
[new robots](docs/source-en/rst_source/development/add_robot.rst), and
[new primitives](docs/source-en/rst_source/development/add_primitive.rst).

## 1. Contribution workflow

### Step 1. Fork the repository and create a branch

1. Fork [RLinf/RPent](https://github.com/RLinf/RPent) on GitHub.
2. Clone your fork and add the main repository as `upstream`:

   ```bash
   git clone https://github.com/<your-username>/RPent.git
   cd RPent
   git remote add upstream https://github.com/RLinf/RPent.git
   ```

3. Create a topic branch from the latest `main`:

   ```bash
   git fetch upstream
   git checkout -b feat/<short-description> upstream/main
   ```

Use a branch name that reflects the change, such as
`feat/new-planner`, `fix/tool-cancellation`, `test/cli-contract`, or
`docs/robotwin-setup`.

### Step 2. Set up the development environment

```bash
# Create and activate an isolated environment.
uv venv --python 3.11
source .venv/bin/activate

# Install the source checkout and the lightweight test dependencies.
uv pip install -e ".[test]"

# For robot work, choose the matching test environment instead.
uv pip install -e ".[test,libero-pro]"  # LIBERO
uv pip install -e ".[test,robocasa]"    # RoboCasa
uv pip install -e ".[test,robotwin]"    # RoboTwin

# Install and enable the repository hooks.
uv pip install pre-commit==4.6.2
pre-commit install
```

The commands above are alternatives: install only the robot extra relevant to
the change. See the
[installation guide](docs/source-en/rst_source/installation.rst) for
environment-specific setup.

### Step 3. Develop

Follow these conventions while developing:

- **Keep the change focused.** A feature, an unrelated refactor, and
  repository-wide cleanup should normally be separate pull requests.
- **Import optional dependencies only where they are used.** For example,
  import `robosuite` inside the RoboCasa code that needs it, not at module
  import time. This keeps `import rpent`, robot discovery, and CLI help usable
  without every simulator installed. Put integration-only dependencies in an
  optional extra.
- **Preserve existing contracts.** If an API or user-visible behavior changes,
  explain the change and how users should adapt.
- **Write tests with the change.** New behavior and bug fixes should include
  tests. Refactors should keep existing tests passing; if behavior changes,
  update the tests and explain why. Unit tests must run offline on an ordinary
  CPU machine; use fakes when model or service output is needed.
- **Follow the project style.** Pre-commit handles Ruff formatting and linting.
  Public Python APIs should have type hints and Google-style docstrings, and
  runtime messages should use the project logger rather than `print`.

### Step 4. Run checks and update documentation

Before opening a pull request, run the same required checks as CI:

```bash
# Format and lint the complete repository.
pre-commit run --all-files

# Run all unit tests.
pytest tests/unit_tests -v

# Run a focused test while developing.
pytest tests/unit_tests/rpent/cli/test_main_contracts.py -k xxx -x
```

If a check fails or modifies files, review the changes and run it again.

Update documentation alongside user-visible changes:

- keep `docs/source-en/` and `docs/source-zh/` aligned when both cover the
  feature;
- update docstrings and examples for public APIs;
- update the README when installation, quick-start, or advertised features
  change.

### Step 5. Commit and open a pull request

Commit messages follow
[Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<optional-scope>): <description>
```

Common types include:

- `feat`: user-visible functionality;
- `fix`: a bug fix;
- `docs`: documentation only;
- `test`: tests only;
- `ci`: CI or workflow changes;
- `refactor`: code restructuring without a behavior change;
- `perf`: a performance improvement;
- `style`: formatting without a semantic change;
- `build`: packaging or build changes;
- `chore`: maintenance work.

Examples:

```text
feat(planner): add an example provider adapter
fix(toolkit): retain the original handler error
test(cli): cover an offline planner run
docs(robotwin): clarify runtime setup
```

Pull request titles should use the same format. Keep the scope lowercase and the
description short and specific.

Push the branch to your fork and open a pull request against
`RLinf/RPent:main`. A helpful pull request description includes:

- the problem and why the change is needed;
- a concise summary of the implementation;
- a linked issue when one exists (`Fixes #123` or `Refs #123`);
- compatibility changes, migrations, new dependencies, and runtime impact;
- the exact automated checks that were run;
- any manual GPU, simulator, service, or robot verification.

Review the complete diff yourself before marking the pull request ready for
review.

## 2. Project red lines

- Do not weaken or delete a failing test merely to make CI pass.
- Do not commit credentials, logs, checkpoints, downloaded assets, or generated
  results.
- Do not make required CI depend on network access, model services, simulators,
  or robot hardware.

## 3. Responsible use of AI

AI coding tools are welcome, but the contributor remains the author of the
change and is responsible for everything submitted.

- Read and understand the complete diff before publishing it.
- Run the code and the relevant tests yourself.
- Remove generated code, comments, or abstractions that do not belong in the
  project.
- Keep AI-assisted pull requests focused and reviewable rather than submitting
  a large unreviewed change.
- Be prepared to explain the design and implementation during review.

AI-assisted code follows the same compatibility, documentation, testing, and
quality requirements as any other contribution.

## 4. Getting help

If you need help or have a question:

- report bugs through [Issues](https://github.com/RLinf/RPent/issues);
- contact the maintainers through the WeChat group linked in the
  [README](README.md).

Thank you for helping make RPent more reliable and useful.
