# RPent Documentation

RPent's documentation is built with Sphinx. English and Simplified Chinese
sources live in parallel trees so local builds and Read the Docs use the same
layout.

## Set up the environment

From the repository root:

```bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
uv venv --python 3.11 .docs-venv
source .docs-venv/bin/activate
uv pip install -r docs/requirements.txt
```

The locale variables should be set in each new terminal used to build the
documentation. The Python version matches the Read the Docs build environment.

## Build and preview

Run the live-reloading English preview:

```bash
cd docs
bash autobuild.sh
```

For the Chinese documentation:

```bash
cd docs
bash autobuild.sh zh
```

Both commands serve the generated site at <http://localhost:8000> and rebuild
it when a source file changes.

If port 8000 is already in use, pass another port or use `0` to select a free
port automatically:

```bash
bash autobuild.sh zh 8001
bash autobuild.sh zh 0
```

To build without starting the preview server:

```bash
cd docs
sphinx-build -W --keep-going source-en build/html-en
sphinx-build -W --keep-going source-zh build/html-zh
```

You can also use Make:

```bash
cd docs
make html LANG=en
make html LANG=zh
```

## Clean generated files

```bash
cd docs
make clean
```

The generated `docs/build/` directory and `.docs-venv/` environment are not
tracked by Git.

## Write documentation

Documentation pages use reStructuredText (RST). Keep English and Chinese pages
at matching relative paths so the language switcher can open the same page in
the other language.

- English: `source-en/rst_source/`
- Chinese: `source-zh/rst_source/`
- RST syntax reference: <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>

## Benchmark figures and demo

The benchmark pages share chart, stylesheet, and video assets from `docs/_static/`.
Both language configurations copy this directory into the built `_static/` tree;
the complete demo MP4 is stored once in the repository. The poster is an unchanged
frame from that video. The demo retains its GPT-6 Astra / GPT-5.6 xhigh labels and
4× playback, separately from the GPT-5.5 evaluation records.

See [the figure generation guide](tools/README.md) for reproducible leaderboard
SVG/PNG exports and checks against the bilingual result tables. Sphinx uses the
committed figures directly; plotting dependencies are only needed to regenerate
them.
