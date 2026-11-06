# Part 4 execution

The Git root contains `code/`. The implementation is
`code/our_src/part4/`, and the repository entry point is
`python code/run_deepmodels.py --help`.

See [Part 4 setup and execution](code/our_src/part4/README.md) for Linux and
Windows installation, separately supplied dataset setup, checkpoint caches and
stage commands. Dependencies are in `code/requirements-deepmodels.txt`; the
professor's requirements and original project organization are preserved.

The current source supports the fetched remote Part 1 layout
`our_src/part1/stagea/` as well as older checkouts. No model inference or benchmark
was run during repository integration.

New stage outputs: `code/results/part4/stageA/` through `stageF/`; shared manifests and comparisons in `code/results/part4/runs/<id>/` (or `smoke/<id>/`). Historical `code/deepmodels/` outputs are preserved.
