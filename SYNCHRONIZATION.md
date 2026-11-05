# Repository synchronization - 2026-11-05

## Current state

Local main was fast-forwarded from f5d18309e7679a01e4d27f3f5e8e620793327876
to fetched origin/main 320a142c155881fd963207e2c55701a079188252, incorporating
all seven incoming commits. HEAD and origin/main are equal (ahead 0, behind 0).
No push, reset, force operation or benchmark was performed. Our implementation
changes remain in the working tree, ready for review and a later source commit.

Before synchronization, all tracked/untracked files were backed up outside this
repository (124 files, approximately 6 MB), along with the Git patch, refs and
SHA256 inventory. A second recovery copy remains in Git stash commit
f5dfb8c168739afd1a6b5fbf040911d6f3cf3144. The stash records old paths and was not
blindly popped over the remote's reorganized files. No local work was discarded.

## Structure follows the remote

```text
code/
  run_part_1.py / run_part_1.ps1     canonical classical A-H entry points
  run_deepmodels.py                 Part 4 A-F entry point
  our_src/
    part1/
      artifact_json.py
      smoke_stages.py
      stagea/ ... stageg/           fetched remote locations
      stageh/                      our retained comparison extension
    deepmodels/                    our unchanged Part 4 implementation
  results/part1/stageA ... stageG   historical artifacts tracked upstream
  results/part1/stageH             future generated H output, ignored
  deepmodels/                      generated Part 4 output/cache, ignored
```

The old run_pipeline.py, run_all.ps1 and old root stage module paths were not
reintroduced. Local A-H behavior is retained under the remote runner names and
nested package locations. Prior local stageA through stageH artifact directories
remain intact as historical outputs, not silently merged into upstream results.

## Our logic is preserved

- Compared 22 incoming Part 1 computational source files against our pre-sync
  files using Python ASTs after explicit import/filename/path substitutions.
  They match: incoming differences are structural rather than new algorithms.
- All nine Part 4 Python files and its complete config.json remain byte-identical
  to the pre-sync local versions.
- Our Stage H implementation is retained, with AST equality after only import
  substitutions and the project-root-depth correction. All experimental values
  remain unchanged; only stage_a_dir/output_dir use results/part1.
- Existing Part 1 experimental values also remain unchanged; its output and
  dependency directories use the fetched canonical result namespace.
- All seven professor-provided Python files under src/ retain their original
  SHA256 hashes. Incoming historical results were not altered.
- The local H runner/smoke support and its test assertions are retained under
  run_part_1 and our_src.part1; source mocks/config references follow the new names.

The remote moves exposed project-root defaults one directory too shallow. These
were corrected in nested stage entry points and the smoke runner. The smoke
configuration reference was updated to stagea/config.json. These are path fixes;
no extraction, classification, selection or evaluation algorithm was rewritten.

## Lightweight validation only

- AST parsing: 54 source files passed.
- Static relative-import resolution: 114 imports resolved within the canonical tree.
- Shared Part 4 data/SVM/evaluation/protocol imports resolve to the canonical
  our_src.part1.stagea and stagec modules, without importing torch or CLIP.
- Part 4 CLI --help passed from the Git root.
- Classical runner --dry-run passed and printed the complete A-H plan with all
  eight configuration paths and consistent dependency/result directories.
- Source hashes, configuration comparisons and absence of duplicate old entry
  points were checked. No model inference, SVM fit or expensive tests were run.

The audit evidence and the full recovery backup are stored in the external
repository_sync folder used for this operation; neither is needed at runtime.

## Publication remains separate

No local implementation commit or push was made. To commit only source and
configuration changes later, review git status, then use the explicit source scope
below from the Git root. The professor sources, upstream result artifacts,
assignment PDF and local ZIP are excluded:

```bash
git switch -c codex/synchronized-local-implementation
git add -- .gitignore PART4.md SYNCHRONIZATION.md code/.gitignore code/PIPELINE.md code/run_part_1.py code/run_part_1.ps1 code/test_pipeline.py code/requirements-deepmodels.txt code/run_deepmodels.py code/our_src/part1 code/our_src/deepmodels
git diff --cached --stat
# Review before committing; do not use git add .
```

Future execution/setup instructions are in PART4.md, code/PIPELINE.md and
code/our_src/deepmodels/README.md. No execution depends on the backup or former
development workspace.
