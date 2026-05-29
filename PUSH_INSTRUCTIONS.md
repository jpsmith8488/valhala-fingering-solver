# Pushing this revision to GitHub

This file is a hand-off note from the chat session that built this
release directory. It is not part of the package itself — delete it
after the push, or leave it as a development log; either is fine.

## What is in this directory

The file tree is exactly what should sit at the root of
`github.com/jpsmith8488/valhala-fingering-solver` after the push:

```
LICENSE
README.md                       # patched
REPRODUCE.md                    # NEW — reviewer-facing reproduction guide
REPRODUCTION_REPORT.txt         # patched
SHA256SUMS.txt                  # regenerated
frozen_values.txt               # patched
make_threshold_figure.py        # NEW — regenerates Figure 1
reproduce_paper.py              # patched (mu_c target 1.15 -> 1.20)
requirements.txt
run_demonstrations.py           # patched (--include-extras flag)
valhala_solver_standalone.py    # UNCHANGED, MD5 846ce6a…
validation_battery.py           # patched (NOVEL -> mechanism-derived)
verify_all.sh                   # NEW — one-command verification driver
verify_figures_tables.py        # patched (mu_c target 1.15 -> 1.20)
```

The `scale/` directory in your existing repo is untouched by this
revision — none of these files belong inside it.

## Recommended commit message

```
Reviewer reproducibility revision

Manuscript-side corrections folded into the code and docs:

- mu_c crossover value 1.15 -> 1.20 (matches frozen-solver sweep);
  reflected in reproduce_paper.py and verify_figures_tables.py.
- run_demonstrations.py: D3 and F8 (not referenced in the manuscript)
  moved behind a --include-extras flag; default now produces exactly
  the six data figures the paper uses.
- validation_battery.py: "NOVEL" relabeled "mechanism-derived" in
  threshold-test labels and module docstring.

New files:
- make_threshold_figure.py — regenerates Figure 1 from the same
  passages used by validation_battery.py.
- REPRODUCE.md — reviewer-facing reproduction guide.
- verify_all.sh — one-command verification driver.

The frozen solver (valhala_solver_standalone.py, MD5 846ce6a…) is
unchanged. All 24 + 22 + 6/3/1 reproducibility checks pass.
```

## Push procedure (on your Mac)

Replace `<path-to-this-directory>` with wherever you unpacked the
release tarball.

```bash
# 1. Clone the existing repo into a scratch directory
cd ~/Documents
git clone git@github.com:jpsmith8488/valhala-fingering-solver.git scratch-valhala
cd scratch-valhala

# 2. Verify you are on main (or whichever branch you publish from)
git status
git branch

# 3. Overlay the new files. The * glob is intentional: it copies
#    every file in the release directory into the repo root, leaving
#    the scale/ subdirectory and .git/ untouched.
cp -v <path-to-this-directory>/* .

# 4. Sanity check: confirm the frozen solver MD5 still matches
md5 valhala_solver_standalone.py
# expect: MD5 (valhala_solver_standalone.py) = 846ce6aae16623c6ca4a551f86df869c

# 5. Run the verifier locally before pushing
pip install -r requirements.txt
bash verify_all.sh
# expect: "All reproducibility checks passed." at the end

# 6. Stage, commit, push
git add -A
git status     # review the diff one last time
git commit -F - <<'EOF'
Reviewer reproducibility revision

Manuscript-side corrections folded into the code and docs:

- mu_c crossover value 1.15 -> 1.20 (matches frozen-solver sweep);
  reflected in reproduce_paper.py and verify_figures_tables.py.
- run_demonstrations.py: D3 and F8 (not referenced in the manuscript)
  moved behind a --include-extras flag; default now produces exactly
  the six data figures the paper uses.
- validation_battery.py: "NOVEL" relabeled "mechanism-derived" in
  threshold-test labels and module docstring.

New files:
- make_threshold_figure.py — regenerates Figure 1 from the same
  passages used by validation_battery.py.
- REPRODUCE.md — reviewer-facing reproduction guide.
- verify_all.sh — one-command verification driver.

The frozen solver (valhala_solver_standalone.py, MD5 846ce6a) is
unchanged. All 24 + 22 + 6/3/1 reproducibility checks pass.
EOF
git push origin main   # or whichever branch you're using
```

## Optional: tag the submission revision

If you'd like a permanent, citable marker for the JM&M submission state:

```bash
git tag -a jmm-submission-2026-05 -m "Snapshot at JM&M submission, May 2026"
git push origin jmm-submission-2026-05
```

Reviewers (and your future self) can then `git checkout jmm-submission-2026-05`
to retrieve the exact code state that backs the submitted manuscript.

## After the push

Delete this PUSH_INSTRUCTIONS.md (it has no purpose in the repo). The
remaining tree is the reviewer package.
