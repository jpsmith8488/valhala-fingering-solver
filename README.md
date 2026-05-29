# ValHaLA and SCALE

This repository hosts the code for two related projects in the
mathematical modeling of piano performance:

- **ValHaLA** (repository root) --- a variational solver for optimal
  piano fingering. See below.
- **SCALE** (the [`scale/`](scale/) directory) --- analysis code, a
  validation script, and the complete Well-Tempered Clavier cost data
  (both books, with per-term cost decomposition) accompanying a separate
  manuscript. See [`scale/`](scale/) for its own documentation.

---

# ValHaLA: Variational Hamiltonian Least Action Solver for Piano Fingering

A self-contained solver that assigns optimal piano fingerings by
minimizing a fourteen-term physical cost function over the keyboard
state space, using dynamic programming. It accompanies the manuscript

> J. P. Smith, *Optimal Piano Fingering from a Physically Parameterized
> Variational Framework with Inverse Parameter Recovery* (submitted,
> *Journal of Mathematics and Music*).

Repository: <https://github.com/jpsmith8488/valhala-fingering-solver>

**Reviewers:** see [`REPRODUCE.md`](REPRODUCE.md) for a single-command
verification path against every figure and table in the manuscript.

## Contents

| File | Description |
|------|-------------|
| `valhala_solver_standalone.py` | Complete 14-term equation-of-state solver (frozen, MD5 `846ce6aae16623c6ca4a551f86df869c`) |
| `run_demonstrations.py` | Regenerates the six data figures (D1, D2, D4, D5, D6, F7) |
| `make_threshold_figure.py` | Regenerates Figure 1 (the threshold figure) |
| `validation_battery.py` | Mechanism-isolation validation suite (Table 3) |
| `reproduce_paper.py` | End-to-end check of every headline quantity |
| `verify_figures_tables.py` | Exact figure- and table-value correspondence check |
| `frozen_values.txt` | Reference values used in the manuscript |
| `SHA256SUMS.txt` | Checksums for every file in this package |
| `REPRODUCE.md` | Reviewer-facing reproducibility instructions |
| `REPRODUCTION_REPORT.txt` | Build report and known issues |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT license |

(The `scale/` directory belongs to the separate SCALE project; see the
note at the top of this file.)

## Requirements

- Python 3.9 or later
- NumPy >= 1.20
- Matplotlib >= 3.5 (figure generation only; the solver itself needs only NumPy)

```bash
pip install -r requirements.txt
```

## Quick start

```python
from valhala_solver_standalone import HamiltonianSolver, SolverConfig
import run_demonstrations as rd

# C major scale, Modern tradition
solver = HamiltonianSolver(config=SolverConfig(tradition='modern', tempo_nps=4.0))
result = solver.solve(rd.make_notes([60, 62, 64, 65, 67, 69, 71, 72], 4.0))
print([int(f) for f in result.fingers])  # [1, 2, 3, 1, 2, 3, 4, 5]
print(result.total_cost)                 # total action (dimensionless)
```

## Reproducing the paper

```bash
python reproduce_paper.py            # checks every headline quantity; exits 0 on success
python verify_figures_tables.py      # checks every figure and table value
python validation_battery.py         # reproduces the Table 3 verdicts
python run_demonstrations.py         # regenerates D1, D2, D4, D5, D6, F7
python make_threshold_figure.py      # regenerates Figure 1 (threshold)
```

`reproduce_paper.py` runs each reported result through the frozen solver
and compares it to the value stated in the manuscript: the emergent
scale fingerings, the Chopin Op. 10/1 edition agreement across three
hand sizes, the hand-size Action-Risk Index peaks, the dry/moist
environment costs, the eight scaling exponents, and the Baroque--Modern
crossover.

`run_demonstrations.py` produces the six data figures by default. The
two supplementary figures (D3 Russian vs French-Cortot, F8 stochastic
robustness heatmap) are not referenced in the manuscript; pass
`--include-extras` to generate them as well.

## Solver API

`HamiltonianSolver(config=SolverConfig(...))` with fields including
`tradition`, `hand_length`, `max_span`, `mu_f`, `tempo_nps`, and
`arm_weight_selection`. `solver.solve(notes)` returns a `SolverResult`
with `.fingers`, `.total_cost`, `.cost_breakdown`, and `.ari_values`.

### Pedagogical traditions

`baroque`, `classical`, `romantic`, `modern`, `russian`, `french`,
`taubman`, `chopin` --- eight documented schools spanning five centuries
of keyboard instruction, encoded as cost-function coefficient vectors.

### Embedded test passages

`passage_bach_invention_13()`, `passage_chopin_op10_1()`,
`passage_bach_wtc_fugue()`, `passage_c_major_scale()`,
`passage_chopin_op25_6()`.

## Finger numbering

1 = thumb, 5 = little finger, for both hands.

## Reproducibility note

The solver is deterministic. The file `valhala_solver_standalone.py` is
the frozen solver against which every number in the manuscript was
computed; the demonstration and validation scripts import it unchanged.
Its MD5 is `846ce6aae16623c6ca4a551f86df869c` and must not change for
the reproducibility claims to hold.

## Citation

```bibtex
@article{smith2026valhala,
  author  = {Smith, Justin P.},
  title   = {Optimal Piano Fingering from a Physically Parameterized
             Variational Framework with Inverse Parameter Recovery},
  journal = {Journal of Mathematics and Music},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT License. See `LICENSE`.
