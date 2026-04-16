# ValHaLA: Variational Hamiltonian Least Action Solver for Piano Fingering

**Supplementary Material** for the accompanying manuscript submitted to
the *Journal of New Music Research*.

## Contents

| File | Lines | Description |
|------|-------|-------------|
| `valhala_solver_standalone.py` | 1,493 | Complete 14-term equation-of-state solver |
| `run_demonstrations.py` | 568 | Generates all 8 paper figures |
| `README.md` | --- | This file |
| `LICENSE` | --- | MIT license |
| `requirements.txt` | --- | Python dependencies |

## Requirements

- Python 3.9 or later
- NumPy >= 1.20
- Matplotlib >= 3.5 (for figure generation only; the solver itself requires only NumPy)

## Installation

```bash
pip install numpy matplotlib
```

## Quick Start

```python
from valhala_solver_standalone import HamiltonianSolver

# Solve a C major scale (MIDI pitches) with Modern tradition
solver = HamiltonianSolver(tradition='modern')
result = solver.solve([60, 62, 64, 65, 67, 69, 71, 72])
print(result['fingers'])  # [1, 2, 3, 1, 2, 3, 4, 5]
print(result['cost'])     # Total action (dimensionless)
```

## Pedagogical Traditions

Eight documented traditions spanning five centuries of keyboard instruction:

| Tradition | Key Figure(s) | Period |
|-----------|--------------|--------|
| `baroque` | Diruta, Santa Maria | 1565--1750 |
| `classical` | C.P.E. Bach, Czerny | 1753--1850 |
| `romantic` | Leschetizky, Liszt | 1850--1920 |
| `modern` | Neuhaus, contemporary | 1920--present |
| `russian` | Igumnov, Goldenweiser | 1900--present |
| `french` | Cortot, Long | 1900--1960 |
| `taubman` | Taubman | 1960--present |
| `chopin` | Chopin (reconstructed) | 1830--1849 |

## Solver API

### `HamiltonianSolver(tradition, hand_length, hand_breadth, ...)`

Main solver class. Key parameters:

- `tradition` (str): Pedagogical tradition name (default: `'modern'`)
- `hand_length` (float): Hand length in mm (default: 190.0)
- `hand_breadth` (float): Hand breadth in mm (default: 85.0)
- `mu_f` (float): Coulomb friction coefficient (default: 0.5)
- `tempo_nps` (float): Notes per second (default: 6.0)
- `technique` (str): `'preload'` or `'impact'` (default: `'preload'`)

### `solver.solve(midi_or_notes)`

Returns a dict with keys:
- `fingers`: List of finger numbers (1=thumb, 5=little)
- `cost`: Total action (dimensionless)
- `cost_breakdown`: Dict mapping equation-of-state terms to their contributions
- `per_note_cost`: Per-note transition costs
- `ari_values`: Action-Risk Index values (4-note rolling window)
- `ari_risk_levels`: Risk classification per note
- `tradition`: Name of the pedagogical tradition used

### Embedded Test Passages

- `passage_bach_invention_13()`: Bach Invention No. 13 in A minor, mm. 1--4
- `passage_chopin_op10_1()`: Chopin Etude Op. 10 No. 1, mm. 1--2
- `passage_bach_wtc_fugue()`: Bach WTC I, C major Fugue, subject
- `passage_c_major_scale()`: Two-octave C major ascending scale
- `passage_chopin_op25_6()`: Chopin Etude Op. 25 No. 6, mm. 1--2

### Generate all paper figures

```bash
python run_demonstrations.py
# Produces 8 PDF figures at 600 DPI in the working directory
```

## Finger Numbering Convention

Throughout: R1--R5 and L1--L5, where 1 = thumb and 5 = little finger.

## License

MIT License. See `LICENSE` for details.
