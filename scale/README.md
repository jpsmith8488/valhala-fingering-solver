# SCALE: Spectral Chromatic Action Landscape Evaluation

**Supplementary Material** for the accompanying manuscript submitted to
the *Journal of New Music Research*.

## Contents

| File / Directory | Description |
|-----------------|-------------|
| `run_scale.py` | SCALE analysis runner (standalone) |
| `run_validation.py` | Synthetic validation battery (4 tests) |
| `data/Book_1_scale_data.json` | Complete Book 1 results (24 keys × 12 offsets, per-term breakdowns) |
| `data/Book_2_scale_data.json` | Complete Book 2 results (24 keys × 12 offsets, per-term breakdowns) |
| `README.md` | This file |

## Requirements

- Python 3.9 or later
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.5 (for figure generation only)
- music21 >= 8.0 (for MusicXML parsing; not needed to inspect pre-computed data)
- The ValHaLA solver (`valhala_solver_standalone.py` from the parent repository)

## Reproducing the Analysis

### 1. Validation (no external data needed)

The validation script uses synthetic passages embedded in the code:

```bash
python run_validation.py --solver-path ../valhala_solver_standalone.py
```

This runs four tests (white-key scale, black-key pentatonic, single repeated
note, chromatic scale) and prints PASS/FAIL verdicts.

### 2. Full WTC Analysis

To reproduce the WTC results, you will need MusicXML files for both books
of Bach's Well-Tempered Clavier.  The files used in the paper were obtained
from the MuseScore community repository:

  https://musescore.com/user/10477211/scores/26636911

Transcriber: Ng Kiat Quan.  Book I encoded 2025-07-18 (MuseScore 4.5.2);
Book II encoded 2022-05-09 (MuseScore 3.6.2).

```bash
python run_scale.py \
    --solver-path ../valhala_solver_standalone.py \
    --book1 path/to/wtc_book1.mxl \
    --book2 path/to/wtc_book2.mxl \
    --output-dir results/
```

### 3. Inspecting Pre-Computed Data

The `data/` directory contains the complete results as JSON files.  Each file
contains per-piece cost vectors, per-term breakdowns (all 14 equation-of-state
terms), rankings, effort spreads, and aggregate statistics.  These can be
loaded directly:

```python
import json
with open("data/Book_1_scale_data.json") as f:
    results = json.load(f)

for piece in results['pieces']:
    print(f"BWV {piece['bwv']} {piece['display_name']}: "
          f"rank {piece['original_rank']}, spread {piece['spread']*100:.1f}%")
```

## Data Format

Each `scale_data.json` contains:

- `metadata`: solver version, tradition, hand length, number of keys, elapsed time
- `pieces`: list of 24 analytical units, each containing:
  - `key`, `bwv`, `display_name`, `accidentals`, `note_count`
  - `cost_total`: 12-element list (total action per transposition offset)
  - `cost_rh`, `cost_lh`: 12-element lists (per-hand action)
  - `breakdown_total`: list of 12 dicts, each mapping the 14 EoS term names to their contribution at that offset
  - `ranks`: 12-element list (rank of each offset, 1 = lowest cost)
  - `original_rank`: rank of offset 0 (the original key)
  - `spread`: proportional effort spread (max - min) / min
  - `optimal_key`, `optimal_idx`: the transposition with lowest total cost

## Citation

If you use this code or data, please cite:

> Smith, J. P. (2026). Spectral Chromatic Action Landscape Evaluation of
> Bach's Well-Tempered Clavier: A Quantitative Complement to Traditional
> Musicological Analysis of Key Choice. *Journal of New Music Research*
> (submitted).

## License

MIT License.  See the parent repository for details.
