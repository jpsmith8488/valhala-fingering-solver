#!/usr/bin/env python3
"""
make_threshold_figure.py — regenerate Figure 1 (threshold.pdf).

Calls the frozen ValHaLA solver on the same three conditional terms
exercised by validation_battery.py and produces the three-panel figure
that appears as Figure 1 in the manuscript: for each term, the Hamming
distance of the optimal fingering from the term-off baseline is plotted
against the term's coefficient, for one passage that engages the term
and one that does not.

This script shares its engaging passages with validation_battery.py so
the figure and the battery report consistent numbers. The non-engaging
passages are chosen to expose the passage-dependent nature of each term:

  Coupling   engaging: a tight cluster on weak-finger pairs
             non-engaging: a wide arpeggio (no weak-pair transitions)
  Stagger    engaging: alternating white/black keys
             non-engaging: an all-black-key passage (no color changes)
  Rotation   engaging: alternating low/high pitches
             non-engaging: a monotonic scale

Output: threshold.pdf in the current directory.

License: MIT
"""
from __future__ import annotations
import copy
import numpy as np

import valhala_solver_standalone as vs
from valhala_solver_standalone import HamiltonianSolver, SolverConfig
import run_demonstrations as rd


# --- Engaging passages (shared with validation_battery.py) ----------
P_COUPLING_ENG = [79, 81, 83, 84, 86]            # weak-pair cluster
P_STAGGER_ENG  = [60, 61, 63, 65, 66, 68]        # alternating colors
P_ROTATION_ENG = [60, 79, 62, 81, 64, 83]        # alternating low/high

# --- Non-engaging counterparts --------------------------------------
P_COUPLING_NON = [60, 67, 72, 79]                # wide arpeggio
P_STAGGER_NON  = [61, 63, 66, 68, 70]            # all black keys
P_ROTATION_NON = [60, 62, 64, 65, 67, 69]        # monotonic scale

NOMINAL = {'alpha_coupling': 1.5, 'alpha_bk': 2.0, 'alpha_rot': 1.5}

COEFFS = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 18.0]


def fingers(r):
    return [int(x) for x in r.fingers]


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def solve(midis, *, tempo=6.0, mu_f=0.5, **scales):
    cfg = SolverConfig(tradition='modern', tempo_nps=tempo,
                       hand_length=190.0, hand_breadth=85.0,
                       max_span=210.0, mu_f=mu_f,
                       arm_weight_selection=False)
    for k, v in scales.items():
        setattr(cfg, k, v)
    return fingers(HamiltonianSolver(config=cfg).solve(
        rd.make_notes(midis, tempo)))


class module_scale:
    def __init__(self, **kw):
        self.kw = kw
        self.saved = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.saved[k] = getattr(vs, k)
            setattr(vs, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(vs, k, v)


def coupling_sweep(passage):
    base = solve(passage, alpha_coupling=0.0)
    return [(c, hamming(solve(passage, alpha_coupling=c), base))
            for c in COEFFS]


def stagger_sweep(passage):
    with module_scale(ALPHA_BLACK_KEY=0.0):
        base = solve(passage)
    out = []
    for c in COEFFS:
        with module_scale(ALPHA_BLACK_KEY=c):
            out.append((c, hamming(solve(passage), base)))
    return out


def rotation_sweep(passage):
    with module_scale(ALPHA_ROTATION=0.0):
        base = solve(passage, tempo=12.0)
    out = []
    for c in COEFFS:
        with module_scale(ALPHA_ROTATION=c):
            out.append((c, hamming(solve(passage, tempo=12.0), base)))
    return out


def main():
    plt = rd.setup_matplotlib()

    fig, axes = plt.subplots(1, 3, figsize=(rd.DOUBLE_COL, 2.3),
                             sharey=True)

    panels = [
        ('Inter-digit coupling ($\\alpha_{\\mathrm{coupling}}$)',
         coupling_sweep(P_COUPLING_ENG),
         coupling_sweep(P_COUPLING_NON),
         NOMINAL['alpha_coupling']),
        ('Depth stagger ($\\alpha_{\\mathrm{bk}}$)',
         stagger_sweep(P_STAGGER_ENG),
         stagger_sweep(P_STAGGER_NON),
         NOMINAL['alpha_bk']),
        ('Forearm rotation ($\\alpha_{\\mathrm{rot}}$)',
         rotation_sweep(P_ROTATION_ENG),
         rotation_sweep(P_ROTATION_NON),
         NOMINAL['alpha_rot']),
    ]

    for ax, (title, eng, non, nominal) in zip(axes, panels):
        xs_e, ys_e = zip(*eng)
        xs_n, ys_n = zip(*non)
        ax.plot(xs_e, ys_e, marker='o', color='#0072B2',
                label='engaging passage')
        ax.plot(xs_n, ys_n, marker='s', color='#D55E00',
                linestyle='--', label='non-engaging passage')
        ax.axvline(nominal, color='gray', linestyle=':', linewidth=0.8)
        ax.set_xlabel('term coefficient')
        ax.set_title(title)
        ax.set_ylim(-0.3, 5.3)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Hamming distance from baseline')
    axes[0].legend(loc='upper left', frameon=False)

    fig.tight_layout()
    fig.savefig('threshold.pdf')
    print('[OK] threshold.pdf')


if __name__ == '__main__':
    main()
