#!/usr/bin/env python3
"""
Demonstration Figure Generator
============================================

Generates all 8 publication-quality figures for the accompanying
manuscript on variational piano fingering.

Demonstrations:
    D1: Baroque vs. Modern tradition (Bach Invention 13)
    D2: Large vs. small hands (Chopin Op.10/1)
    D3: Russian vs. French-Cortot (Bach WTC I Fugue)
    D4: Tempo dependence (C major scale)
    D5: Dry vs. moist finger pads (Chopin Op.25/6)
    D6: Baroque-Modern crossover (order parameter)
    F7: Scaling law S(f) ~ f^alpha (all 8 traditions)
    F8: Stochastic robustness heatmap (Bach Inv. 13, Modern)

Usage:
    pip install numpy matplotlib
    python run_demonstrations.py

Output:
    8 PDF figures in the working directory.

License: MIT
"""

import math
import sys
import numpy as np

from valhala_solver_standalone import (
    HamiltonianSolver, SolverConfig, NoteEvent, TRADITIONS,
    passage_bach_invention_13, passage_chopin_op10_1,
    passage_bach_wtc_fugue, passage_c_major_scale,
    passage_chopin_op25_6, total_action,
)

# =====================================================================
# Matplotlib configuration (publication style)
# =====================================================================

def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
    })
    return plt

SINGLE_COL = 3.375
DOUBLE_COL = 6.75

# Okabe-Ito colorblind-safe palette
COLORS = {
    'baroque': '#E69F00',
    'classical': '#56B4E9',
    'romantic': '#CC79A7',
    'modern': '#0072B2',
    'russian': '#D55E00',
    'french': '#009E73',
    'taubman': '#F0E442',
    'chopin': '#000000',
}


def make_notes(midis, tempo_nps=6.0):
    dt = 1.0 / tempo_nps
    return [NoteEvent(midi=m, onset=i * dt, duration=dt * 0.9,
                      beat_position=(i % 4) / 4.0)
            for i, m in enumerate(midis)]


def normalize_ari(ari):
    if not ari or max(ari) == 0:
        return [0.0] * len(ari)
    log_ari = [math.log1p(abs(a)) for a in ari]
    mx = max(log_ari) if max(log_ari) > 0 else 1.0
    return [100.0 * v / mx for v in log_ari]


# =====================================================================
# Demonstration Runners
# =====================================================================

def run_demo1():
    midis = passage_bach_invention_13()
    results = {}
    for trad in ['baroque', 'modern']:
        cfg = SolverConfig(tradition=trad, tempo_nps=6.0)
        solver = HamiltonianSolver(config=cfg)
        notes = make_notes(midis, 6.0)
        r = solver.solve(notes)
        results[trad] = {
            'fingers': r.fingers, 'cost': r.total_cost,
            'ari': normalize_ari(r.ari_values),
            'breakdown': r.cost_breakdown,
        }
    return results, midis


def run_demo2():
    midis = passage_chopin_op10_1()
    results = {}
    for label, hl, ms in [('large', 210.0, 240.0), ('small', 170.0, 180.0)]:
        cfg = SolverConfig(hand_length=hl, max_span=ms,
                           tradition='modern', tempo_nps=4.0)
        solver = HamiltonianSolver(config=cfg)
        notes = make_notes(midis, 4.0)
        r = solver.solve(notes)
        results[label] = {
            'fingers': r.fingers, 'cost': r.total_cost,
            'ari': normalize_ari(r.ari_values),
        }
    return results, midis


def run_demo3():
    midis = passage_bach_wtc_fugue()
    results = {}
    for trad, tech in [('russian', 'preload'), ('french', 'impact')]:
        cfg = SolverConfig(tradition=trad, technique=tech, tempo_nps=5.0)
        solver = HamiltonianSolver(config=cfg)
        notes = make_notes(midis, 5.0)
        r = solver.solve(notes)
        results[trad] = {
            'fingers': r.fingers, 'cost': r.total_cost,
            'ari': normalize_ari(r.ari_values),
            'breakdown': r.cost_breakdown,
        }
    return results, midis


def run_demo4():
    midis = passage_c_major_scale()
    results = {}
    for label, nps in [('adagio', 4.0), ('allegro', 10.0), ('presto', 16.0)]:
        cfg = SolverConfig(tradition='modern', tempo_nps=nps)
        solver = HamiltonianSolver(config=cfg)
        notes = make_notes(midis, nps)
        r = solver.solve(notes)
        results[label] = {
            'fingers': r.fingers, 'cost': r.total_cost,
            'tempo': nps,
            'ari': normalize_ari(r.ari_values),
        }
    return results, midis


def run_demo5():
    midis = passage_chopin_op25_6()
    results = {}
    for label, mu in [('dry', 0.5), ('moist', 1.0)]:
        cfg = SolverConfig(tradition='modern', mu_f=mu, tempo_nps=6.0)
        solver = HamiltonianSolver(config=cfg)
        notes = make_notes(midis, 6.0)
        r = solver.solve(notes)
        tob = sum(1 for i, f in enumerate(r.fingers)
                  if f == 1 and midis[i] % 12 in {1, 3, 6, 8, 10})
        results[label] = {
            'fingers': r.fingers, 'cost': r.total_cost,
            'mu_f': mu, 'thumb_on_black': tob,
            'ari': normalize_ari(r.ari_values),
        }
    return results, midis


def run_demo6():
    midis = passage_bach_invention_13()
    mu_values = np.linspace(0.0, 2.5, 51)
    phi_values = []
    cost_values = []
    for mu in mu_values:
        cfg = SolverConfig(tradition='modern', tempo_nps=6.0)
        solver = HamiltonianSolver(config=cfg)
        solver.tradition.thumb_under = 3 + mu * 4
        solver.tradition.thumb_on_black = 2 + mu * 20
        solver.tradition.metric_weight = mu
        notes = make_notes(midis, 6.0)
        r = solver.solve(notes)
        tu = 0
        for i in range(1, len(r.fingers)):
            f1, f2 = r.fingers[i-1], r.fingers[i]
            if f1 != 1 and f2 == 1 and midis[i] > midis[i-1]:
                tu += 1
        phi = tu / max(len(r.fingers) - 1, 1)
        phi_values.append(phi)
        cost_values.append(r.total_cost)
    return {'mu': mu_values.tolist(), 'phi': phi_values,
            'cost': cost_values}, midis


def run_scaling_law():
    midis = passage_c_major_scale()
    tempos = np.linspace(2.0, 18.0, 17)
    results = {}
    for trad in TRADITIONS:
        costs = []
        for f in tempos:
            cfg = SolverConfig(tradition=trad, tempo_nps=f)
            solver = HamiltonianSolver(config=cfg)
            S = total_action(solver, midis, f)
            costs.append(S)
        results[trad] = costs
    return {'tempos': tempos.tolist(), **results}


def run_stochastic():
    midis = passage_bach_invention_13()
    cfg = SolverConfig(tradition='modern', tempo_nps=6.0)
    solver = HamiltonianSolver(config=cfg)
    notes = make_notes(midis, 6.0)
    result = solver.solve_stochastic(notes, n_trials=100, sigma=0.15)
    return result, midis


# =====================================================================
# Figure Generation
# =====================================================================

def fig_demo1(results, midis):
    plt = setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    terms = ['topographic', 'coupling', 'tradition', 'adhesion',
             'gravity', 'kinetic', 'key_action']
    term_labels = ['Topo', 'Coupl', 'Trad', 'Adhes', 'Grav', 'Kin', 'Key']
    x = np.arange(len(terms))
    w = 0.35
    for i, (trad, color) in enumerate([('baroque', COLORS['baroque']),
                                        ('modern', COLORS['modern'])]):
        vals = [results[trad]['breakdown'].get(t, 0) for t in terms]
        ax1.bar(x + i * w, vals, w, label=trad.capitalize(),
                color=color, edgecolor='black', linewidth=0.4)
    ax1.set_xticks(x + w / 2)
    ax1.set_xticklabels(term_labels, rotation=45, ha='right')
    ax1.set_ylabel('Cost contribution')
    ax1.set_title('(a) Cost decomposition by term')
    ax1.legend(frameon=False)
    ax1.axhline(0, color='gray', linewidth=0.5)

    xn = range(len(results['baroque']['ari']))
    for trad, color, ls in [('baroque', COLORS['baroque'], '-'),
                             ('modern', COLORS['modern'], '--')]:
        ari = results[trad]['ari']
        ax2.plot(range(len(ari)), ari, color=color, linestyle=ls,
                 label=trad.capitalize(), marker='.', markersize=3)
    ax2.set_xlabel('Note index')
    ax2.set_ylabel('Action-Risk Index')
    ax2.set_title('(b) ARI profile')
    ax2.legend(frameon=False)

    plt.tight_layout()
    fig.savefig('demo1_baroque_modern.pdf')
    plt.close(fig)
    print("    [OK] demo1_baroque_modern.pdf")


def fig_demo2(results, midis):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    for label, color, ls, mk in [('large', 'b', '-', 'o'),
                                  ('small', 'r', '--', 's')]:
        ari = results[label]['ari']
        ax.plot(range(len(ari)), ari, color=color, linestyle=ls,
                label=f'{label.capitalize()} '
                      f'({"210" if label=="large" else "170"} mm)',
                marker=mk, markersize=3)
    ax.set_xlabel('Note index')
    ax.set_ylabel('Action-Risk Index')
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig('demo2_hand_size.pdf')
    plt.close(fig)
    print("    [OK] demo2_hand_size.pdf")


def fig_demo3(results, midis):
    plt = setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    terms = ['topographic', 'coupling', 'tradition', 'gravity',
             'kinetic', 'key_action', 'contact']
    term_labels = ['Topo', 'Coupl', 'Trad', 'Grav', 'Kin', 'Key', 'Cont']
    x = np.arange(len(terms))
    w = 0.35
    for i, (trad, color) in enumerate([('russian', COLORS['russian']),
                                        ('french', COLORS['french'])]):
        vals = [results[trad]['breakdown'].get(t, 0) for t in terms]
        ax1.bar(x + i * w, vals, w, label=TRADITIONS[trad].name,
                color=color, edgecolor='black', linewidth=0.4)
    ax1.set_xticks(x + w / 2)
    ax1.set_xticklabels(term_labels, rotation=45, ha='right')
    ax1.set_ylabel('Cost contribution')
    ax1.set_title('(a) Cost decomposition')
    ax1.legend(frameon=False)
    ax1.axhline(0, color='gray', linewidth=0.5)

    xn = range(len(results['russian']['ari']))
    ax2.plot(xn, results['russian']['ari'], color=COLORS['russian'],
             label='Russian', marker='.', markersize=3)
    ax2.plot(range(len(results['french']['ari'])),
             results['french']['ari'], color=COLORS['french'],
             label='French-Cortot', marker='.', markersize=3, linestyle='--')
    ax2.set_xlabel('Note index')
    ax2.set_ylabel('Action-Risk Index')
    ax2.set_title('(b) ARI profile')
    ax2.legend(frameon=False)

    plt.tight_layout()
    fig.savefig('demo3_russian_french.pdf')
    plt.close(fig)
    print("    [OK] demo3_russian_french.pdf")


def fig_demo4(results, midis):
    plt = setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL, 3.5),
                                    gridspec_kw={'height_ratios': [1, 1]})
    colors_t = ['#0072B2', '#E69F00', '#D55E00']
    for i, (label, c) in enumerate(zip(['adagio', 'allegro', 'presto'],
                                        colors_t)):
        fingers = results[label]['fingers']
        ax1.plot(range(len(fingers)), fingers, marker='o', markersize=3,
                 color=c,
                 label=f"{label.capitalize()} ({results[label]['tempo']:.0f} n/s)")
    ax1.set_ylabel('Finger')
    ax1.set_yticks([1, 2, 3, 4, 5])
    ax1.set_title('(a) Optimal fingering at three tempi')
    ax1.legend(frameon=False, fontsize=7)

    for label, c in zip(['adagio', 'allegro', 'presto'], colors_t):
        ari = results[label]['ari']
        ax2.plot(range(len(ari)), ari,
                 marker='.', markersize=2, color=c, label=label.capitalize())
    ax2.set_xlabel('Note index')
    ax2.set_ylabel('Action-Risk Index')
    ax2.set_title('(b) ARI profiles')
    ax2.legend(frameon=False, fontsize=7)

    plt.tight_layout()
    fig.savefig('demo4_tempo.pdf')
    plt.close(fig)
    print("    [OK] demo4_tempo.pdf")


def fig_demo5(results, midis):
    plt = setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    for label, color, ls in [('dry', '#CD853F', '-'), ('moist', '#4682B4', '--')]:
        fingers = results[label]['fingers']
        ax1.plot(range(len(fingers)), fingers,
                 color=color, linestyle=ls, marker='o', markersize=3,
                 label=f"{label.capitalize()} "
                       f"($\\mu_f$={results[label]['mu_f']})")
    for i, m in enumerate(midis):
        if m % 12 in {1, 3, 6, 8, 10}:
            ax1.axvspan(i - 0.3, i + 0.3, color='gray', alpha=0.15)
    ax1.set_ylabel('Finger')
    ax1.set_yticks([1, 2, 3, 4, 5])
    ax1.set_title('(a) Fingering (gray = black key)')
    ax1.legend(frameon=False, fontsize=7)

    labels = ['dry', 'moist']
    costs = [results[l]['cost'] for l in labels]
    tob = [results[l]['thumb_on_black'] for l in labels]
    ax2.bar([0, 1], costs, color=['#CD853F', '#4682B4'],
            width=0.5, edgecolor='black', linewidth=0.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f"Dry\n(ToB={tob[0]})", f"Moist\n(ToB={tob[1]})"])
    ax2.set_ylabel('Total action $J$')
    ax2.set_title('(b) Total cost')

    plt.tight_layout()
    fig.savefig('demo5_adhesion.pdf')
    plt.close(fig)
    print("    [OK] demo5_adhesion.pdf")


def fig_demo6(data):
    plt = setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    mu = data['mu']
    phi = data['phi']

    ax1.plot(mu, phi, 'ko-', markersize=2)
    ax1.set_xlabel(r'Metric weight $\mu$')
    ax1.set_ylabel(r'$\phi$ (thumb-under fraction)')
    ax1.set_title(r'(a) Order parameter $\phi(\mu)$')

    if len(mu) > 2:
        chi = np.gradient(phi, mu)
        ax2.plot(mu, chi, 'b.-', markersize=2)
        ax2.set_xlabel(r'Metric weight $\mu$')
        ax2.set_ylabel(r'$\chi = \partial\phi/\partial\mu$')
        ax2.set_title(r'(b) Susceptibility $\chi(\mu)$')
        ax2.axhline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    fig.savefig('demo6_crossover.pdf')
    plt.close(fig)
    print("    [OK] demo6_crossover.pdf")


def fig_scaling_law(data):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))

    tempos = np.array(data['tempos'])
    alphas = {}

    for trad in TRADITIONS:
        costs = np.array(data[trad])
        color = COLORS.get(trad, '#888888')
        mask = np.isfinite(costs) & (costs > 0) & (tempos > 0)
        if mask.sum() >= 3:
            log_f = np.log(tempos[mask])
            log_S = np.log(costs[mask])
            coeffs = np.polyfit(log_f, log_S, 1)
            alpha = coeffs[0]
            alphas[trad] = alpha
            fit_S = np.exp(np.polyval(coeffs, np.log(tempos[mask])))
            ax.loglog(tempos[mask], costs[mask], marker='.',
                      color=color, linestyle='none', markersize=3,
                      label=f'{TRADITIONS[trad].name} '
                            f'($\\alpha$={alpha:.2f})')
            ax.loglog(tempos[mask], fit_S, color=color,
                      linestyle='-', alpha=0.5, linewidth=0.7)

    ax.set_xlabel('Tempo $f$ (notes/s)')
    ax.set_ylabel('Total action $S$')
    ax.legend(frameon=False, fontsize=5.5, loc='upper left')

    plt.tight_layout()
    fig.savefig('scaling_law.pdf')
    plt.close(fig)
    print("    [OK] scaling_law.pdf")
    print("         Scaling exponents:")
    for trad, alpha in sorted(alphas.items(), key=lambda x: -x[1]):
        print(f"           {trad:12s}: alpha = {alpha:.3f}")


def fig_stochastic(data, midis):
    plt = setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL, 3.5),
                                    gridspec_kw={'height_ratios': [1, 1.2]})

    robustness = data['robustness']
    det = data['deterministic']
    det_fingers = det['fingers'] if isinstance(det, dict) else det.fingers
    N = len(robustness)
    x = range(N)

    colors_r = ['#28A745' if r >= 0.7 else '#FFC107' if r >= 0.4
                else '#DC3545' for r in robustness]
    ax1.bar(x, robustness, color=colors_r, edgecolor='none', width=0.8)
    ax1.axhline(0.7, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax1.set_ylabel('Robustness')
    ax1.set_title(f'(a) Per-note robustness ({data["trials"]} trials, '
                  f'$\\sigma$={data["sigma"]})')
    ax1.set_ylim(0, 1.05)

    dists = np.array(data['finger_distributions'])
    im = ax2.imshow(dists.T, aspect='auto', cmap='YlOrRd',
                    origin='lower', interpolation='nearest')
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(['1', '2', '3', '4', '5'])
    ax2.set_ylabel('Finger')
    ax2.set_xlabel('Note index')
    ax2.set_title('(b) Finger probability distribution')
    for i, f in enumerate(det_fingers):
        if i < N:
            ax2.plot(i, f - 1, 'k+', markersize=5, markeredgewidth=0.8)
    fig.colorbar(im, ax=ax2, label='Probability', shrink=0.8)

    plt.tight_layout()
    fig.savefig('stochastic_robustness.pdf')
    plt.close(fig)
    print("    [OK] stochastic_robustness.pdf")


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 65)
    print("ValHaLA v2.1 — Demonstration Figure Generator")
    print("14-term equation of state for piano fingering")
    print("=" * 65)

    print("\n  [D1] Baroque vs. Modern...")
    d1, m1 = run_demo1()
    for t in ['baroque', 'modern']:
        f = '-'.join(str(x) for x in d1[t]['fingers'][:12])
        print(f"    {t.capitalize():12s}: J={d1[t]['cost']:.1f}  "
              f"F=[{f}...]")
    fig_demo1(d1, m1)

    print("\n  [D2] Large vs. small hands...")
    d2, m2 = run_demo2()
    for l in ['large', 'small']:
        print(f"    {l.capitalize():12s}: J={d2[l]['cost']:.1f}")
    fig_demo2(d2, m2)

    print("\n  [D3] Russian vs. French-Cortot...")
    d3, m3 = run_demo3()
    for t in ['russian', 'french']:
        print(f"    {TRADITIONS[t].name:12s}: J={d3[t]['cost']:.1f}")
    fig_demo3(d3, m3)

    print("\n  [D4] Tempo dependence...")
    d4, m4 = run_demo4()
    for label in ['adagio', 'allegro', 'presto']:
        f = '-'.join(str(x) for x in d4[label]['fingers'])
        print(f"    {label.capitalize():12s}: J={d4[label]['cost']:.1f}  "
              f"F=[{f}]")
    fig_demo4(d4, m4)

    print("\n  [D5] Dry vs. moist...")
    d5, m5 = run_demo5()
    for l in ['dry', 'moist']:
        print(f"    {l.capitalize():12s}: J={d5[l]['cost']:.1f}  "
              f"ToB={d5[l]['thumb_on_black']}")
    fig_demo5(d5, m5)

    print("\n  [D6] Baroque-Modern crossover...")
    d6, m6 = run_demo6()
    fig_demo6(d6)

    print("\n  [F7] Scaling law (all 8 traditions)...")
    scaling = run_scaling_law()
    fig_scaling_law(scaling)

    print("\n  [F8] Stochastic robustness...")
    stoch, ms = run_stochastic()
    fig_stochastic(stoch, ms)

    import glob
    figs = sorted(glob.glob('*.pdf'))
    print(f"\n{'=' * 65}")
    print(f"  COMPLETE: {len(figs)} PDF figures generated.")
    for f in figs:
        print(f"    {f}")
    print(f"{'=' * 65}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
