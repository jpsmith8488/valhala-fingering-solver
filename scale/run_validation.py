#!/usr/bin/env python3
"""
run_validation.py — SCALE Synthetic Validation Battery
=======================================================
Four tests with synthetic passages to verify that SCALE produces
expected outputs on inputs with known biomechanical properties.

Usage:
    python run_validation.py --solver-path ../valhala_solver_standalone.py
"""

import sys, os, json, argparse
import importlib.util

KEY_NAMES = ["C", "C#/Db", "D", "Eb", "E", "F",
              "F#/Gb", "G", "Ab", "A", "Bb", "B"]
KEY_ACC = {0:0, 1:7, 2:2, 3:3, 4:4, 5:1, 6:6, 7:1, 8:4, 9:3, 10:2, 11:5}


def load_solver(solver_path):
    """Import HamiltonianSolver from the specified file path."""
    spec = importlib.util.spec_from_file_location("solver_module", solver_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HamiltonianSolver


def run_scale(pitches, solver):
    """Run SCALE on a pitch sequence; return 12 costs."""
    costs = []
    for offset in range(12):
        transposed = [max(21, min(108, p + offset)) for p in pitches]
        result = solver.solve(transposed)
        costs.append(float(result['cost']))
    return costs


def main():
    parser = argparse.ArgumentParser(
        description="SCALE synthetic validation battery")
    parser.add_argument("--solver-path", required=True,
                        help="Path to valhala_solver_standalone.py")
    args = parser.parse_args()

    SolverClass = load_solver(args.solver_path)
    solver = SolverClass(tradition="modern", hand_length=190)

    results = {}
    all_pass = True

    # ── Test 1: White-key C major scale ──
    print("\n" + "="*60)
    print("  Test 1: C major scale, two octaves (all white keys)")
    print("="*60)
    pitches1 = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84]
    costs1 = run_scale(pitches1, solver)
    ranked1 = sorted(range(12), key=lambda i: costs1[i])
    for i in range(12):
        rank = ranked1.index(i) + 1
        marker = " *" if rank == 1 else ""
        print(f"  {KEY_NAMES[i]:<8s}  {KEY_ACC[i]:>1d} acc  cost {costs1[i]:>8.2f}  rank {rank:>2d}{marker}")
    top4 = [KEY_ACC[ranked1[i]] for i in range(4)]
    bot4 = [KEY_ACC[ranked1[i]] for i in range(8, 12)]
    spread1 = (max(costs1) - min(costs1)) / min(costs1)
    t1_pass = (sum(top4)/4 < sum(bot4)/4) and (spread1 > 0.5)
    print(f"\n  Top-4 mean acc: {sum(top4)/4:.1f}, Bot-4 mean acc: {sum(bot4)/4:.1f}")
    print(f"  Spread: {spread1*100:.1f}%")
    print(f"  VERDICT: {'PASS' if t1_pass else 'FAIL'}")
    if not t1_pass: all_pass = False

    # ── Test 2: Black-key F# pentatonic ──
    print("\n" + "="*60)
    print("  Test 2: F# pentatonic, two octaves (all black keys)")
    print("="*60)
    pitches2 = [66, 68, 70, 73, 75, 78, 80, 82, 85, 87, 90, 92, 94]
    costs2 = run_scale(pitches2, solver)
    ranked2 = sorted(range(12), key=lambda i: costs2[i])
    for i in range(12):
        rank = ranked2.index(i) + 1
        marker = " *" if rank == 1 else ""
        print(f"  {KEY_NAMES[i]:<8s}  {KEY_ACC[i]:>1d} acc  cost {costs2[i]:>8.2f}  rank {rank:>2d}{marker}")
    orig_rank = ranked2.index(0) + 1
    spread2 = (max(costs2) - min(costs2)) / min(costs2)
    t2_pass = (orig_rank >= 9) and (costs2[0] > costs2[ranked2[0]])
    print(f"\n  Original (offset 0) rank: {orig_rank}/12")
    print(f"  Spread: {spread2*100:.1f}%")
    print(f"  VERDICT: {'PASS' if t2_pass else 'FAIL'}")
    if not t2_pass: all_pass = False

    # ── Test 3: Single repeated note ──
    print("\n" + "="*60)
    print("  Test 3: Single repeated note C5 (isolates per-note cost)")
    print("="*60)
    pitches3 = [72] * 12
    costs3 = run_scale(pitches3, solver)
    ranked3 = sorted(range(12), key=lambda i: costs3[i])
    white_costs = [costs3[i] for i in range(12) if i not in {1, 3, 6, 8, 10}]
    black_costs = [costs3[i] for i in range(12) if i in {1, 3, 6, 8, 10}]
    for i in range(12):
        rank = ranked3.index(i) + 1
        bw = "black" if i in {1, 3, 6, 8, 10} else "white"
        marker = " *" if rank == 1 else ""
        print(f"  {KEY_NAMES[i]:<8s}  {bw:<5s}  cost {costs3[i]:>8.2f}  rank {rank:>2d}{marker}")
    no_overlap = max(white_costs) < min(black_costs)
    print(f"\n  White key costs: {min(white_costs):.2f} – {max(white_costs):.2f}")
    print(f"  Black key costs: {min(black_costs):.2f} – {max(black_costs):.2f}")
    print(f"  Binary separation (no overlap): {no_overlap}")
    t3_pass = no_overlap
    print(f"  VERDICT: {'PASS' if t3_pass else 'FAIL'} — "
          f"{'perfect' if no_overlap else 'imperfect'} black/white binary split")
    if not t3_pass: all_pass = False

    # ── Test 4: Chromatic scale ──
    print("\n" + "="*60)
    print("  Test 4: Chromatic scale, two octaves")
    print("="*60)
    pitches4 = list(range(60, 84))
    costs4 = run_scale(pitches4, solver)
    ranked4 = sorted(range(12), key=lambda i: costs4[i])
    for i in range(12):
        rank = ranked4.index(i) + 1
        marker = " *" if rank == 1 else ""
        print(f"  {KEY_NAMES[i]:<8s}  {KEY_ACC[i]:>1d} acc  cost {costs4[i]:>8.2f}  rank {rank:>2d}{marker}")
    spread4 = (max(costs4) - min(costs4)) / min(costs4)
    t4_pass = 0.05 < spread4 < 0.30
    print(f"\n  Spread: {spread4*100:.1f}% (expected 5–30%)")
    print(f"  VERDICT: {'PASS' if t4_pass else 'FAIL'}")
    if not t4_pass: all_pass = False

    # ── Summary ──
    print("\n" + "="*60)
    print(f"  SUMMARY: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print(f"    Test 1 (white-key scale):     {'PASS' if t1_pass else 'FAIL'}")
    print(f"    Test 2 (black-key pentatonic): {'PASS' if t2_pass else 'FAIL'}")
    print(f"    Test 3 (repeated note):        {'PASS' if t3_pass else 'FAIL'}")
    print(f"    Test 4 (chromatic scale):       {'PASS' if t4_pass else 'FAIL'}")
    print("="*60)


if __name__ == "__main__":
    main()
