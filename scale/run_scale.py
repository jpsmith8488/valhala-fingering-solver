#!/usr/bin/env python3
"""
run_scale.py — SCALE Analysis Runner (Standalone)
===================================================
Computes the Spectral Chromatic Action Landscape Evaluation for both
books of Bach's Well-Tempered Clavier.

Usage:
    python run_scale.py \
        --solver-path ../valhala_solver_standalone.py \
        --book1 path/to/wtc_book1.mxl \
        --book2 path/to/wtc_book2.mxl \
        --output-dir results/

Requires: numpy, scipy, music21, matplotlib
"""

import sys, os, json, time, argparse
import importlib.util
import numpy as np

# ── BWV Catalog ────────────────────────────────────────────────
WTC_BWV = {
    1: [("C major",846),("C minor",847),("C# major",848),("C# minor",849),
        ("D major",850),("D minor",851),("Eb major",852),("Eb minor",853),
        ("E major",854),("E minor",855),("F major",856),("F minor",857),
        ("F# major",858),("F# minor",859),("G major",860),("G minor",861),
        ("Ab major",862),("G# minor",863),("A major",864),("A minor",865),
        ("Bb major",866),("Bb minor",867),("B major",868),("B minor",869)],
    2: [("C major",870),("C minor",871),("C# major",872),("C# minor",873),
        ("D major",874),("D minor",875),("Eb major",876),("Eb minor",877),
        ("E major",878),("E minor",879),("F major",880),("F minor",881),
        ("F# major",882),("F# minor",883),("G major",884),("G minor",885),
        ("Ab major",886),("G# minor",887),("A major",888),("A minor",889),
        ("Bb major",890),("Bb minor",891),("B major",892),("B minor",893)]
}

KEY_ACC = {
    "C major":0,"A minor":0,"G major":1,"E minor":1,"F major":1,"D minor":1,
    "D major":2,"B minor":2,"Bb major":2,"G minor":2,"A major":3,"F# minor":3,
    "Eb major":3,"C minor":3,"E major":4,"C# minor":4,"Ab major":4,"F minor":4,
    "B major":5,"G# minor":5,"Bb minor":5,"F# major":6,"Eb minor":6,"C# major":7,
}


def load_solver(solver_path):
    """Import HamiltonianSolver from file path."""
    spec = importlib.util.spec_from_file_location("solver_module", solver_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HamiltonianSolver


def extract_and_pair(mxl_path):
    """Extract pieces from MXL, return 48 segments paired into 24 units."""
    import music21

    score = music21.converter.parse(str(mxl_path))
    parts = list(score.parts)
    if len(parts) < 2:
        parts = [parts[0], parts[0]]

    measures = list(parts[0].getElementsByClass('Measure'))
    total_measures = len(measures)

    # Find piece boundaries at final barlines
    boundaries = [0]
    for i, m in enumerate(measures):
        for barline in m.getElementsByClass('Barline'):
            if hasattr(barline, 'type') and barline.type == 'final':
                if i > 0:
                    boundaries.append(i + 1)
    boundaries.append(total_measures)

    # Filter: minimum 8-measure gap
    filtered = [boundaries[0]]
    for b in boundaries[1:]:
        if b - filtered[-1] >= 8:
            filtered.append(b)
    if filtered[-1] != total_measures:
        filtered.append(total_measures)

    # Extract segments
    segments = []
    for seg_idx in range(len(filtered) - 1):
        start_m, end_m = filtered[seg_idx], filtered[seg_idx + 1]
        if end_m - start_m < 4:
            continue

        rh_pitches, lh_pitches = [], []
        for part_idx, pitch_list in [(0, rh_pitches),
                                      (min(1, len(parts)-1), lh_pitches)]:
            part_measures = list(parts[part_idx].getElementsByClass('Measure'))
            for mi in range(start_m, min(end_m, len(part_measures))):
                for n in part_measures[mi].recurse().notes:
                    if hasattr(n, 'pitch'):
                        pitch_list.append(n.pitch.midi)
                    elif hasattr(n, 'pitches'):
                        for p in n.pitches:
                            pitch_list.append(p.midi)

        segments.append({
            'rh_pitches': rh_pitches,
            'lh_pitches': lh_pitches,
            'note_count': len(rh_pitches) + len(lh_pitches),
            'segment_index': seg_idx,
        })

    # Keep longest 48, re-sort by position, pair consecutively
    segments.sort(key=lambda s: s['note_count'], reverse=True)
    kept = segments[:48]
    kept.sort(key=lambda s: s['segment_index'])
    return kept


def run_scale_analysis(pairs, bwv_catalog, solver, output_dir):
    """Run SCALE on 24 paired units."""
    results = {
        'metadata': {
            'solver': 'HamiltonianSolver v2.1',
            'tradition': 'Modern',
            'hand_length': 190,
            'n_keys': 24,
        },
        'pieces': [],
    }

    total_solves = 0
    t0 = time.time()

    for pair_idx, (key_name, bwv) in enumerate(bwv_catalog):
        seg_a = pairs[pair_idx * 2] if pair_idx * 2 < len(pairs) else None
        seg_b = pairs[pair_idx * 2 + 1] if pair_idx * 2 + 1 < len(pairs) else None

        rh = (seg_a['rh_pitches'] if seg_a else []) + (seg_b['rh_pitches'] if seg_b else [])
        lh = (seg_a['lh_pitches'] if seg_a else []) + (seg_b['lh_pitches'] if seg_b else [])

        cost_total, cost_rh, cost_lh = [], [], []
        breakdown_total = []

        for offset in range(12):
            rh_t = [max(21, min(108, p + offset)) for p in rh]
            lh_t = [max(21, min(108, p + offset)) for p in lh]

            rh_cost, rh_bd = 0.0, {}
            if len(rh_t) >= 2:
                r = solver.solve(rh_t)
                rh_cost = float(r['cost'])
                rh_bd = {k: float(v) for k, v in r.get('cost_breakdown', {}).items()}
                total_solves += 1

            lh_cost, lh_bd = 0.0, {}
            if len(lh_t) >= 2:
                r = solver.solve(lh_t)
                lh_cost = float(r['cost'])
                lh_bd = {k: float(v) for k, v in r.get('cost_breakdown', {}).items()}
                total_solves += 1

            total = rh_cost + lh_cost
            combined_bd = {}
            for t in set(list(rh_bd.keys()) + list(lh_bd.keys())):
                combined_bd[t] = rh_bd.get(t, 0.0) + lh_bd.get(t, 0.0)

            cost_total.append(total)
            cost_rh.append(rh_cost)
            cost_lh.append(lh_cost)
            breakdown_total.append(combined_bd)

        ranked = np.argsort(cost_total)
        ranks = np.zeros(12, dtype=int)
        for rv, idx in enumerate(ranked):
            ranks[idx] = rv + 1

        s_min, s_max = min(cost_total), max(cost_total)
        spread = (s_max - s_min) / s_min if s_min > 0 else 0

        acc = KEY_ACC.get(key_name, 0)
        display = key_name.replace('#', '\u266f').replace('b ', '\u266d ')

        results['pieces'].append({
            'key': key_name, 'bwv': bwv, 'display_name': display,
            'accidentals': acc,
            'note_count': len(rh) + len(lh),
            'cost_total': cost_total, 'cost_rh': cost_rh, 'cost_lh': cost_lh,
            'breakdown_total': breakdown_total,
            'ranks': ranks.tolist(),
            'original_rank': int(ranks[0]),
            'optimal_key': ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"][int(ranked[0])],
            'optimal_idx': int(ranked[0]),
            'spread': spread,
        })

        print(f"  [{pair_idx+1:2d}/24] BWV {bwv} {display}: "
              f"rank {int(ranks[0])}, spread {spread*100:.1f}%")

    elapsed = time.time() - t0
    results['metadata']['total_solves'] = total_solves
    results['metadata']['elapsed_seconds'] = elapsed

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scale_data.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  {total_solves} solver invocations in {elapsed:.1f}s")
    print(f"  Saved: {output_dir}/scale_data.json")
    return results


def main():
    parser = argparse.ArgumentParser(description="SCALE analysis of the WTC")
    parser.add_argument("--solver-path", required=True,
                        help="Path to valhala_solver_standalone.py")
    parser.add_argument("--book1", help="Path to WTC Book 1 MXL file")
    parser.add_argument("--book2", help="Path to WTC Book 2 MXL file")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    SolverClass = load_solver(args.solver_path)
    solver = SolverClass(tradition="modern", hand_length=190)

    # Quick verification
    test = solver.solve([60, 62, 64, 65, 67, 69, 71, 72])
    assert test['fingers'] == [1, 2, 3, 1, 2, 3, 4, 5], \
        f"Solver verification failed: {test['fingers']}"
    print("Solver verified.\n")

    for bn, path, catalog in [
        (1, args.book1, WTC_BWV[1]),
        (2, args.book2, WTC_BWV[2]),
    ]:
        if not path:
            print(f"Book {bn}: skipped (no path provided)")
            continue
        if not os.path.exists(path):
            print(f"Book {bn}: file not found at {path}")
            continue

        print(f"Book {bn}: {os.path.basename(path)}")
        pairs = extract_and_pair(path)
        print(f"  Extracted {len(pairs)} segments, using 48")
        out = os.path.join(args.output_dir, f"Book_{bn}")
        run_scale_analysis(pairs, catalog, solver, out)
        print()


if __name__ == "__main__":
    main()
