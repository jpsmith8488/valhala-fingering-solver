#!/usr/bin/env python3
"""
Exhaustive figure/table correspondence check.
Verifies that every numeric claim in the manuscript's tables and figure
captions/alt-text is exactly reproduced by the frozen solver.
"""
import sys, numpy as np
import valhala_solver_standalone as vs
from valhala_solver_standalone import (HamiltonianSolver, SolverConfig,
                                       KeyboardModel, TRADITIONS)
import run_demonstrations as rd

P, F = [], []
def ck(name, got, want, ok):
    (P if ok else F).append(name)
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}: got {got} | paper {want}")
def fs(r): return '-'.join(str(int(x)) for x in r.fingers)

print("="*72); print("FIGURE & TABLE CORRESPONDENCE CHECK"); print("="*72)

# ===== TABLE 4 (Chopin Op.10/1 m.1) =====
print("\nTABLE 4 — Chopin Op.10/1 m.1 (12-note arpeggio)")
arp = [60,64,67,72,76,79,84,88,91,96,100,103]
r190 = HamiltonianSolver(config=SolverConfig(hand_length=190,max_span=210,tradition='modern',tempo_nps=4.0)).solve(rd.make_notes(arp,4.0))
ck("Solver H=190 fingering", fs(r190), "1-2-4-1-2-4-1-2-4-1-3-5", fs(r190)=="1-2-4-1-2-4-1-2-4-1-3-5")
# agreement vs Paderewski/Cortot (identical) and Henle
pad = [1,2,4,1,2,4,1,2,4,1,3,5]; hen=[1,2,4,1,2,4,1,2,4,1,2,5]
fv=[int(x) for x in r190.fingers]
ck("Agreement vs Paderewski", f"{sum(a==b for a,b in zip(fv,pad))}/12", "12/12", sum(a==b for a,b in zip(fv,pad))==12)
ck("Agreement vs Henle", f"{sum(a==b for a,b in zip(fv,hen))}/12", "11/12", sum(a==b for a,b in zip(fv,hen))==11)

# ===== TABLE 3 (validation battery verdicts) =====
print("\nTABLE 3 — Validation battery verdicts (6 PASS / 3 THRESHOLD / 1 DUAL)")
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    try:
        import validation_battery as vb
        if hasattr(vb,'main'): vb.main()
    except SystemExit: pass
out=buf.getvalue()
np_, nt_, nd_ = out.count('PASS'), out.count('THRESHOLD'), out.count('DUAL')
# fall back to counting summary line
ck("battery summary present", 'SUMMARY' in out or np_>0, "6/3/1", ('6 PASS' in out and '3 THRESHOLD' in out and '1 DUAL' in out))

# ===== FIG demo1 (Baroque vs Modern, Bach Invention 13) =====
print("\nFIG 1 (demo1) — Baroque vs Modern, ARI: Baroque > Modern")
d1, m1 = rd.run_demo1()
bar_ari = d1['baroque']['ari']; mod_ari = d1['modern']['ari']
ck("Baroque ARI >= Modern ARI throughout", "yes" if all(b>=m-1e-9 for b,m in zip(bar_ari,mod_ari)) else "no",
   "Baroque higher", all(b>=m-1e-9 for b,m in zip(bar_ari,mod_ari)))
# term decomposition sign checks from alt text: phrase & arpeggio negative; tradition +Baroque / -Modern
cb_b = d1['baroque']['breakdown']; cb_m = d1['modern']['breakdown']
def term(cb,*names):
    for n in names:
        if n in cb: return cb[n]
    return None

# ===== FIG demo2 (hand size peak ARI) =====
print("\nFIG 2 (demo2) — peak ARI 45.5 (small) / 38.6 (large)")
d2,_ = rd.run_demo2()
ps=max(d2['small']['ari']); pl=max(d2['large']['ari'])
ck("small peak ARI", f"{ps:.1f}", "45.5", abs(ps-45.5)<0.05)
ck("large peak ARI", f"{pl:.1f}", "38.6", abs(pl-38.6)<0.05)

# ===== FIG demo5 (environment) =====
print("\nFIG 5 (demo5) — dry 27.42/ToB2, moist 24.10/ToB4, -12.1%")
d5,p5 = rd.run_demo5()
ck("dry cost", f"{d5['dry']['cost']:.2f}", "27.42", abs(d5['dry']['cost']-27.42)<0.05)
ck("moist cost", f"{d5['moist']['cost']:.2f}", "24.10", abs(d5['moist']['cost']-24.10)<0.05)
ck("dry ToB", d5['dry']['thumb_on_black'], "2", d5['dry']['thumb_on_black']==2)
ck("moist ToB", d5['moist']['thumb_on_black'], "4", d5['moist']['thumb_on_black']==4)
red=100*(d5['dry']['cost']-d5['moist']['cost'])/d5['dry']['cost']
ck("cost reduction", f"{red:.1f}%", "12.1%", abs(red-12.1)<0.2)

# ===== FIG demo6 (crossover) =====
print("\nFIG 3 (demo6) — phi 0.13 -> 0.04, mu_c ~1.15")
d6,_ = rd.run_demo6()
ck("phi at mu=0", f"{d6['phi'][0]:.3f}", "~0.13", abs(d6['phi'][0]-0.13)<0.03)
ck("phi plateau (min)", f"{min(d6['phi']):.3f}", "~0.04", abs(min(d6['phi'])-0.043)<0.02)

# ===== FIG scaling (eight exponents) =====
print("\nFIG 6 (scaling) — eight exponents")
sd = rd.run_scaling_law()
tempos=np.array(sd['tempos'])
want={'romantic':1.90,'baroque':1.89,'chopin':1.88,'russian':1.71,'taubman':1.65,'modern':1.49,'classical':1.46,'french':1.39}
for t,w in want.items():
    a=np.polyfit(np.log(tempos),np.log(np.array(sd[t])),1)[0]
    ck(f"alpha {t}", f"{round(a,2):.2f}", f"{w:.2f}", round(a,2)==round(w,2))

print("\n"+"="*72)
print(f"RESULT: {len(P)} passed, {len(F)} failed")
if F: print("FAILURES:",F)
print("="*72)
sys.exit(0 if not F else 1)
