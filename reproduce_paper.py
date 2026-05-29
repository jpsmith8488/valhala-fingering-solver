#!/usr/bin/env python3
"""
End-to-end reproduction check for the ValHaLA fingering paper.
Runs every headline quantity through the frozen solver and compares
against the values stated in the manuscript. Exit 0 iff all pass.

Frozen solver MD5: 846ce6aae16623c6ca4a551f86df869c
Methods mirror run_demonstrations.py (the canonical figure generator).
"""
import sys
import numpy as np
import valhala_solver_standalone as vs
from valhala_solver_standalone import (HamiltonianSolver, SolverConfig,
                                       KeyboardModel)
import run_demonstrations as rd

PASS, FAIL = [], []
def check(name, got, want, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}: got {got}, paper says {want}")
def fstr(r): return '-'.join(str(int(x)) for x in r.fingers)

print("="*70)
print("ValHaLA end-to-end reproduction check (frozen MD5 846ce6a)")
print("="*70)

# ---- D1: emergent scale fingerings (one octave) ----
print("\n[D1] Emergent scale fingerings (C4-C5)")
scale1 = [60,62,64,65,67,69,71,72]
mod = HamiltonianSolver(config=SolverConfig(tradition='modern',tempo_nps=4.0)).solve(rd.make_notes(scale1,4.0))
bar = HamiltonianSolver(config=SolverConfig(tradition='baroque',tempo_nps=4.0)).solve(rd.make_notes(scale1,4.0))
check("Modern scale", fstr(mod), "1-2-3-1-2-3-4-5", fstr(mod)=="1-2-3-1-2-3-4-5")
check("Baroque scale", fstr(bar), "2-3-2-3-2-3-4-5", fstr(bar)=="2-3-2-3-2-3-4-5")

# ---- Table 4: Chopin Op.10/1 m.1, 12-note ascending C-major arpeggio ----
print("\n[Table 4] Chopin Op.10/1 m.1 (12-note arpeggio C4 up), target = Paderewski/Cortot")
arp12 = [60,64,67,72,76,79,84,88,91,96,100,103]
target10 = "1-2-4-1-2-4-1-2-4-1-3-5"
for H,Dmax in [(170,180),(190,210),(210,240)]:
    r = HamiltonianSolver(config=SolverConfig(hand_length=H,max_span=Dmax,tradition='modern',tempo_nps=4.0)).solve(rd.make_notes(arp12,4.0))
    check(f"Chopin Op10/1 H={H}", fstr(r), target10, fstr(r)==target10)

# ---- D2: hand-size peak ARI (uses the arch passage, per run_demo2) ----
print("\n[D2] Hand-size peak ARI (Op.10/1 arch, 4 nps) — mirrors run_demo2")
op10arch = vs.passage_chopin_op10_1()
small = HamiltonianSolver(config=SolverConfig(hand_length=170,max_span=180,tradition='modern',tempo_nps=4.0)).solve(rd.make_notes(op10arch,4.0))
large = HamiltonianSolver(config=SolverConfig(hand_length=210,max_span=240,tradition='modern',tempo_nps=4.0)).solve(rd.make_notes(op10arch,4.0))
ps, pl = max(small.ari_values), max(large.ari_values)
check("small peak ARI", f"{ps:.2f}", "45.5", abs(ps-45.52)<0.05)
check("large peak ARI", f"{pl:.2f}", "38.6", abs(pl-38.60)<0.05)
inc = 100*(small.total_cost-large.total_cost)/large.total_cost
check("total-action increase", f"{inc:.1f}%", "23%", abs(inc-23.2)<1.0)

# ---- D5: Chopin Op.25/6 dry vs moist ----
print("\n[D5] Chopin Op.25/6 dry vs moist")
op25 = vs.passage_chopin_op25_6()
dry = HamiltonianSolver(config=SolverConfig(tradition='modern',tempo_nps=6.0,mu_f=0.5)).solve(rd.make_notes(op25,6.0))
moist = HamiltonianSolver(config=SolverConfig(tradition='modern',tempo_nps=6.0,mu_f=1.0)).solve(rd.make_notes(op25,6.0))
def tob(r): return sum(1 for f,m in zip(r.fingers,op25) if int(f)==1 and KeyboardModel.is_black(m))
check("dry cost", f"{dry.total_cost:.2f}", "27.42", abs(dry.total_cost-27.42)<0.05)
check("dry ToB", tob(dry), "2", tob(dry)==2)
check("moist cost", f"{moist.total_cost:.2f}", "24.10", abs(moist.total_cost-24.10)<0.05)
check("moist ToB", tob(moist), "4", tob(moist)==4)
red = 100*(dry.total_cost-moist.total_cost)/dry.total_cost
check("cost reduction", f"{red:.1f}%", "12.1%", abs(red-12.1)<0.2)

# ---- D7: scaling-law exponents ----
print("\n[D7] Scaling-law exponents S(f)=J+T, OLS log-log fit, f in [2,18], 17 pts")
exps = {'romantic':1.90,'baroque':1.89,'chopin':1.88,'russian':1.71,
        'taubman':1.65,'modern':1.49,'classical':1.46,'french':1.39}
scale2 = vs.passage_c_major_scale()
fs = np.linspace(2,18,17)
for trad,want in exps.items():
    solver = HamiltonianSolver(config=SolverConfig(tradition=trad,tempo_nps=6.0))
    S = [vs.total_action(solver, scale2, f) for f in fs]
    slope = np.polyfit(np.log(fs), np.log(S), 1)[0]
    check(f"alpha {trad}", f"{slope:.2f}", f"{want:.2f}", abs(slope-want)<0.015)

# ---- D3: Baroque-Modern crossover (mirror run_demo6 exactly) ----
print("\n[D3] Baroque-Modern crossover phi(mu), mu_c — mirrors run_demo6")
inv = vs.passage_bach_invention_13()
mu_values = np.linspace(0.0, 2.5, 51)
phis=[]
for mu in mu_values:
    solver = HamiltonianSolver(config=SolverConfig(tradition='modern', tempo_nps=6.0))
    solver.tradition.thumb_under = 3 + mu*4
    solver.tradition.thumb_on_black = 2 + mu*20
    solver.tradition.metric_weight = mu
    r = solver.solve(rd.make_notes(inv,6.0))
    f = r.fingers
    tu = sum(1 for i in range(1,len(f)) if f[i-1]!=1 and f[i]==1 and inv[i]>inv[i-1])
    phis.append(tu/max(len(f)-1,1))
phi_lo = phis[0]
phi_hi = min(phis)  # plateau after crossover
dphis = [phis[i]-phis[i-1] for i in range(1,len(phis))]
mu_c = mu_values[int(np.argmin(dphis))+1]
check("phi at mu=0", f"{phi_lo:.3f}", "~0.13", abs(phi_lo-0.13)<0.03)
check("phi plateau", f"{phi_hi:.3f}", "~0.04", abs(phi_hi-0.043)<0.02)
check("mu_c (largest drop)", f"{mu_c:.2f}", "~1.20", abs(mu_c-1.20)<0.15)

print("\n"+"="*70)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL: print("FAILURES:", FAIL)
print("="*70)
sys.exit(0 if not FAIL else 1)
