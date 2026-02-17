from src.prhp_core import simulate_prhp

results = simulate_prhp(levels=9, variants=['ADHD-collectivist', 'autistic-individualist'], n_monte=100, seed=42)
for v, r in results.items():
    print(f"{v}: Fidelity = {r['mean_fidelity']:.3f} ± {r['std']:.3f}")

