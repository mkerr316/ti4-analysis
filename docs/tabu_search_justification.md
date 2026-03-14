# Tabu Search Inclusion as Methodological Control

**For insertion into the paper's Methodology / Algorithm Selection section.**

---

## Why Tabu Search is included in the benchmark

Grichshenko et al. (2020) demonstrated that Tabu Search (TS) outperforms steepest-ascent hill-climbing for map generation in *Terra Mystica*, a related tabletop game with a hexagonal spatial layout. TS escapes local optima via a deterministic tabu list that prevents the algorithm from revisiting recently explored configurations, enabling non-improving moves that a greedy hill-climber would reject.

Our SA implementation addresses the same local-optima escape mechanism through a complementary stochastic pathway: the Metropolis acceptance criterion `P(accept) = exp(-Δ/T)` probabilistically accepts worsening moves at a rate controlled by temperature. With `T₀` calibrated to an 80% initial acceptance rate (Section X.Y), SA permits approximately four out of five uphill moves during early exploration — functionally analogous to TS's deterministic acceptance of non-improving moves, but without requiring tuning of tabu tenure or neighbourhood structure.

### Rationale for inclusion as a methodological control

Including TS in the benchmark addresses the following research question: **Is SA's stochastic escape mechanism redundant with TS's deterministic memory, or do the two approaches find qualitatively different solutions?**

The Terra Mystica PCG literature (Grichshenko et al., 2020) demonstrated TS superiority over steepest-ascent hill-climbing, but this comparison is not informative about SA because steepest-ascent HC cannot escape *any* local optimum and serves as a weaker baseline than SA. No published evidence demonstrates that TS produces superior map configurations to SA on combinatorial tile-placement problems when both algorithms receive the same evaluation budget.

By including TS with a grid-tuned tenure coefficient k (θ = max(3, ⌈k·√S⌉), tuned on the same disjoint seed range as SA and NSGA-II), the benchmark produces one of three outcomes:

1. **TS ≈ SA at convergence:** SA's stochastic escape is functionally equivalent to TS's deterministic memory for this problem class. SA's lower per-iteration cost (1 evaluation vs C(S,2)) makes it the preferred instrument for the main experiment.

2. **TS > SA at high budgets:** TS's exhaustive neighbourhood scan uncovers optima that SA's random walk misses. This would justify TS as the preferred instrument despite higher per-step cost, or motivate a hybrid SA-TS approach.

3. **SA > TS at all budgets:** SA's trajectory depth advantage (1,000 sequential steps vs TS's ≈ 2 full iterations at budget 1,000) is essential for this landscape's ruggedness. TS's breadth-per-step does not compensate for its shallow decision history.

The multi-budget saturation study (1k–500k evaluations) captures the crossover point — if one exists — where TS's per-decision superiority overcomes SA's trajectory depth advantage.

### Evaluation budget accounting

Each candidate 2-swap costs one evaluation. A single TS iteration evaluates all C(S,2) ≈ 435 pairs (for S = 30 swappable positions), making TS the evaluation-heaviest algorithm per iteration. At budget 1,000, TS completes only ≈ 2 full iterations versus SA's 1,000 sequential steps — the depth-vs-breadth tradeoff that the saturation study characterises.

### References

- Grichshenko, A., Lankveld, G. van, & Spronck, P. (2020). Using Tabu Search Algorithm for Map Generation in the Terra Mystica Tabletop Game. *Proceedings of the 4th International Conference on Intelligent Systems, Metaheuristics & Swarm Intelligence (ISMSI)*, 66–70. ACM.
- Glover, F. (1989). Tabu search — Part I. *ORSA Journal on Computing*, 1(3), 190–206.
- Glover, F. (1990). Tabu search — Part II. *ORSA Journal on Computing*, 2(1), 4–32.
