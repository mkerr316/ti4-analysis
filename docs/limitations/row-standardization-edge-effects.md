# Row-Standardization and Edge-Tile Leverage

## The flaw

Row-standardizing the spatial weight matrix **W** ensures edge systems aren't artificially penalized: without it, outer-ring systems (2–3 neighbours) would have systematically smaller spatial lags than central systems (up to 6 neighbours), deflating their contribution to Moran's I and LISA.

However, row-standardization has a flip side: **each neighbour's weight is inversely proportional to neighbour count**. So:

- An **edge tile** with 2 neighbours gives each neighbour weight **50%** of its spatial lag.
- A **central tile** with 6 neighbours gives each neighbour **~16.7%**.

So the *influence of a single neighbour* on the spatial lag (and thus on local Moran \(I_i\) and LSAP) is much larger at the boundary. Edge tiles are more sensitive to each adjacent system; one “bad” or “good” neighbour dominates the local picture.

---

## Is it a limitation or a feature?

**As limitation:** Local Moran (and LSAP) at edge positions is more sensitive to each individual neighbour than at central positions. The statistic behaves differently at the boundary.

**As substantive feature:** In TI4, if you are on the edge of the galaxy, the few systems adjacent to you *do* dominate your strategic position — movement, conflict, and resource access are determined by those neighbours. Giving each of them a larger weight in the spatial lag is consistent with that reality.

So: it is a **methodological sensitivity** at the boundary, but one that **aligns with the strategic meaning** of the game.

---

## What we could do (and why we don't)

Alternatives that would reduce the “one neighbour = 50%” effect:

1. **Unstandardized (binary) W**  
   Edge spatial lags would be smaller (sum of 2–3 vs 6). That reintroduces the original problem: edge systems would be artificially *deflated* and penalized in global and local Moran. Not desirable.

2. **Symmetric / double standardization**  
   e.g. \(w_{ij} / \sqrt{(\text{row}_i \text{ sum})(\text{row}_j \text{ sum})}\). Would dampen the edge effect but:
   - The spatial lag is no longer “average of my neighbours”;
   - The **LSAP normalization bound** in the README (division by \(n(n-1)\)) relies on row-standardization (each row sums to 1); the bound would need to be rederived or dropped;
   - Less standard in applied Moran/LISA practice.

3. **Degree-based or other weighting**  
   Would change the interpretation of the spatial lag and the current LSAP math.

**Conclusion:** There is no standard, drop-in alternative that both (a) keeps “row-standardized = weighted average of neighbours” and (b) equalizes per-neighbour influence across core vs edge. Changing **W** would alter the meaning of the spatial lag and the existing LSAP formulation. The choice to retain row-standardization is deliberate.

---

## The fix: acknowledge in methodology

Add a short paragraph in the methodology (e.g. in the README or paper, immediately after the existing row-standardization paragraph) that:

1. **States the consequence:** Row-standardization implies that each neighbour’s weight is inversely proportional to neighbour count, so a single neighbour can contribute up to ~50% of the spatial lag at edge tiles vs ~16% at central tiles.
2. **Reframes as strategic reality:** This magnifies the leverage of edge neighbours in the statistic, but it accurately reflects the game: if you are on the edge of the galaxy, the few systems adjacent to you dominate your strategic position.

**Optional sentence** (to show the choice was considered):  
“Using unstandardized or symmetric weights would equalize per-neighbour influence across positions but would either reintroduce systematic deflation of edge spatial lags (binary **W**) or change the interpretation of the spatial lag and the LSAP normalization (symmetric **W**); we therefore retain row-standardization.”

---

## Heuristic Scale Equalization

The optional $\sqrt{k_i}$ degree-based scaling for local $I_i$ acts as a computationally expedient heuristic to reduce scale differences at grid boundaries. While the exact variance of the local Moran statistic under a spatial null depends on higher-order moments and local topology, scaling by $\sqrt{k_i}$ provides a first-order empirical adjustment to mitigate heteroskedasticity in the optimization gradient. In the nominal benchmark, this heuristic is disabled to preserve the formal $n(n-1)$ normalization bound.

---

## Summary

| Question | Answer |
|----------|--------|
| **Can we “fix” it methodologically?** | Only by changing **W** (symmetric, degree weighting, etc.), which changes the meaning of the spatial lag and the LSAP bound; no standard fix preserves the current interpretation. Optional $\sqrt{k_i}$ heuristic scale equalization is available under `--corrected-landscape`; nominal benchmark keeps it off for the $n(n-1)$ bound. |
| **Is it a limitation?** | Yes, in the sense that local sensitivity is higher at edges; it is also a substantively reasonable reflection of TI4. |
| **Is the narrative fix sufficient?** | Optional $\sqrt{k_i}$ heuristic mitigates scale differences when enabled; acknowledging the edge-leverage effect and framing it as aligned with strategic reality completes the methodology. |

---

## References in codebase

- **W** construction and row-standardization: `src/ti4_analysis/algorithms/map_topology.py` (binary adjacency, then row-standardize; zero-sum rows guarded with denominator 1.0).
- Methodology text (row-standardization paragraph): `README.md`, spatial weights subsection (Moran's I / LSAP).
- LSAP bound derivation (depends on row-standardized **W**): `README.md`, LSAP normalization bound paragraph.
