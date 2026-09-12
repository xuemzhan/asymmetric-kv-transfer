## Theory Contribution Review

### 3 Fatal Theory Holes
1. (introduction) "Across 6 model pairs spanning 0.6B to 8B parameters, K-only injection improves log-likelihood by to on 4/6 pairs (paired , 3 seeds) and recovers EM on all pairs, while V-only injection is significant only on the two small-gap pairs (B4: ; B0.6: ) and neutral or negative elsewhere (B0.6: , ns)." — At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.
2. (introduction) "Prior work transfers entire KV caches without identifying which components (keys vs.\ values) are transferable." — The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

### Concrete Moves
- Tighten the paper's theoretical positioning in introduction to resolve abstract and conclusion claims need explicit evidence traceability.
- Tighten the paper's theoretical positioning in introduction to resolve novelty claim should be grounded against the closest prior work.
