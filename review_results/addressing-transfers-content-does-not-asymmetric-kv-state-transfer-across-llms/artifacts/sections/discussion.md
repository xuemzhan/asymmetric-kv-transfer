## Analysis
sec:analysis
### Correlation Transferability
A striking finding emerges from comparing geometric alignment
with functional transferability.
CCA results.
Canonical Correlation Analysis (CCA) reveals that
first canonical correlation for both K and V
across all model pairs (Table~tab:cca).
This indicates near-perfect linear alignability:
teacher and student KV representations lie in essentially
the same linear subspace.
The dissociation.
Despite near-perfect geometric alignment for both K and V,
only K transfers functionally.
V-only is weak or negative on these pairs (Table~tab:main),
even though trails by roughly 0.005
(Figure~fig:cca; K heads exceed V heads on 75--79\% of heads).
This dissociation is the paper's central finding.
Interpretation.
Linear-geometric compatibility (CCA ) is necessary but not sufficient.
The bottleneck is downstream decode, where content is consumed through weight-parameterized pathways:
K geometry is model-agnostic; V content is weight-bound.
### Layer Localization of V Advantage
Per-layer V-only replacement reveals concentrated hotspots
at layers 8 and 12 (Table~tab:layers).
Concentration.
Layer 12 alone accounts for (EM 1.00).
Layer 8 contributes .
Injecting both selectively () yields 
with EM 0.75--0.89.
Injecting all 28 layers (as in G0 V-only) is harmful: ,
EM 0.14--0.29---below Self (0.43).
Mechanism.
V transferable information is sparse (layers 8/12 only);
full-layer V injection fails because 26/28 layers inject noise,
explaining the G0 V-only failure.
This reconciles near-perfect geometric alignment ()
with functional failure: alignment is averaged over all layers,
but only two carry transferable content.
Asymmetry with K.
K advantage distributes globally (Figure~fig:layers):
the best single K layer (L21) gives , and
cumulative prefix injection gives to 
with EM ---K saturates as coverage grows,
while early layers alone () hurt ().
K geometry is a property of the entire network,
while V content is localized to specific processing stages.
### Cost Crossover
Mapper cost.
Affine mapper parameters: 58.78M (224.2 MiB at float32), fixed regardless of sequence length.
Cache cost.
Per-token KV cache: 112 KB (28-layer student), scaling linearly with sequence length.
Crossover.
State/weight transfer cost crossover at 2050 tokens (Figure~fig:cost).
Below 2050 tokens, KV cache transfer is cheaper;
above, mapper transfer is preferred.
### Boundary Conditions
The 6-pair design reveals the capability-gap gating boundary:
1.7B0.6B (V success at minimum gap).
V-only (sig in all 3 seeds), EM 0.430.80.
This is the only 0.6B-student pair where V transfers, and the only
2828 equal-layer pair besides 8B4B.
The gating factor is equal-layer alignment, not absolute size.
4B1.7B / 8B1.7B (V likelihood--EM dissociation).
V-only (sig) but EM collapses to 0.12/0.07.
V content improves likelihood yet breaks student greedy decoding;
under ``V is shared content'' theory, both should succeed together.
This dissociation localizes the failure to decode-time consumption.
8B4B (saturation).
Self EM = 100\%, so is partially calibration gain,
not pure task improvement.
The student already solves the task; K transfer improves calibration
while V transfer preserves EM (0.88).
### Synthesis
The evidence converges on a clear picture:
K is addressing geometry (model-agnostic, composable, transfers
near-universally); V is weight-bound content (transfers only under
equal-layer capability matching, and its likelihood gains can coexist with
decoding collapse).
This explains the functional asymmetry despite geometric equivalence,
and refines the earlier ``V never transfers'' claim:
V does transfer on closely matched pairs, but the boundary is narrow.