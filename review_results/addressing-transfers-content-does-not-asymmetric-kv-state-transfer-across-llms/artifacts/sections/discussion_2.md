## Discussion
sec:discussion
### Implications
For LLM serving.
Transfer K state, not V, across model versions---but only when the
teacher--student capability gap is large. On closely matched pairs,
V transfer is viable (8B4B: EM 0.88) and can be selectively
exploited via layers 8/12, avoiding the 26 noise layers.
The cost crossover at 2050 tokens remains the practical decision rule.
For distillation.
Prior work focuses on matching attention patterns (K geometry).
This is misguided: K geometry is already generic.
Focus instead on V/content pathways, which carry task-specific knowledge---and
note that V knowledge is concentrated in a few layers, making selective
layer distillation a promising target.
For model stitching.
Heterogeneous reassembly (teacher K + student V) beats both full models.
Addressing geometry is a composable substrate, and the sparse localization
of V advantage (L8/L12) suggests where cross-model stitching should focus.
### Limitations
Single model family.
All 6 pairs come from Qwen3; cross-family transfer (e.g., LlamaQwen)
remains untested and may behave differently under different RoPE
configurations and attention implementations.
EM saturation on 3/6 pairs.
8B4B, 4B1.7B, 8B1.7B saturate at Self EM = 100\%;
there, only is diagnostic, and V's likelihood gains
must be interpreted with caution.
V collapse on 1.7B students.
V-only improves likelihood but collapses greedy EM (0.07--0.12) on
4B1.7B and 8B1.7B. Whether this reflects calibration effects
or genuine content misalignment is not fully resolved.
SQuAD scale.
The second-domain validation uses 30 questions; larger benchmarks
would tighten the cross-domain claim.
Mapper family.
Only linear mappers (Affine/Whitened/Procrustes/CCA/RidgePerHead) were tested.
Nonlinear mappers (MLP/attention-based) could alter the boundary,
especially for V transfer on mid-gap pairs.
### Future Work
Cross-family transfer.
Test whether the gated K/V asymmetry holds across model families
(Llama/Qwen/Mistral), where RoPE and attention implementations differ.
Per-hop difficulty analysis.
Scale to per-hop breakdown (hop-1 through hop-4) to characterize
how transfer success varies with reasoning depth.
Theoretical modeling.
Why does K geometry transfer but V content only under capability matching?
A theoretical framework would predict the gating boundary
from model architecture and training data.
Selective layer transfer in production.
Leverage the L8/L12 V hotspot to design layer-selective cache transfer
that avoids the 26 noise layers, and measure actual serving gains.
### Conclusion
We have shown that KV state transfer across LLMs exhibits
a sharp, capability-gated functional asymmetry:
keys (addressing) transfer near-universally across 6 model pairs
(EM ; positive in 5/6), while values (content) transfer
only under equal-layer capability matching (8B4B EM 0.88,
1.7B0.6B EM 0.80), collapse greedy decoding on mid-gap students
(EM 0.07--0.12), and fail entirely on large-gap pairs
(, ns) despite near-perfect linear alignment
(CCA for both K and V).
The V advantage localizes to layers 8/12; selective injection
( ) beats full-layer injection ( ).
The asymmetry replicates on a second domain (SQuAD), where V-only
injection destroys decoding (EM 0.03), and heterogeneous reassembly
(teacher K + student V) beats both full models ( vs /).
Our results establish that KV state transfer is asymmetric by construction:
addressing geometry is composable across models,
while content is weight-bound---transferable only where
weights already agree.
This has implications for serving, distillation,
and modular model construction.
thebibliography10
bansal2021stitching
Yamini Bansal, Preetum Nakkiran, and Boaz Barak.
 How could we ever measure the perceptron alignment of neural
 networks?
 In International Conference on Learning Representations, 2021.
 Model stitching.
chen2023accelerating
Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, et~al.
 Accelerating large language model decoding with speculative sampling.
 arXiv preprint arXiv:2302.01318, 2023.
dao2022flashattention
Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R\'e.
 Flashattention: Fast and memory-efficient exact attention with
 io-awareness.
 Advances in Neural Information Processing Systems, 35, 2022.
dery2026latentalign
Lucio~M. Dery, Zohar Yahav, Henry Prior, Qixuan Feng, Jiajun Shen, and Arthur
 Szlam.
 Latent space communication via k-v cache alignment.
 arXiv preprint arXiv:2601.06123, 2026.
dettmers2024qlora
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer.
 Qlora: Efficient finetuning of quantized language models.
 Advances in Neural Information Processing Systems, 36, 2024.
frantar2023gptq
Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh.
 Gptq: Accurate post-training quantization for generative pre-trained
 transformers.
 arXiv preprint arXiv:2210.17323, 2023.
fu2026c2c
Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, and
 Yu~Wang.
 Cache-to-cache: Direct semantic communication between large language
 models.
 In International Conference on Learning Representations, 2026.
 arXiv:2510.03215.
heo2026crossmodel
Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub, Mohammad~Mahdi
 Kamani, Ritika Borkar, Makesh~Tarun Chandran, Pantea Zardoshti, and
 Bita~Darvish Rouhani.
 Cross-model kv cache transfer in llm families: A closed-form linear
 mapping for prefill reuse.
 arXiv preprint arXiv:2608.03893, 2026.
hinton2015distilling
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.
 Distilling the knowledge in a neural network.
 arXiv preprint arXiv:1503.02531, 2015.
hoffmann2022chinchilla
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et~al.
 Training compute-optimal large language models.
 Advances in Neural Information Processing Systems, 35, 2022.
kaplan2020scaling
Jared Kaplan, Sam McCandlish, Tom Henighan, et~al.
 Scaling laws for neural language models.
 arXiv preprint arXiv:2001.08361, 2020.
lee2026translators
Jin-woo Lee, Minkyung Song, Junghyun Oh, Seunghoon Han, Soyoung Park, Gwangseon
 Jang, and Sungsu Lim.
 Mixture-of-translators: Translating kv caches across heterogeneous
 large language models.
 arXiv preprint arXiv:2607.28979, 2026.
leviathan2023fast
Yan Leviathan, Matan Kalman, and Yossi Matias.
 Fast inference from transformers via speculative decoding.
 International Conference on Machine Learning, pages
 19274--19290, 2023.
su2024rope
Jianlin Su, Murtadha Ahmed, Yu~Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu.
 Roformer: Enhanced transformer with rotary position embedding.
 Neurocomputing, 568:127063, 2024.
touvron2023llama
Hugo Touvron, Thibaut Lavril, Gautier Izacard, et~al.
 Llama: Open and efficient foundation language models.
 arXiv preprint arXiv:2302.13971, 2023.
vaswani2017attention
Ashish Vaswani, Noam Shazeer, Niki Parmar, et~al.
 Attention is all you need.
 Advances in Neural Information Processing Systems, 30, 2017.
thebibliography
document