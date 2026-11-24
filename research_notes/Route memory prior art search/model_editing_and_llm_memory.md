# Model editing, lifelong knowledge editing and memory modules for frozen LLMs: prior art for "route memory with surprise nudging"

Verification legend used throughout:
- **[V]** = read from the primary source (or its abstract/tables) in this research session.
- **[R]** = stated from the researcher's recall of the primary paper; the link is the primary source but it was NOT re-opened in this session. Treat as "needs a check before it is quoted in a paper".
- Numbers in tables are all [V] and are the cited papers' own reported numbers (no independent replication was found).

Method under check (short name "RM"): frozen LM; one forward pass per told prompt; per token store (route = binary on/off of every MLP unit in every block, told next token, model's predicted distribution); at generation, similarity = share of equal units to a power, carried along the stored sequence; logits = log p_model + eta * sum_places P[place] * (told - predicted); no gate, no threshold, no scope classifier.

---

## 1. Weight-editing methods (ROME, MEMIT, PMET, AlphaEdit, r-ROME/EMMET, MEND, KE, knowledge neurons, UltraEdit)

### Takeaway
All of these change the model's weights (or train a hypernetwork that does), almost all need a per-edit optimisation or a gradient, and none stores the model's output prediction error beside the weights; they differ from RM on every axis except "the edit is a rank-one/low-rank association keyed on an internal activation". Their well-documented failure is collapse under sequential edits, which RM cannot have in the same form because it never writes to weights.

### Cited Findings
- **ROME** [R]: stores one fact as a rank-one update to one mid-layer MLP output matrix; key = that MLP's input activation at the subject's last token (averaged over prefixes), value = a vector found by gradient optimisation so the layer output makes the model predict the new object; needs a pre-computed key covariance from Wikipedia; no scope gate (the key direction itself is the implicit scope); weights changed; applied by simply running the edited model. — [Meng et al. 2022, arXiv 2202.05262](https://arxiv.org/abs/2202.05262)
- **MEMIT** [R]: same key/value idea spread over a range of MLP layers, batch of thousands of facts in one least-squares update; per-fact value optimisation still required. — [Meng et al. 2023, arXiv 2210.07229](https://arxiv.org/abs/2210.07229)
- **PMET** [R]: optimises both attention and MLP hidden states but writes only the MLP weights. — [Li et al. 2023, arXiv 2308.08742](https://arxiv.org/abs/2308.08742)
- **r-ROME / EMMET** [R]: r-ROME fixes an implementation inconsistency in ROME's key usage that causes "disabling edits"; EMMET unifies ROME and MEMIT as the same preservation-memorisation objective with equality vs least-squares constraints. — [Gupta et al., Rebuilding ROME, arXiv 2403.07175](https://arxiv.org/abs/2403.07175); [Gupta et al., A Unified Framework for Model Editing, arXiv 2403.14236](https://arxiv.org/abs/2403.14236)
- **AlphaEdit** [V, via its appearance as a baseline, method details R]: MEMIT-style update projected onto the null space of preserved-knowledge keys before it is added to the weights. — [Fang et al. ICLR 2025, arXiv 2410.02355](https://arxiv.org/html/2410.02355v4)
- **MEND** [R]: a trained hypernetwork maps the fine-tuning gradient of an edit (rank-one decomposed) to a weight update; needs meta-training on an edit dataset; then one forward+backward pass per edit. — [Mitchell et al. 2022, arXiv 2110.11309](https://arxiv.org/abs/2110.11309)
- **KE (KnowledgeEditor)** [R]: trained hypernetwork predicts weight updates under a constraint to keep other predictions unchanged. — [De Cao et al. 2021, arXiv 2104.08164](https://arxiv.org/abs/2104.08164)
- **Knowledge neurons** [R]: attributes a fact to a few MLP neurons by integrated gradients, then edits by suppressing/amplifying them or altering their value vectors; continuous attribution scores, not binary on/off patterns used as an address. — [Dai et al. 2022, arXiv 2104.08696](https://arxiv.org/abs/2104.08696)
- **UltraEdit** [V, from search summary of the abstract]: "training-, subject-, and memory-free"; computes the parameter shift in one step from a hidden state and its gradient; still edits weights and still uses a gradient. — [Gu et al. 2025, arXiv 2505.14679](https://arxiv.org/abs/2505.14679)
- **SKIM (Oct 2026)** [V, abstract only]: "Self-Keyed In-Parameter Memorization": closed-form write of a dialogue turn into parameters, keys = many model-generated questions about the turn; no external memory. Despite "Frozen Language Models" in the title, the content is stored in parameters. — [Piao et al. 2026, arXiv 2610.36948](https://arxiv.org/abs/2610.36948)
- Sequential-edit collapse of these methods, per the WISE paper's own table (ZsRE, LLaMA-2-7B, Rel/Gen/Loc): ROME T=1 0.85/0.80/0.99 → T=1000 0.01/0.01/0.00; MEMIT T=1 0.84/0.81/0.99 → T=1000 0.04/0.04/0.02; MEND T=1 0.95/0.93/0.98 → T=100 0.00/0.00/0.00; MEMIT-MASS (all 1000 at once) 0.69/0.65/0.62. [V] — [WISE, arXiv 2405.14768](https://arxiv.org/html/2405.14768v2)

### Inferences
- Exact differences from RM: (a) weights are modified vs RM never modifies; (b) the key is one layer's continuous MLP input at one chosen token (subject last token) vs RM's binary gating pattern over all blocks at every token; (c) the value is an optimised hidden vector that indirectly causes the target vs RM's value being the output-space residual (told − predicted) applied directly at the logits; (d) per-edit optimisation (ROME/MEMIT/PMET/AlphaEdit) or meta-training (MEND/KE) vs RM's single forward pass; (e) they need a subject span / triple structure vs RM takes raw text.
- One true overlap: RM's update is "equivalent to one gradient step on the output weights per stored token". MEND and UltraEdit also start from the edit's gradient; FT-L (fine-tune one layer) is the plain version. RM differs by restricting to the output layer, keeping the step in dual (kernel) form beside the weights, and replacing the last-hidden-state inner product by route similarity. A reviewer may call RM "kernelised last-layer fine-tuning/dynamic evaluation kept in an external buffer" (see section 4).

### Gaps
- PMET, r-ROME, EMMET, MEND, KE, knowledge-neurons details are from recall, not re-read this session.
- UltraEdit's method section was not opened; only the abstract-level description was seen.

---

## 2. Memory-based / external methods (SERAC, GRACE, T-Patcher, MELO, WISE, MEMOIR, IKE, MeLLo, Larimar, MemoryLLM/M+, Memory Decoder, retrieval editors, INLAY, steering)

### Takeaway
The memory-based editors share RM's "frozen base model + side store" shape, but every one that keys on internal activations (GRACE, MELO, WISE, MEMOIR) pairs the key with an explicit gate (radius, threshold or router) and trains the stored value by gradient descent; RM's combination of no gate, no training and an output-space error value was not found in this family. Two components of RM do already exist separately: activation-pattern addressing with Hamming overlap (MEMOIR, 2025) and gradient-free, decode-time logit bias for edits on a frozen model (INLAY, 2026).

### Cited Findings
- **SERAC** [V via survey/ar5iv summary]: frozen base model + explicit cache of edit texts + a trained scope classifier that decides whether an input falls under a cached edit + a trained counterfactual model (T5-small in the original) that produces the answer for in-scope inputs. Classifier and counterfactual model are trained with supervision on an edit dataset; no per-edit training (an edit is appended to the cache). — [Mitchell et al. 2022, arXiv 2206.06520](https://ar5iv.labs.arxiv.org/html/2206.06520); [Wang et al. survey, ACM CSUR](https://dl.acm.org/doi/full/10.1145/3698590)
- **GRACE** [V]: adaptor at one chosen layer, original weights frozen; a discrete codebook whose keys are cached activations from layer l−1 and whose values replace the layer-l output; each key has a deferral radius epsilon; "if an encoding does not land within any key's epsilon-ball, then the frozen pre-trained model is used directly"; a new value is randomly initialised and then trained (fine-tuned by backprop to produce the target) per edit. — [Hartvigsen et al. NeurIPS 2023, arXiv 2211.11031](https://arxiv.org/pdf/2211.11031)
- GRACE's measured profile (from WISE's table, ZsRE, LLaMA-2-7B): Rel/Gen/Loc 0.99/0.36/1.00 at T=1 and 0.93/0.08/1.00 at T=1000; Mistral-7B 1.00/0.02/1.00 at T=1000. WISE attributes the poor generalisation to the codebook matching non-parametric token representations without "comprehending" them, so paraphrases fail to land in any key's ball. [V] — [WISE, arXiv 2405.14768](https://arxiv.org/html/2405.14768v2)
- **T-Patcher** [V one-line, rest R]: adds a few trainable "patch" neurons to the last MLP layer per mistake; patch trained by gradient so it fires only on the edited input. — [Huang et al. 2023, arXiv 2301.09785](https://arxiv.org/abs/2301.09785); one-line description confirmed in [Stable Knowledge Editing, arXiv 2402.13048](https://arxiv.org/html/2402.13048v1)
- **MELO** [R]: GRACE-style vector database on a layer's activations with cluster radii that indexes which trained dynamic-LoRA block to switch on; LoRA blocks are trained per edit batch. — [Yu et al. AAAI 2024, arXiv 2312.11795](https://arxiv.org/abs/2312.11795)
- **WISE** [V]: copies one FFN value matrix into a side memory; only the side copy is trained (autoregressive NLL on the edit + a margin loss), with random gradient masks per shard and TIES merging; routing indicator = L2 norm of activation × (side − main weights); at inference the side memory is used only if the indicator exceeds epsilon (the minimum over seen edits). ZsRE LLaMA-2-7B Rel/Gen/Loc: T=1 0.98/0.92/1.00; T=1000 0.77/0.72/1.00. — [Wang et al. NeurIPS 2024, arXiv 2405.14768](https://arxiv.org/html/2405.14768v2)
- **MEMOIR** [V]: residual memory module at one FFN projection; per edit, a sparse mask is computed by "TopHash" (top-k of the prompt-averaged, mean-centred input activation, then a fixed random permutation; k = 4096) and gradient updates are restricted to the masked columns; the mask of every edit is stored; at inference the query's mask is compared to stored masks by Hamming distance, and the residual memory is used only if overlap ratio ≥ tau, otherwise the original layer alone. ZsRE T=1000 Rel/Gen/Loc: LLaMA-3-8B 0.94/0.85/1.00, Mistral-7B 0.94/0.89/1.00; LLaMA-3 at 15,000 edits 0.87/0.78/0.98. — [Wang et al. NeurIPS 2025, arXiv 2506.07899](https://arxiv.org/html/2506.07899v4)
- **IKE** [R]: no parameters touched; the new fact plus retrieved copy/update/retain demonstrations are put in the prompt; scope is handled implicitly by the demonstrations and the retriever. — [Zheng et al. 2023, arXiv 2305.12740](https://arxiv.org/abs/2305.12740)
- **MeLLo** [R]: edits stored as text in a memory; the model decomposes a multi-hop question, retrieves edits per sub-question and checks for contradiction; introduced with the MQuAKE benchmark. — [Zhong et al. 2023, arXiv 2305.14795](https://arxiv.org/abs/2305.14795)
- **Larimar** [V]: frozen-at-edit-time LLM decoder conditioned on a distributed episodic memory; memory writes are one-shot ("updates happen in a single step", no retraining), reported 4-10x faster than baselines in sequential editing; but the encoder, memory and decoder coupling is trained end-to-end by gradient descent on generic data beforehand. — [Das et al. ICML 2024, PMLR 235](https://proceedings.mlr.press/v235/das24a.html); [arXiv 2403.11901](https://arxiv.org/pdf/2403.11901)
- **MemoryLLM / M+** [R]: a fixed-size pool of memory tokens (hidden states) in every layer of a transformer that is trained to self-update by overwriting part of the pool with compressed hidden states of new text; M+ adds a retriever over a long-term pool; the model must be trained to use the pool. — [Wang et al. 2024, arXiv 2402.04624](https://arxiv.org/abs/2402.04624); [M+, arXiv 2502.00592](https://arxiv.org/abs/2502.00592)
- **Memory Decoder** [V]: a small transformer decoder is trained (KL to a kNN-LM distribution + LM loss) to imitate a non-parametric retriever; at inference its output distribution is interpolated with the frozen base model's; one 0.5B memory serves the Qwen2.5 family. — [Cao et al. NeurIPS 2025, arXiv 2508.09874](https://arxiv.org/html/2508.09874v1)
- **ResMem, "From Lookups to Lessons" (Nov 2026)** [V, search summary only]: trains a logit residual that is added to the base model's logits so base + residual matches an evidence-reading teacher; residual alone scores F1 0.06, i.e. it is only meaningful when added to base logits. — [arXiv 2611.00450](https://arxiv.org/html/2611.00450)
- **INLAY (Aug 2026)** [V, abstract only]: memory-based editor in the SERAC lineage; model frozen, gradient-free; "at decode time, an edit is applied as a bias added along one token's unembedding direction" (i.e. a logit-space push); it has per-query routing actions including abstention. The paper's main result is negative: on 1,689 queries over three datasets, an oracle per-query router ties a one-line static policy in all nine cells, and abstention never wins, because counterfactual benchmarks contain no negative cases; adding withheld-edit negatives gives +0.042 headroom. — [Singh 2026, arXiv 2608.26292](https://arxiv.org/abs/2608.26292)
- **JitRL (Jan 2026)** [V, search summary]: training-free test-time adaptation of a frozen LLM agent; non-parametric memory of (state, action, reward) triplets; retrieved experience gives an advantage estimate that is added to the action logits. Policy learning for agents, not text memory. — [Li et al. 2026, arXiv 2601.18510](https://arxiv.org/pdf/2601.18510v1)
- **EREN** [V title only, method R]: edits kept as a notebook of text; the LLM reads retrieved notes and decides relevance itself. — [Chen et al. 2024, arXiv 2403.17431](https://arxiv.org/html/2403.17431v1)
- **RECIPE, LTE, DeepEdit** [R]: RECIPE turns each edit into a learned continuous prompt with a learned "knowledge sentinel" threshold for retrieval; LTE fine-tunes the model to follow retrieved edits; DeepEdit is constrained decoding/search over retrieved edits. — [RECIPE, arXiv 2405.03279](https://arxiv.org/abs/2405.03279); [LTE, arXiv 2402.11905](https://arxiv.org/abs/2402.11905); [DeepEdit, arXiv 2401.10471](https://arxiv.org/abs/2401.10471)
- **Steering family (REMEDI, ReFT, in-context vectors, function vectors, task vectors)** [R]: add or learn a vector/low-rank intervention in the residual stream at chosen layers and positions; REMEDI and ReFT train the intervention; ICV / function / task vectors are extracted from forward passes of demonstrations (training-free) but encode a task or style, not a set of token-level facts, and act on hidden states not logits. — [REMEDI, arXiv 2304.00740](https://arxiv.org/abs/2304.00740); [ReFT, arXiv 2404.03592](https://arxiv.org/abs/2404.03592); [ICV, arXiv 2311.06668](https://arxiv.org/abs/2311.06668); [Function vectors, arXiv 2310.15213](https://arxiv.org/abs/2310.15213); [Task vectors, arXiv 2310.15916](https://arxiv.org/abs/2310.15916)

### Comparison table (RM vs the nearest editors)

| Method | Stored | Key comes from | Gate / router | Weights changed | Training per edit | Applied at |
|---|---|---|---|---|---|---|
| RM (ours) | route + told token + predicted distribution, per token | binary on/off of all MLP units, all blocks, every token; sequence-carried | none | no | none (1 forward pass) | logits, additive (told − predicted) |
| GRACE | key vector + trained value vector | one layer's continuous activation (one token) | yes, epsilon-ball per key | no (adaptor) | yes, value by backprop | replaces layer-l hidden state |
| MELO | cluster keys + LoRA blocks | one layer's activation | yes, radius | no (LoRA side) | yes | LoRA at layers |
| WISE | side copy of FFN matrix | activation × weight-difference norm | yes, threshold + trained margin | side copy only | yes | swaps FFN output |
| MEMOIR | sparse masks + residual memory columns | top-k mask of one FFN input (prompt mean) | yes, Hamming overlap ≥ tau | residual module | yes | residual FFN output |
| SERAC | edit texts | trained classifier embedding | yes, trained scope classifier | no | no (pre-trained parts) | separate counterfactual model answers |
| INLAY | edits in addressable memory | not stated in abstract | yes, routing incl. abstention | no | no (gradient-free) | logit bias on one token's unembedding direction |
| Larimar | latent episodic memory matrix | trained encoder | read by addressing weights | no at edit time | no at edit time (pre-trained end-to-end) | decoder conditioned on read vector |
| IKE | text demonstrations | text retriever | implicit | no | no | prompt |
| kNN-LM (sec. 4) | last hidden state + next token | final-layer continuous state | none (fixed lambda) | no | none (1 forward pass) | probability interpolation |

### Inferences
- GRACE is the editor whose shape is nearest to RM's (frozen model, activation-keyed external store, edits appended one at a time), but RM removes or replaces each of its three parts: no deferral radius, no trained value, and key = whole-network binary pattern instead of one layer's vector. GRACE's published profile (reliability ≈ 1.0, generalisation 0.02-0.37) is the default expectation for any exact activation-keyed memory, so RM's "told facts recalled 100% / 97%" corresponds to the Rel column only and is not evidence of generalisation.
- MEMOIR is the closest prior use of a sparse/binary activation pattern compared by Hamming overlap as an address. It differs in using one layer, a top-k mask of a prompt-averaged activation (one mask per prompt, not per token), a threshold, and gradient-trained content.
- INLAY is the closest prior "gradient-free, frozen model, edit applied as a logit push at decode time" editor. Differences: it pushes a chosen target token's direction (not told − predicted), and it has routing. Its abstract did not state where keys come from, so overlap on the key cannot be ruled out.
- RM's deliberate lack of a gate has support in INLAY's negative result (on current counterfactual benchmarks an oracle router gains nothing), but the same paper shows a gate starts to matter as soon as negatives exist; an honest RM paper should cite it both ways.

### Gaps
- "kNN-KE" as a named method: no paper by that name was found in two searches; cannot confirm it exists.
- LEMoE ([arXiv 2406.20030](https://arxiv.org/abs/2406.20030)), ELDER ([arXiv 2408.11869](https://arxiv.org/abs/2408.11869)) and MEMAT were not opened; from recall LEMoE and ELDER are mixtures of trained LoRA/adapters with a learned or key-based router (both gated and trained), MEMAT adds attention-head corrections to MEMIT (weight edit). Unverified.
- INLAY's key source and full method (only the abstract was read). This is the one paper most worth reading in full before a novelty claim.
- GRACE's exact per-edit training loop and MELO, T-Patcher, IKE, MemoryLLM details are from recall.

---

## 3. Is there any method that (i) needs one forward pass and no optimisation, (ii) stores the model's own output prediction error and applies it at the logits, or (iii) deliberately has no scope gate?

### Takeaway
(i) and (iii) each exist already, most clearly in the kNN-LM / neural cache line (one forward pass per stored token, no gate, applied at the output) and in IKE, Larimar writes and INLAY for editing; (ii) in the exact form "store (told − predicted) per token in an external buffer and add it, similarity-weighted, to frozen logits" was not found in the editing/memory literature searched, though it is mathematically a kernel form of last-layer gradient descent, which is old. No source found combines (i)+(ii)+(iii) with a whole-network binary activation key.

### Cited Findings
- (i) One forward pass, no optimisation: kNN-LM builds its datastore with a single forward pass over text, storing (context vector, next token), and "makes no changes to the underlying LM". [V] — [Khandelwal et al. ICLR 2020, arXiv 1911.00172](https://arxiv.org/pdf/1911.00172)
- (i) Larimar's memory write is one-shot, no fine-tuning (but the system was trained beforehand). [V] — [Das et al. 2024](https://proceedings.mlr.press/v235/das24a.html)
- (i) INLAY is gradient-free on a frozen model. [V, abstract] — [arXiv 2608.26292](https://arxiv.org/abs/2608.26292)
- (ii) Logit-additive memory on a frozen model, training-free: JitRL adds a retrieved advantage estimate to action logits [V summary] — [arXiv 2601.18510](https://arxiv.org/pdf/2601.18510v1); INLAY adds a bias along a token's unembedding direction [V abstract] — [arXiv 2608.26292](https://arxiv.org/abs/2608.26292); ResMem adds a trained logit residual [V summary] — [arXiv 2611.00450](https://arxiv.org/html/2611.00450). None of the three is described as storing the base model's own prediction error.
- (ii) A search for a kNN/cache memory that stores a residual or "predicted distribution" correction returned no such method; the standard kNN-LM datastore "is never corrected or edited; it is only queried" and holds targets only. [V] — [kNN-LM, arXiv 1911.00172](https://arxiv.org/pdf/1911.00172)
- (iii) No gate: kNN-LM and the continuous cache always interpolate with a fixed weight; there is no decision whether a memory applies. [V] — [kNN-LM](https://arxiv.org/pdf/1911.00172); [Dynamic Evaluation paper's description of the neural cache, arXiv 1709.07432](https://arxiv.org/pdf/1709.07432)
- (iii) Among editors proper, the weight editors (ROME, MEMIT, FT) have no explicit gate but that is because the edit is in the weights; all activation-keyed memory editors read in this session (GRACE, WISE, MEMOIR, SERAC, INLAY) have an explicit gate. [V] — [GRACE](https://arxiv.org/pdf/2211.11031); [WISE](https://arxiv.org/html/2405.14768v2); [MEMOIR](https://arxiv.org/html/2506.07899v4)
- Analysis (not a method) supporting the binary-route idea: a March 2026 preprint on GPT-2 Small reports MLP neurons operate in a near-binary regime for most tokens and frames binarised patterns as "binary routing" of continuous signals. [V, search summary] — [The Discrete Charm of the MLP, arXiv 2603.10985](https://arxiv.org/html/2603.10985)

### Inferences
- The term (told − predicted) is exactly the gradient of the cross-entropy loss with respect to the logits. Summing it over stored places with a similarity weight is one step of gradient descent on the output layer written in dual (kernel) form, where the kernel is route similarity instead of the inner product of final hidden states. So claim (ii) is new as a stored object in an editing memory as far as this search found, but it is not a new mathematical object; the method's own description ("equivalent to one gradient step on the output weights per stored token") already concedes this. The defensible novelty is the combination: residual stored instead of target, whole-network binary gating pattern as the kernel, sequence-carried similarity, no gate, zero optimisation.
- Storing the residual instead of the target has a concrete consequence that distinguishes RM from kNN-LM: tokens the model already predicts well contribute nothing, so told text the model already knows does not push. kNN-LM pushes toward every stored target equally.
- The kernel "share of gating units equal, to a power" is close in spirit to the gating part of the neural tangent kernel of a ReLU network. If another researcher's notes cover NTK / kernel-regression views, the two should be joined; no source was read on this here.

### Gaps
- No exhaustive check was possible that nobody stores output residuals in a kNN datastore; the claim rests on two targeted searches returning nothing, not on a survey statement.
- The March 2026 "binary routing" paper was not opened; whether it proposes any memory use of the patterns is unknown (the search summary says it is analysis only).

---

## 4. Non-editing ancestors that a reviewer will name: kNN-LM, continuous cache, dynamic evaluation, Hebbian softmax

### Takeaway
Mechanistically the closest existing family to RM is not an "editor" at all: it is kNN-LM / neural cache (frozen model, one forward pass, per-token store, no gate, output-level mixing) together with dynamic evaluation and Hebbian softmax (fast learning at the output layer from prediction error or activations). RM sits between them: a kNN-LM-style store whose content is dynamic evaluation's last-layer gradient.

### Cited Findings
- kNN-LM: frozen LM, datastore of (context representation → next token) built in one pass, nearest-neighbour distribution interpolated with the model distribution by a tuned lambda; gains are additive with the continuous cache (WikiText-103 perplexity 15.81 with cache vs 16.06 without). [V] — [Khandelwal et al., arXiv 1911.00172](https://arxiv.org/pdf/1911.00172)
- Neural/continuous cache: "learns a type of non-parametric output layer on the fly at test time", nearest-neighbour-like over recent hidden states; the dynamic evaluation paper contrasts it with its own gradient-descent adaptation and notes the cache "cannot adjust the recurrent hidden state dynamics". [V] — [Krause et al., Dynamic Evaluation of Neural Sequence Models, arXiv 1709.07432](https://arxiv.org/pdf/1709.07432)
- Dynamic evaluation: adapts model parameters by gradient descent on the sequence seen so far at test time; revisited for LLMs in 2023. [V] — [arXiv 1709.07432](https://arxiv.org/pdf/1709.07432); [Revisiting Dynamic Evaluation, NeurIPS 2023 workshop](https://neurips.cc/virtual/2023/80505)
- Hebbian softmax: "accumulates activations into the softmax layer weights" for rarely seen classes (a Hebbian update on the output layer), mixed with gradient descent; WikiText-103 perplexity 29.2 at the time. It writes into the output weights and stores activations for the told class, not an error term. [V] — [Rae et al. ICML 2018, arXiv 1803.10049](https://arxiv.org/pdf/1803.10049)
- Representation memorisation: "Hebbian parameter adaptation that directly wires the output network's parameters with similar representations retrieved from the memory", aimed at learning new knowledge without forgetting. [V, search summary] — [Mi, Lin, Faltings 2021, arXiv 2108.12596](https://arxiv.org/abs/2108.12596)
- Test-time training on nearest neighbours: retrieves neighbours of each test instance and fine-tunes on them before predicting (gradient steps, weights changed temporarily). [V] — [Hardt and Sun, ICLR 2024, arXiv 2305.18466](https://arxiv.org/html/2305.18466v3)
- Memory Decoder shows the current (2025) form of this line: train a small model to stand in for the kNN distribution, then interpolate. [V] — [arXiv 2508.09874](https://arxiv.org/html/2508.09874v1)

### Inferences
- Exact differences from kNN-LM: (1) key = binary on/off of all MLP units in all blocks vs one continuous final-layer vector; (2) similarity over all stored places with soft weights and carried along the sequence vs top-k nearest neighbours per step independently; (3) stored value = (told − predicted) added in log space vs one-hot target interpolated in probability space; (4) purpose: a small told set meant to shift the model vs a huge corpus datastore.
- Exact differences from dynamic evaluation: no weights are changed, no backward pass, and the step is not on the last hidden state's inner-product kernel. If the route kernel were replaced by the final-hidden-state inner product, RM's nudge would be literally one SGD step on the unembedding matrix; this is the sentence a reviewer will write, and the paper should write it first.
- These four papers must be cited as ancestors; calling RM a "model editing" method without them would be the main prior-art risk.

### Gaps
- No source was found that states the formal equivalence between a residual-storing cache and last-layer dynamic evaluation; it is asserted here as an inference from the definitions.

---

## 5. Failure modes and benchmarks to position against

### Takeaway
The field's standard yardsticks are reliability / generalisation / locality (plus portability) on zsRE and CounterFact at T = 1 to 1000+ sequential edits on 7-8B models, with GRACE ≈ 1.00/0.02-0.37/1.00, WISE ≈ 0.77/0.72/1.00, MEMOIR ≈ 0.94/0.85/1.00 and ROME/MEMIT ≈ 0 at T=1000. RM's current evidence (60 sentences, Qwen3-0.6B, recall of told text, likelihood of unrelated text) covers only the reliability column and a perplexity-style locality proxy, so it cannot yet be placed on these tables.

### Cited Findings
- zsRE sequential editing, LLaMA-2-7B, Rel/Gen/Loc/Avg (WISE paper, Table of main results) [V] — [arXiv 2405.14768](https://arxiv.org/html/2405.14768v2):

| Method | T=1 | T=10 | T=100 | T=1000 |
|---|---|---|---|---|
| FT-L | 0.57/0.52/0.96/0.68 | 0.48/0.48/0.76/0.57 | 0.30/0.27/0.23/0.27 | 0.19/0.16/0.03/0.13 |
| MEND | 0.95/0.93/0.98/0.95 | 0.26/0.28/0.28/0.27 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 |
| ROME | 0.85/0.80/0.99/0.88 | 0.64/0.62/0.75/0.67 | 0.23/0.22/0.04/0.16 | 0.01/0.01/0.00/0.01 |
| MEMIT | 0.84/0.81/0.99/0.88 | 0.58/0.58/0.85/0.67 | 0.02/0.02/0.02/0.02 | 0.04/0.04/0.02/0.03 |
| MEMIT-MASS | 0.84/0.81/0.99/0.88 | 0.75/0.72/0.97/0.81 | 0.76/0.68/0.85/0.76 | 0.69/0.65/0.62/0.65 |
| DEFER | 0.68/0.58/0.56/0.61 | 0.65/0.47/0.36/0.49 | 0.20/0.12/0.27/0.20 | 0.03/0.03/0.74/0.27 |
| GRACE | 0.99/0.36/1.00/0.78 | 0.96/0.16/1.00/0.71 | 0.96/0.15/1.00/0.70 | 0.93/0.08/1.00/0.67 |
| WISE | 0.98/0.92/1.00/0.97 | 0.94/0.88/1.00/0.94 | 0.90/0.81/1.00/0.90 | 0.77/0.72/1.00/0.83 |

- Same table, Mistral-7B at T=1000: GRACE 1.00/0.02/1.00, WISE 0.70/0.67/1.00, ROME 0.04/0.04/0.02, MEMIT 0.04/0.04/0.02, MEMIT-MASS 0.73/0.70/0.62. [V] — [arXiv 2405.14768](https://arxiv.org/html/2405.14768v2)
- Hallucination set (SelfCheckGPT; Rel = perplexity of the corrected text, lower is better; Loc higher is better), LLaMA-2-7B at T=600: GRACE 9.34/1.00, WISE 3.12/0.99, MEMIT 107.61/0.02, ROME 104.93/0.02, MEMIT-MASS 13.47/0.94. [V] — [arXiv 2405.14768](https://arxiv.org/html/2405.14768v2)
- Later numbers (MEMOIR paper, zsRE, LLaMA-3-8B, T=1000, Rel/Gen/Loc): ROME 0.03/0.03/0.02; MEMIT 0.00/0.00/0.00; GRACE 1.00/0.37/1.00; WISE 0.66/0.64/1.00; AlphaEdit 0.84/0.77/0.56; MEMOIR 0.94/0.85/1.00. At 15,000 edits: GRACE 0.99/0.28/1.00; WISE 0.43/0.41/1.00; AlphaEdit 0.02/0.02/0.00; MEMOIR 0.87/0.78/0.98. [V] — [arXiv 2506.07899](https://arxiv.org/html/2506.07899v4)
- Cost comparison reported in a 2025 paper: edit time per fact about 21 s for WISE vs about 4.3 s for GRACE. [V, search summary] — [Knowledge Updating? No More Model Editing! Just Selective Contextual Reasoning, arXiv 2503.05212](https://arxiv.org/pdf/2503.05212)
- An in-context-editing line paper (2026) reports retention under in-context editing is still "around 60%". [V, search summary; exact setting not checked] — [arXiv 2608.25100](https://arxiv.org/html/2608.25100)
- Sequential collapse: forgetting under ROME/MEMIT comes in two phases, gradual then abrupt/catastrophic; ROME's catastrophic point varies from about 100 to about 1000 edits across samples; MEMIT collapses before about 1500 edits on GPT-J and around 4000 on GPT2-XL; MEND forgets all prior edits before 100; "disabling edits" are fundamental to ROME. [V] — [Gupta, Rao, Anumanchipalli, Findings of ACL 2024, arXiv 2401.07453](https://arxiv.org/html/2401.07453v4)
- Butterfly effect: "even a single edit can trigger model collapse"; perplexity of the edited model is proposed and validated as a surrogate that correlates strongly with downstream task performance; the locality metric is "insufficiently effective" for detecting collapse; collapse is common in sequential editing after only a few edits. [V] — [Yang et al., Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.322)
- Long-form evaluation (LEME): edits are normally scored only on the next few tokens; short-form metrics fail to correlate with long-form generative outcomes; ROME is rated worse than no edit on some long-form measures (partial reading). [V, abstract + follow-up description] — [Rosati et al. NAACL 2024](https://aclanthology.org/2024.naacl-long.208/)
- Benchmark provenance [R]: zsRE (question-answering relation extraction, used since KE/MEND) and CounterFact (counterfactual facts with paraphrase and neighbourhood prompts) — [ROME, arXiv 2202.05262](https://arxiv.org/abs/2202.05262); MQuAKE (multi-hop questions whose answers change after an edit) — [arXiv 2305.14795](https://arxiv.org/abs/2305.14795); RippleEdits (logical consequences of an edit) — [Cohen et al., arXiv 2307.12976](https://arxiv.org/abs/2307.12976); KnowEdit and the EasyEdit toolkit — [Zhang et al., arXiv 2401.01286](https://arxiv.org/abs/2401.01286); the four-property frame reliability / generalisation / locality / portability — [Yao et al., arXiv 2305.13172](https://arxiv.org/abs/2305.13172)
- Benchmark weakness: current counterfactual benchmarks have no negative cases, so scope decisions never change the score. [V, abstract] — [INLAY, arXiv 2608.26292](https://arxiv.org/abs/2608.26292)

### Inferences
- Honest positioning of the current RM result: "100% / 97% recall of told sentences" = reliability on the exact prompts, the column where GRACE already scores 0.93-1.00 at 1000 edits on 7B models. The informative columns are generalisation (paraphrased prompts: GRACE 0.02-0.37, WISE 0.67-0.72, MEMOIR 0.85-0.89) and portability (MQuAKE, RippleEdits), neither of which RM has reported.
- RM's locality will not be 1.00 by construction (nothing is gated; "unrelated text drifts slightly toward told content"). Benchmark locality is exact agreement with pre-edit outputs on unrelated prompts, which is stricter than "unrelated text stays as likely under the frozen model". RM's likelihood measure is closer to the Butterfly paper's perplexity surrogate; saying so, and reporting both, avoids an apples-to-oranges claim.
- RM is structurally immune to the Gupta/Butterfly weight-collapse mechanism (no weights are written), which is a fair claim. The analogous risk for RM is additive interference: as the store grows, the summed un-gated pushes grow, so the test to run is locality and fluency versus number of told items (60 → 1000 → 10,000), matching the T axis above.
- Because RM is meant to be used in free generation, LEME-style long-form evaluation is the most relevant existing protocol, and it is one where weight editors look bad.
- Scale mismatch to state plainly: published tables use 7-8B models and 1000-15,000 edits; RM so far is a 0.6B model and 60 sentences.

### Gaps
- No published numbers for IKE under long sequential editing in the same table format were retrieved; IKE is usually reported single-edit, and the "around 60% retention" figure above is a search-summary reading whose setting was not checked.
- CounterFact sequential numbers for GRACE/WISE/MEMIT were not extracted (only zsRE and the hallucination set).
- MQuAKE, RippleEdits, KnowEdit typical scores were not retrieved.
- The alpha/beta/gamma margins of WISE and tau of MEMOIR were not in the text read.

---

## 6. 2025-2026 training-free / test-time-learning memories (Titans, Nested Learning/HOPE, Cartridges, Generative Adapter, Dynamic Cheatsheet, memory layers, others)

### Takeaway
The recent "learning at inference" work either trains a new architecture from scratch with a gradient-updated memory (Titans, HOPE, memory layers), trains something per corpus (Cartridges, Generative Adapter, Memory Decoder), or stays at the text level (Dynamic Cheatsheet, agent memories); none found is a no-training add-on to an off-the-shelf frozen LM that records activation trajectories as its memory. Titans/HOPE do share RM's word "surprise", with a different meaning.

### Cited Findings
- **Titans**: a neural long-term memory module that "learns to memorize at test time"; a new architecture family (three variants) evaluated on language modelling, reasoning, genomics, time series; claims > 2M-token contexts. [V, abstract] — [Behrouz, Zhong, Mirrokni, arXiv 2501.00663](https://arxiv.org/abs/2501.00663). Method detail [R]: surprise = gradient of an associative-memory loss with respect to the memory network's parameters, with momentum ("past surprise") and a forgetting decay; the memory MLP's weights are updated by these gradient steps during inference; the whole architecture is trained from scratch, not attached to a pretrained LM.
- **Nested Learning / HOPE**: Google describes Titans as prioritising memories "based on how surprising they are", with surprise as the local error signal of a data point; HOPE is a self-modifying Titans variant with a "continuum memory system" = modules updating at different frequencies; published at NeurIPS 2025; reports lower perplexity than modern recurrent models and standard transformers. [V] — [Google Research blog, 7 Nov 2025](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning); [OpenReview PDF](https://openreview.net/pdf/9082b3fb3c37bdc2a3b5d69681382ebe783d49e3.pdf)
- **Memory layers at scale** (Meta, Berges et al. 2024) [R]: trainable key-value lookup layers (product-key memory) added to the architecture and trained with the model; not a frozen-model add-on. — [arXiv 2412.09764](https://arxiv.org/abs/2412.09764)
- **Cartridges** [R]: a KV-cache for a corpus is trained offline by "self-study" (gradient descent on synthetic conversations), then loaded at inference; training per corpus. — [Eyuboglu et al. 2025, arXiv 2506.06266](https://arxiv.org/abs/2506.06266)
- **Generative Adapter** [R]: a trained generator maps a context to LoRA-style adapter weights in a single forward pass; no per-context gradient steps, but the generator is trained and weights are (adapter-)modified. — [Chen et al. 2024, arXiv 2411.05877](https://arxiv.org/abs/2411.05877)
- **Dynamic Cheatsheet** [R]: test-time learning by keeping an evolving text memory of strategies that is fed back into the prompt; black-box, no activations. — [Suzgun et al. 2025, arXiv 2504.07952](https://arxiv.org/abs/2504.07952)
- **Memory³** [V, search summary]: converts knowledge into explicit key-value vector pairs injected into attention layers ("explicit memories"); the model is trained to use them. — [Yang et al. 2024, arXiv 2407.01178](https://arxiv.org/pdf/2407.01178)
- **Training-free test-time contrastive learning for LLMs (Apr 2026)** and a 2025 survey of memory operations in LLM agents surfaced in search but are text/prompt-level. [V titles only] — [arXiv 2604.13552](https://arxiv.org/html/2604.13552); [arXiv 2505.00675](https://arxiv.org/pdf/2505.00675)
- Searches for a memory "keyed by binary activation pattern of MLP neurons" and for memory "as recorded activation trajectories" returned no method; the nearest hits were the binary-routing analysis paper and Geva-style key-value views of MLPs. [V] — [arXiv 2603.10985](https://arxiv.org/html/2603.10985)

### Inferences
- "Surprise" collision: Titans' surprise is a gradient with respect to a memory network's parameters and it drives real weight updates in a model trained for it. RM's surprise is the output residual (told − predicted) stored verbatim and never applied to weights. The two share the principle "only prediction error writes" and the paper should cite Titans/HOPE for the principle while stating the mechanical difference.
- RM's intent ("the model is meant to drift/learn from everything it is told rather than recall isolated facts") is the continual-learning framing of Titans/HOPE and dynamic evaluation, not the editing framing of GRACE/WISE. Positioning RM as "test-time learning for a frozen pretrained LM with a non-parametric store" matches its design better than "model editing", and avoids being judged solely on locality = 1.00.
- The unoccupied spot this search found: zero-training, frozen off-the-shelf LM, token-level store of internal activation patterns, applied at logits. The neighbours on each side are kNN-LM (same spot, but final-layer continuous key and targets, 2019) and INLAY (frozen, gradient-free, logit bias, but routed, 2026).

### Gaps
- Titans' and HOPE's equations were not re-read this session (abstract and blog only).
- Memory layers, Cartridges, Generative Adapter, Dynamic Cheatsheet are from recall.
- Work after roughly October 2026 is thinly covered: only what the search engine surfaced (SKIM, ResMem, INLAY). A direct arXiv listing sweep for Nov 2026 was not done.

---

## 7. Which existing method is closest overall, and why

### Takeaway
Closest overall mechanism: kNN-LM (with the continuous cache and last-layer dynamic evaluation as its siblings). Closest within the model-editing literature: GRACE, with MEMOIR closest on the key and INLAY closest on the gradient-free logit-level application. No single source found does all of RM's steps.

### Cited Findings
- kNN-LM shares five of RM's properties: frozen model with "no changes to the underlying LM", datastore from one forward pass, one entry per token, no gate, applied at the output distribution. [V] — [arXiv 1911.00172](https://arxiv.org/pdf/1911.00172)
- GRACE shares: frozen weights, external store keyed on internal activations, edits added one by one; differs by its epsilon deferral gate, trained values, single-layer key and hidden-state replacement. [V] — [arXiv 2211.11031](https://arxiv.org/pdf/2211.11031)
- MEMOIR shares: sparse activation-pattern address compared by Hamming overlap; differs by threshold, single layer, per-prompt mask, gradient-trained residual memory. [V] — [arXiv 2506.07899](https://arxiv.org/html/2506.07899v4)
- INLAY shares: frozen, gradient-free, decode-time bias in unembedding (logit) space; differs by routing and by pushing a target direction rather than a stored prediction error. [V, abstract] — [arXiv 2608.26292](https://arxiv.org/abs/2608.26292)
- Larimar shares the one-shot write and the "episodic memory beside a frozen LM" framing; differs by needing end-to-end pre-training of encoder, memory and decoder. [V] — [arXiv 2403.11901](https://arxiv.org/pdf/2403.11901)

### Inferences
- What already exists, component by component: frozen model + per-token store from one forward pass (kNN-LM, 2019); no gate (kNN-LM, cache); prediction-error-driven output-layer learning at test time (dynamic evaluation, 2017; Hebbian softmax, 2018); activation-pattern Hamming addressing (MEMOIR, 2025); gradient-free logit-space edit on a frozen model (INLAY, 2026); "surprise writes memory" (Titans, 2024/25).
- What was not found anywhere: (1) the key being the binary on/off pattern of every MLP unit in every block, per token; (2) the stored value being the model's own output residual (told − predicted) kept in an external buffer; (3) similarity carried token by token along the stored sequence; (4) these combined with no gate and no optimisation. A novelty claim should be limited to this combination and, most defensibly, to (1) and (2).
- The safest one-line description for a paper: "a kNN-LM-style non-parametric memory whose keys are whole-network binary activation routes and whose values are output-layer gradients, i.e. dual-form last-layer dynamic evaluation on a frozen LM", followed by the GRACE/WISE/MEMOIR comparison on the standard sequential-edit tables.

### Gaps
- Adjacent areas likely covered by other researchers and not searched here: NTK / kernel-regression views of gating patterns, Hopfield / sparse distributed memory (Kanerva) with binary addresses, locality-sensitive hashing of activations, in-context learning as implicit gradient descent. Any of these could contain a closer match on the key (1) than the editing literature does.
- INLAY full text and the Nov 2026 arXiv listings remain the two concrete unread risks.
