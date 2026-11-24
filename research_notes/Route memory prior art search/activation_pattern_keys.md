# Activation patterns as keys, kernels or signatures (prior art for "route memory")

Scope of this note: published work that uses binary/discrete activation patterns of a network as a similarity measure, retrieval key, memory address or kernel, and how each differs from route memory (all-layer binary MLP gate pattern per token, Hamming agreement raised to a power, carried along the stored sequence, used as the kernel for stored logit corrections on a frozen LM).

Search effort: 19 web searches/fetches on 2026-11-24. "Verified" below means I read the primary page or a search result quoting it this session. Items marked **[from memory, not re-verified]** are well-known papers I cite by their canonical link without having re-read them this session; the report writer should treat their details as unverified.

Plain answer up front: **I found no published work that uses the all-layer binary MLP activation pattern of a language model as the retrieval key of a memory.** Every component has a precedent somewhere (binary on/off patterns compared by Hamming distance; all-layer nearest-neighbour lookups; token-level datastores that correct LM logits; pointer-following along stored sequences), but I did not find them combined, and I did not find the binary all-MLP-unit pattern used as a key in any LM memory. Absence in ~19 queries is evidence, not proof.

## 1. Theory: are gate patterns a known kernel (NPF/NPK, linear regions, NTK gate term, path kernel)? Has gate agreement been used for retrieval?

### Takeaway
Yes, gate patterns are a known kernel object: the Neural Path Kernel (Lakshminarayanan & Singh 2020) is built purely from which gates are on, but it counts paths that are ACTIVE FOR BOTH inputs (a product-like overlap across layers, times the input inner product), not the Hamming agreement route memory uses, and it is a theory of training/generalisation, never a retrieval procedure. Hamming distance between ReLU on/off codes is also an established geometric measure (linear-region / polytope literature), again not used as a memory kernel.

### Cited Findings
- Lakshminarayanan & Singh encode the on/off state of the gates for an input as a "neural path feature" (NPF) and the weights as a "neural path value" (NPV); the network output is the inner product of NPF and NPV. — [arXiv 2006.10529](https://arxiv.org/abs/2006.10529)
- The NPK closed form (Lemma 2.1) is H = Sigma ⊙ Lambda, i.e. H(s,s') = <x_s, x_s'> · Lambda(s,s'), where Lambda is the number of paths active for both inputs (a path is active only if every gate along it is on). Only gates ON for both inputs count; off-off agreement is not counted; the paper does not describe it as Hamming-related and gives no retrieval procedure. It does say an active sub-network "can be said to hold the memory for a given input". — [ar5iv 2006.10529](https://ar5iv.labs.arxiv.org/html/2006.10529) (verified by fetch; first 100k characters of the page read)
- The paper states NPF/NPV are zeroth-order quantities while NTF/NTK are first-order; in a deep gated network (DGN) with fixed gates, infinite width, learning the weights equals a kernel method with the NPK, and the NPK equals a constant times the NTK there. — [arXiv 2006.10529](https://arxiv.org/abs/2006.10529); [arXiv 2203.16455](https://arxiv.org/pdf/2203.16455) (follow-up by Lakshminarayanan, Singh, Rajkumar on deep linearly gated networks; search-result summary only)
- Experiments in NPF/NPK are MNIST and CIFAR-10, not language models. — [arXiv 2006.10529](https://arxiv.org/abs/2006.10529)
- Domingos: a model learned by gradient descent is approximately a kernel machine with a "path kernel" = integral over the training trajectory of the dot product of the model's gradients at the two points. "Path" here means the path of the parameters during training, NOT an activation path through the network. — [arXiv 2012.00152](https://arxiv.org/abs/2012.00152)
- Polytope lens (Black, Sharkey et al.): each ReLU neuron is a hyperplane; each region has a binary "spline code" (1 where the neuron is on); adjacent polytopes' codes differ in one neuron; smooth activations such as GELU need values in (0,1) rather than strict binary codes. — [arXiv 2211.12312](https://arxiv.org/abs/2211.12312) (search-result excerpt)
- "ReLU code space": binary activation vectors as points on a hypercube with a truncated Hamming distance d(a,b) = min(d_H(a,b), theta) used as a measure of proximity of linear regions / network quality. — [arXiv 2005.09903](https://arxiv.org/pdf/2005.09903) (search-result excerpt)
- ReLU networks, polyhedral decompositions and persistent homology: Hamming distance between binary activation vectors proposed as a proxy for proximity between polyhedra. — [arXiv 2306.17418](https://arxiv.org/pdf/2306.17418) (search-result excerpt)
- Sign random projections (SimHash): each sign bit agrees with probability 1 − theta/pi, so Hamming distance between sign codes estimates the angle between two vectors. — [DTU LSH lecture notes](https://www2.compute.dtu.dk/courses/02289/2022/approxds/lsh-4x1.pdf); [arXiv 2306.01751](https://arxiv.org/pdf/2306.01751)
- Hanin & Rolnick, "Deep ReLU Networks Have Surprisingly Few Activation Patterns" — counts activation regions; no similarity/retrieval use. **[from memory, not re-verified]** — [arXiv 1906.00904](https://arxiv.org/abs/1906.00904)

### Inferences
- Route agreement (share of units whose gate sign agrees) is NOT the NPK. NPK = input inner product × count of co-active paths (multiplicative across layers, on-on only). Route similarity = additive Hamming agreement across all units, on-on and off-off both counted, then raised to a power. Raising agreement to a power moves it somewhat toward the NPK's multiplicative behaviour, but they remain different functions. The honest statement is: "gates as a kernel" is known theory (NPK); "Hamming agreement of gates as a retrieval kernel" was not found.
- By the SimHash result, the Hamming agreement of gate signs within ONE layer is an estimate of an angular similarity between that layer's inputs under that layer's (trained, not random) projection directions. So a sceptical reviewer can say: route similarity is an LSH-style binarised angular similarity over the concatenated MLP inputs of all blocks. That is a fair reduction of the similarity measure itself; it does not reduce the use (memory key in an LM) to prior work.
- The NTK of a ReLU net contains a term equal to the fraction of units active for both inputs (the derivative/arc-cosine degree-0 term); I could not verify a citable statement of this in this session (see Gaps).

### Gaps
- Cho & Saul 2009 arc-cosine kernel and the exact NTK "gate term" statement were not retrieved; the search returned nothing on them. Check primary sources before citing.
- I did not read the last ~30k characters of the NPF/NPK ar5iv page (appendix); a retrieval-style experiment there is unlikely but not excluded.
- No source found that proves or disproves that Hamming agreement of trained gate patterns is positive semi-definite as used (it is PSD as a kernel on bit vectors; elementwise powers of PSD kernels with integer exponent stay PSD by the Schur product theorem — this is my reasoning, not a cited result).

## 2. Gated Linear Networks (Veness et al.) and Gaussian GLNs

### Takeaway
GLNs select weights by a binary context code computed with half-space gating and learn those weights locally online; the paper trail explicitly links half-space gating to SimHash LSH. It is the closest "select what to use by a binary gating pattern" architecture, but the gates are on side-information in a purpose-built network trained online, not the activation pattern of a frozen pretrained transformer used as a memory key.

### Cited Findings
- A GLN neuron is a gated geometric mixer: contextual gating maps particular inputs to particular sets of weights; neurons model nonlinear functions via data-dependent gating plus online convex optimisation. — [arXiv 1910.01526](https://arxiv.org/abs/1910.01526); [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17202)
- Half-space gating: within each polytope formed by the intersecting half-spaces, the gated weights are constant. — [arXiv 2002.11611](https://ar5iv.labs.arxiv.org/html/2002.11611) (Gated Linear Contextual Bandits; search-result excerpt)
- The Gated Linear Contextual Bandits paper says its approach "relies on the close connection between half-space gating and the SimHash variant of locality-sensitivity hashing". — [arXiv 2002.11611](https://ar5iv.labs.arxiv.org/html/2002.11611) (search-result excerpt)
- Original online-learning formulation. — [arXiv 1712.01897](https://arxiv.org/abs/1712.01897)

### Inferences
- GLN = binary gating code → index into a weight table, learned online by local gradient steps. Route memory = binary gating code of a frozen LM → soft similarity to stored tokens → add stored logit corrections. Shared idea: a binary region code decides which stored quantity applies. Differences: (a) GLN's code is a hard index per neuron, route memory's is a soft Hamming kernel over the whole network; (b) GLN trains weights, route memory stores per-token corrections with no training; (c) GLN is not a transformer LM add-on.

### Gaps
- I did not find a source that states a nearest-neighbour or kernel reading of GLNs explicitly; the SimHash link is the nearest citable statement.
- Gaussian GLN paper not retrieved this session.

## 3. Activation patterns as hashes/keys: LSH, neural codes, semantic hashing, Deep kNN, runtime monitors

### Takeaway
Binary on/off activation patterns stored from training data and compared to a new input's pattern by Hamming distance is published prior art (Cheng, Nührenberg & Yasuoka 2018; Olber et al. 2022), but for runtime monitoring / out-of-distribution detection in image classifiers, on selected layers, with no output correction and no language model. Deep k-NN (Papernot & McDaniel 2018) is the precedent for nearest-neighbour lookups at EVERY layer, on continuous activations, for classification confidence.

### Cited Findings
- Cheng, Nührenberg & Yasuoka, "Runtime Monitoring Neuron Activation Patterns": after training, feed the training data through the network and store the on/off neuron activation patterns (in binary decision diagrams); at run time, if the monitor holds no similar pattern it warns that the decision is not supported by training data; the stored set is enlarged to all patterns within a Hamming-distance threshold of existing ones. — [arXiv 1809.06573](https://arxiv.org/abs/1809.06573)
- Olber et al., "Detection of out-of-distribution samples using binary neuron activation patterns": extract binary activation patterns from hidden (convolutional) layers of ReLU nets and compare with those collected in training via Hamming distance to flag OOD; evaluated on seven image datasets; it criticises the Cheng et al. approach as unscalable. — [arXiv 2212.14268](https://arxiv.org/abs/2212.14268) (abstract fetched; Hamming-distance detail from search-result summary of the paper body)
- Deep k-Nearest Neighbors: for each layer of the DNN, a nearest-neighbour search finds training points whose layer output is closest to the test input's; neighbour labels give conformal confidence/credibility; a prediction gets high conformity only if it agrees with training data at all representation levels. — [arXiv 1803.04765](https://arxiv.org/abs/1803.04765)
- Learning-to-hash methods rank by Hamming distance over compact learned binary codes for approximate nearest-neighbour search. — [arXiv 2112.04014](https://arxiv.org/pdf/2112.04014) (Neural Activation Coding; search-result excerpt); [arXiv 1810.01008](https://arxiv.org/pdf/1810.01008)
- Recent LLM-efficiency work replaces floating-point key similarity in attention with Hamming distance over binary hash codes of KV-cache keys (DASH-KV). These are learned/random hashes of attention keys for speed, not MLP gate patterns and not an external memory. — [arXiv 2604.19351](https://arxiv.org/abs/2604.19351) (search-result excerpt)
- Babenko et al. "Neural Codes for Image Retrieval" (activations of a CNN layer as retrieval descriptors, continuous). **[from memory, not re-verified]** — [arXiv 1404.1777](https://arxiv.org/abs/1404.1777)
- Reformer (LSH attention: hash queries/keys with random projections to bucket attention) **[from memory, not re-verified]** — [arXiv 2001.04451](https://arxiv.org/abs/2001.04451); SLIDE (LSH to pick which neurons to compute) **[from memory, not re-verified]** — [arXiv 1903.03129](https://arxiv.org/abs/1903.03129)

### Inferences
- The single closest precedent for the similarity measure is Cheng et al. 2018 / Olber et al. 2022: store binary on/off patterns of a ReLU net, compare a new input's pattern to stored ones by Hamming distance. Differences from route memory: vision classifiers, per-input (not per-token) patterns, a subset of layers, the output is an accept/reject flag, nothing is retrieved or added to the model's output, no sequence carry.
- Deep kNN is the closest precedent for "use every layer for the lookup": continuous activations, one kNN per layer, class labels of neighbours; no binarisation, no LM, no logit correction.
- Semantic hashing / deep hashing / neural codes all LEARN a code for retrieval or use one layer's continuous activations; route memory uses the network's native gate signs with no learned code. Reformer/SLIDE/MONGOOSE hash for compute savings, not as memory keys.

### Gaps
- Semantic hashing (Salakhutdinov & Hinton 2009), MONGOOSE, and "activation atlas" were not retrieved this session; no sourced statements for them.
- I did not find any binary-key variant of kNN-LM/kNN-MT (datastore compression there is reported as product quantisation and PCA: [arXiv 2301.02828](https://arxiv.org/pdf/2301.02828), [arXiv 2406.06073](https://arxiv.org/pdf/2406.06073)); one more targeted search of the ACL Anthology would firm this up.

## 4. Activation patterns as signatures in NLP/LLMs (binary MLP routing, key-value memories, SAE features as keys, neuron-level detectors, contextual sparsity)

### Takeaway
In LLMs, binary MLP firing patterns are studied descriptively (Balogh, March 2026, "binary routing" in GPT-2) and neuron activations are used as detectors (hallucination, pre-training-data membership), but none of these uses the pattern as a similarity measure between tokens or as a memory key. The LM-memory line (kNN-LM, RetoMaton) keys on a continuous final-layer hidden state. This is the gap route memory sits in.

### Cited Findings
- "The Discrete Charm of the MLP: Binary Routing of Continuous Signals in Transformer Feed-Forward Layers" (Peter Balogh, arXiv 11 March 2026): GPT-2 Small on WikiText-103; GELU outputs thresholded at 0.1 to fire/not-fire; analyses a 7-neuron "consensus" set and 8-bit patterns of the top-8 discriminative layer-11 neurons, plus cross-layer analysis. Patterns are a descriptive tool: no Hamming distance between token patterns, no retrieval or memory application. Binarisation loses little for the routing decision (79.2% binary vs 78.8% continuous). Consensus structure did not replicate cleanly in GPT-2 Medium/Large. — [arXiv 2603.10985](https://arxiv.org/html/2603.10985v1) (verified by fetch)
- kNN-LM: datastore key = the LM's output-layer hidden vector for a left context, value = the next token; prediction interpolates the LM distribution with a softmax over negative distances to retrieved neighbours; no further training. — [arXiv 2301.02828](https://arxiv.org/pdf/2301.02828) (Xu, Alon, Neubig analysis describing kNN-LM)
- Geva et al. view of feed-forward layers as key-value memories (keys = neuron activations triggered by input patterns, values = output vectors); a NeurIPS 2025 paper finds FF key-value neurons "nearly as interpretable" as SAE features. This is a reading of the weights, not an external store keyed by patterns. — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/120144); [project page](https://muyo8692.github.io/projects/ff-kv-sae/)
- S3-Attention (2026 preprint): trains top-k SAEs to turn attention key/query projections into discrete sparse feature IDs and uses feature co-activation (optionally with BM25) as a searchable index to recall evidence spans for long-context inference. Discrete activation features as a retrieval index in an LLM — but trained SAEs over attention states, retrieving text spans. — [arXiv 2601.17702](https://arxiv.org/html/2601.17702v2) (search-result summary only; method not read in full)
- NA-PDD (EMNLP 2025): labels neurons "member"/"non-member" by how often they fire on training vs non-training text to detect pre-training data; needs a labelled calibration set. A membership detector, not a memory. — [arXiv 2507.16414](https://arxiv.org/pdf/2507.16414)
- H-Neurons and related neuron-level hallucination detectors use continuous neuron contributions with a trained sparse classifier; the search found no LLM OOD/hallucination method using binarised patterns with Hamming distance. — [arXiv 2512.01797](https://arxiv.org/abs/2512.01797); [arXiv 2609.14448](https://arxiv.org/pdf/2609.14448)
- Deja Vu: "contextual sparsity" = small input-dependent sets of heads and MLP neurons that reproduce the dense output; up to 85% sparsity; trained lookahead predictors guess which neurons will be active, for speed. Predicts the pattern; does not use it as a key. — [arXiv 2310.17157](https://arxiv.org/abs/2310.17157)
- A small-transformer polyhedral study counts unique MLP activation patterns (layer-0: 67,313; layer-1: 145,834) and calls the MLPs "a distributed quantisation". Descriptive only. — [arXiv 2605.24577](https://arxiv.org/pdf/2605.24577) (search-result excerpt)
- "What Survives When Activation Steering Is Used as Memory" (Oct 2026): a training-free memory beside a frozen LLM that stores changes in late-layer states under a key that is a whitened, normalised summary of the conversation gist. Training-free activation-side memory, but the key is a continuous residual summary, not a binary MLP pattern. — [arXiv 2610.14958](https://arxiv.org/abs/2610.14958) (search-result summary only; NOT verified from the paper)
- MLP Memory (2025): an MLP pretrained to mimic non-parametric retrieval distributions; requires training. — [arXiv 2508.01832](https://arxiv.org/pdf/2508.01832)

### Inferences
- Balogh 2026 is the closest LLM-side paper on the object itself (binary MLP firing patterns in a transformer LM), and it stops at description: a handful of neurons in one layer, no cross-token similarity, no memory. Route memory uses every MLP unit of every block as one key and builds a memory on it. Cite Balogh as supporting evidence that binary patterns carry most of the routing information, not as anticipation.
- kNN-LM is the closest on function (token-level datastore from a frozen LM, no training, next-token distribution corrected from neighbours). Differences: key is one continuous final-layer vector with L2 distance; value is the next token, interpolated as a probability, where route memory stores a logit correction (told minus the model's own stored prediction) and weights it by a Hamming kernel carried along the sequence.
- So the defensible novelty claim is narrow: "the key is the all-layer binary MLP gate pattern, compared by Hamming agreement". The memory architecture around it (token datastore, non-parametric correction of a frozen LM) is kNN-LM-family prior art and should be stated as such.

### Gaps
- Dai et al. "Knowledge Neurons", Sukhbaatar et al. "persistent memory", the "lazy neuron" paper and Geva et al.'s original paper were not re-read this session; their relevance is as background only.
- S3-Attention and arXiv 2610.14958 were seen only as search summaries; their exact key construction should be read before the report says how they differ.
- I could not search OpenReview submissions directly; an under-review 2026 submission with this idea would not have shown up.

## 5. Episodic / activation replay; sparse distributed memory addressed by binary codes

### Takeaway
Kanerva's Sparse Distributed Memory is the classical "memory addressed by Hamming distance between binary codes", and Bricken & Pehlevan show attention approximates it; route memory can fairly be described as an SDM-style read whose addresses are the model's own gate patterns. I found no work that instantiates SDM with a network's activation pattern as the address in an LM.

### Cited Findings
- SDM works over n-dimensional binary vectors; each memory has an address and a pointer; reads retrieve the pointers of all neurons within a Hamming distance d of the query and average them; this hypersphere-intersection weighting approximates the exponential of attention's softmax. Checked on pre-trained GPT-2 ("a weak yes"). — [arXiv 2111.05498](https://arxiv.org/abs/2111.05498); [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/8171ac2c5544a5cb54ac0f38bf477af4-Abstract.html)
- The search found no 2024–2026 work applying a binary-key SDM as an external memory for LLMs. — [search result set headed by arXiv 2111.05498](https://arxiv.org/abs/2111.05498)
- A 2026 paper uses kNN-LMs as an episodic-retrieval mechanism and notes this is only a functional analogy to hippocampal memory. — [arXiv 2608.23851](https://arxiv.org/pdf/2608.23851) (search-result excerpt)

### Inferences
- "sim^scale" on a Hamming agreement is functionally the same family as SDM's read weighting (weight falls off roughly exponentially with Hamming distance); a reviewer who knows SDM will read route memory as "SDM/attention-style kernel regression with gate patterns as addresses". That is a framing risk, not an anticipation: SDM's addresses are given binary vectors, not a pretrained network's activation pattern.
- Activation patching / reinstatement literature stores and re-inserts activations to study causality; it does not use them as lookup keys. (No source retrieved; general knowledge.)

### Gaps
- Bricken et al. 2023 (SDM as a continual learner), hippocampal-replay LLM memories and "trajectory memory" papers were not retrieved; nothing sourced on them.

## 6. Choosing the next token by matching its predicted activation pattern (the "through the weights" variant)

### Takeaway
I found nothing that picks the next token by computing, for every vocabulary token, the first-block activation pattern it would produce and matching that against a stored pattern. The nearest things are forward candidate scoring (CoDec) and the logit lens, which go activation → token through the output head, not through predicted input-side patterns.

### Cited Findings
- CoDec-style guided decoding runs candidate tokens through the model to get their activations and scores candidates by an activation-based confidence; a forward search over candidates, for factuality, not a match against a stored target pattern. — [arXiv 2406.13230](https://arxiv.org/html/2406.13230) (search-result summary)
- Logit-lens style decoding projects a hidden state through the output head and takes the argmax; maps activation to token, does not search for the input token that would produce a target activation. — [arXiv 2606.11386](https://arxiv.org/pdf/2606.11386) (search-result excerpt describing the standard definition)
- LatentQA trains a decoder to answer questions about an activation; trained, and produces text about the activation rather than selecting a token by pattern match. — [arXiv 2412.08686](https://arxiv.org/pdf/2412.08686)

### Inferences
- This variant appears unpublished as stated. The closest general family is embedding/hidden-state inversion (find the input that produces a given activation), which is usually gradient-based or a trained inverter and is used for privacy attacks, not decoding from a memory. (No source retrieved for inversion work; general knowledge.)

### Gaps
- No search was run on "embedding inversion" / "hidden-state inversion" specifically; that literature should be checked before claiming the variant is new.

## 7. Sequence-carried similarity (P_t[n] = sim_t[n]^scale · (1 + P_(t-1)[n-1]))

### Takeaway
Following stored sequences rather than doing an independent lookup per token is prior art in RetoMaton (pointers from each datastore entry to its successor); I found no published soft recurrence of route memory's form over activation patterns, and no DTW/string-kernel over hidden-state sequences used as an LM memory kernel.

### Cited Findings
- RetoMaton: datastore entries are (key, value, pointer) with key = LM hidden state, value = target token, pointer = the entry's successor in the text; close keys are clustered into automaton states; traversing the automaton in parallel with LM inference lowers perplexity by up to 1.85 or saves up to 83% of nearest-neighbour searches versus kNN-LM. — [arXiv 2201.12431](https://arxiv.org/abs/2201.12431); [ICML 2022](https://icml.cc/virtual/2022/poster/16305)
- A 2025 follow-up builds a local, task-adaptive weighted finite automaton from domain corpora, fine-tuning-free. — [arXiv 2508.19271](https://arxiv.org/html/2508.19271v1)
- The search returned nothing on sequence alignment of stored hidden states or on Grave et al.'s continuous cache. — [search result set headed by arXiv 2201.12431](https://arxiv.org/abs/2201.12431)

### Inferences
- The recurrence is a gap-free local-alignment score: position n's weight grows with the run of consecutive good matches ending at n (a product of similarities along the diagonal, summed over run lengths). That is the structure of a contiguous-substring (n-gram/suffix) string kernel and of Smith-Waterman without gaps. RetoMaton reaches a similar effect with hard pointers. The soft, multiplicative diagonal carry over activation-pattern similarity is, as far as I found, not published; but a reviewer may call it "a soft RetoMaton" or "a suffix-match weighting", so it should be claimed as a design detail, not as the headline.

### Gaps
- Continuous cache (Grave et al. 2016, [arXiv 1612.04426](https://arxiv.org/abs/1612.04426) **[from memory, not re-verified]**), pointer-sentinel models, infinity-gram / suffix-array LMs and string kernels were not retrieved; the comparison above to string kernels is my own reasoning.

## 8. Direct answer: has anyone used all-layer binary MLP activation patterns as the retrieval key of a language-model memory?

### Takeaway
Not that I could find. The closest three, and exactly how they differ: (1) Cheng et al. 2018 / Olber et al. 2022 — stored binary on/off patterns compared by Hamming distance, but in image classifiers for OOD flags, nothing retrieved or corrected; (2) kNN-LM / RetoMaton — token-level training-free memory correcting a frozen LM, but keyed on one continuous final-layer vector; (3) NPF/NPK — gates as a kernel, but co-active path counting in theory of training, no retrieval. Balogh 2026 shows binary MLP patterns in GPT-2 carry the routing information but builds no memory.

### Cited Findings
- Binary on/off patterns + Hamming distance to stored patterns: [arXiv 1809.06573](https://arxiv.org/abs/1809.06573), [arXiv 2212.14268](https://arxiv.org/abs/2212.14268) — vision, OOD only.
- Token datastore on a frozen LM, continuous final-layer key: [arXiv 2301.02828](https://arxiv.org/pdf/2301.02828) (describing kNN-LM), [arXiv 2201.12431](https://arxiv.org/abs/2201.12431).
- Gates as kernel: [arXiv 2006.10529](https://arxiv.org/abs/2006.10529) — on-on path overlap, no retrieval.
- Binary MLP patterns in a transformer LM, descriptive: [arXiv 2603.10985](https://arxiv.org/html/2603.10985v1) — no Hamming similarity, no memory.
- A targeted 2026 query for all-layer MLP firing patterns as datastore keys in a frozen LLM returned no match; the nearest hit keys on a continuous gist vector. — [arXiv 2610.14958](https://arxiv.org/abs/2610.14958) (summary only)

### Inferences
- Likely-safe claim: "To our knowledge, no prior work uses the binary gate pattern of all MLP units across all layers of a pretrained LM as the key of a non-parametric memory."
- Claims to avoid: that Hamming similarity of activation patterns is new (it is not: Cheng 2018, Olber 2022, polytope/ReLU-code work); that gates-as-kernel is new (NPK); that a training-free token datastore correcting a frozen LM is new (kNN-LM); that sequence-following retrieval is new (RetoMaton).
- Expected reviewer question: why binary all-layer patterns rather than kNN-LM's final hidden vector — the paper would need a head-to-head against kNN-LM/RetoMaton keys on the same stored passes.

### Gaps
- Negative result from ~19 queries; not exhaustive. Not searched: OpenReview full text, Google Scholar citation graphs of Cheng 2018 / NPK / Balogh 2026 (the most likely places a close match would cite), patents, and non-English venues.
- Papers citing arXiv 2603.10985 since March 2026 were not checked; a follow-up building a memory on binary routing is the most plausible place for a late-2026 collision.
