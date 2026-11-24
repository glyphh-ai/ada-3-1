# Findings

## Sim on what differs (2026-11-24)

`edit_test.py published centered`, as written in PROTOCOL.md. Calibration picked
the `global` version (scale 40, step 300,000); third scoring of the same 1,000
test edits.

| method | reliability | generalization | locality |
|---|---|---|---|
| route memory, sim centred on each unit's usual firing | 1.00 | 0.79 | 0.98 |
| route memory, plain sim, larger step | 1.00 | 0.77 | 0.99 |
| route memory, plain sim, first matched run | 1.00 | 0.72 | 0.99 |
| MEMOIR (published, LLaMA-3-8B) | 0.94 | 0.85 | 1.00 |

Exact: 0.9994 / 0.7933 / 0.9816 (calibration at the pick 0.998 / 0.744 / 0.981;
the plain sim's best calibration point was 0.999 / 0.724 / 0.987). A small gain:
+0.02 generalization for -0.01 locality. Removing the global usual firing does
not separate the questions much better.

The `token` version (usual firing taken per token) is not a result: it scored
0.28 reliability on calibration because of a numerical fault of mine. A token
whose route is the same every time it is told (the first word of a sentence)
has nothing left after centring, and its cosine became noise over nothing, read
as a perfect match. Fixed with a floor on the centred length; rerun on
calibration only (`published centered token-calibration`) before the test edits
are touched again.

## Where the similarity is lost on reworded questions (2026-11-24)

`diagnose.py`, calibration records only (1,000 told, each reworded question
traced at scale 80, step 300,000; 2,883 answer tokens).

1. **It is the first answer token.** Right with memory: first token 53.2%,
   second 92.6%, third or later 94.2% (model alone 3.6% / 40.8% / 61.0%). Once
   the answer has started, the carry through the answer's own tokens holds.
2. **It is finding, not pushing.** At the first token the right place has the
   largest weight in 42.9% of questions (then the token is right 83.0%); in the
   other 57.1% another memory's place is larger (right 30.8%). Of the 468
   first-token failures, 84.4% have the right place not on top; its median
   rank is 4 of 13,791, with 1.8% of all weight (46.3% when it succeeds).
3. **Rewording breaks the carry.** In 635 of 1,000 questions the only closing
   token the reworded question shares with the told one is the last (`?`):
   right place on top 28.8%, token right 43.6%. With 7 or more closing tokens
   unchanged: 77.1% and 81.4%. The carry runs token against token at the same
   offset, so inserted or reordered words leave it only the final token.
4. **One token cannot tell the questions apart well enough.** Last question
   token against the told question's last token, no carry: sim to the right
   place 0.877, to the closest other memory 0.893; the right place is the
   closest in 42.9%. By blocks: 1-8 36.1%, 9-16 37.2%, 17-24 38.2%, 25-32
   48.5%. All sims sit in a narrow band (0.85-0.92): most units fire the same
   for every question's last token, and the part that differs between
   questions is a small share of the route, strongest in the last blocks.

## The step curve, continued (2026-11-24)

`edit_test.py published steps`, as written in PROTOCOL.md. Same model, same
1,000 test edits, the grid continued upward on the same calibration records
(picked scale 80, step 300,000). Second scoring of the test edits; the first is
the section below.

| method | reliability | generalization | locality |
|---|---|---|---|
| route memory, surprise (scale 80, step 300,000) | 1.00 | 0.77 | 0.99 |
| route memory, surprise (first matched run: scale 60, step 3,000) | 1.00 | 0.72 | 0.99 |
| MEMOIR (published, LLaMA-3-8B) | 0.94 | 0.85 | 1.00 |
| WISE (published, LLaMA-2-7B) | 0.77 | 0.72 | 1.00 |

Exact: 1.0000 / 0.7736 / 0.9897 (calibration at the pick: 0.999 / 0.724 /
0.987). Generalization is now above WISE and 0.08 under MEMOIR.

The step is running out: on calibration, generalization flattens near 0.74-0.76
however large the step (scale 50: 0.71, 0.73, 0.74, 0.74, 0.74; scale 60 ends
at 0.76 with locality down to 0.86; scale 80 still creeps up, 0.70 -> 0.72 at
the last tripling). The rest of the gap is not a strength setting: for about a
quarter of answer tokens the reworded question's route does not carry enough
similarity to the told question for any step to help.

## The editing table, matched to the published setup (2026-11-24)

`edit_test.py published`, as written in PROTOCOL.md, run once. Full-precision
LLaMA-3-8B-Instruct (BF16, MLX), raw prompts, the published runs' own edit set
(first 1,000 records of `zsre_mend_edit.json`, same order), all 1,000 told
first (one forward pass each, nothing trained, no weights changed), scored with
a line-for-line port of EasyEdit's `test_prediction_acc`. Settings picked on
1,000 calibration records sharing no question or subject with the edit set
(scale 60, step 3000).

| method | reliability | generalization | locality |
|---|---|---|---|
| route memory, surprise (this run) | 1.00 | 0.72 | 0.99 |
| model alone (this run) | 0.31 | 0.30 | 1.00 |
| MEMOIR (published, LLaMA-3-8B) | 0.94 | 0.85 | 1.00 |
| WISE (published, LLaMA-2-7B) | 0.77 | 0.72 | 1.00 |
| GRACE (published, LLaMA-2-7B) | 0.93 | 0.08 | 1.00 |

Exact: 0.9994 / 0.7164 / 0.9916. Calibration at the picked setting gave 0.998 /
0.653 / 0.990. Reliability is the best in the table; generalization equals
WISE's and is 0.13 under MEMOIR's; locality is 0.99 with no gate.

The pick is again the largest step in the grid: at scale 60, calibration
generalization rises with every step (0.38, 0.45, 0.52, 0.59, 0.65) while
locality stays at 0.99, so the curve is not finished. Scale 40 reaches 0.69
generalization at 0.75 locality; scale 20 and 30 lose locality.

Still not matched: the scoring is my port, not EasyEdit run on a CUDA machine;
the other methods' rows are their published numbers, not reruns; WISE and GRACE
are LLaMA-2-7B rows.

## The editing table, first run (4-bit model, a different zsRE file) (2026-11-24)

`edit_test.py`, as written in PROTOCOL.md, run once. zsRE, 1,000 edits told at
once (one forward pass each, nothing trained, no weights changed), LLaMA-3-8B-
Instruct 4-bit, `next_surprise`, route sim on 32,768 sampled units. Settings
picked on calibration edits 1000-1999 (scale 40, step 300); test edits 0-999.

| method | reliability | generalization | locality |
|---|---|---|---|
| route memory, surprise (this run, 4-bit model) | 1.00 | 0.62 | 0.93 |
| model alone (this run) | 0.26 | 0.25 | 1.00 |
| MEMOIR (published, LLaMA-3-8B) | 0.94 | 0.85 | 1.00 |
| WISE (published, LLaMA-2-7B) | 0.77 | 0.72 | 1.00 |
| GRACE (published, LLaMA-2-7B) | 0.93 | 0.08 | 1.00 |

Reliability is at the top of the table, generalization (reworded question) is
0.62: well above GRACE and the model alone, below WISE and MEMOIR. Locality is
0.93 where the gated methods report 1.00. Calibration gave the same picture
(1.00 / 0.60 / 0.94), so the settings carried over.

Calibration shows the trade: at scale 40, raising the step from 3 to 300 lifts
generalization 0.29 -> 0.60 and lowers locality 1.00 -> 0.94; the pick is the
largest step in the grid, so the curve is not finished. Scale 80 keeps locality
at 1.00 with generalization 0.26-0.39; scale 20 and below break locality, and
scale 5-10 break everything.

Limits of the comparison: the published rows are full-precision models scored
by EasyEdit itself; this run is a 4-bit model scored by my own version of the
same teacher-forced token accuracy, with raw `question answer` prompts. The
other methods were not rerun here.

## Earlier, on Qwen3-0.6B

## Honest test, surprise (2026-11-24): works

`honest2.py`, `next_surprise` (one backprop step on the output weights per told
token, applied beside the weights: `scores = log p + eta * sum P * (told -
predicted)`; scale 40, eta 10 set on the demo's five sentences), seed 707, run
once, 60 told prompts at once, 10 tokens generated.

| | with memory | model alone |
|---|---|---|
| told fact comes back (60) | 100.0% | 0.0% |
| unrelated openings: generation differs from the model alone (60) | 15.0% | 0.0% |
| unrelated openings: model's own log probability per token of the text | -1.13 | -1.13 |
| the same, only the 9 that differ | -1.25 | -1.28 |

Every kind 10 of 10. The nine unrelated generations that changed stayed logical
by the frozen model's own measure (-1.25 against -1.28 for its own text) and
read as drift toward what was told, not noise: `She had a lot` became `She has
a bag`, `Lily has 12 marbles` became `Lily has 1200 dollars`. With the pull on
routes (below) the same 60 told prompts changed every unrelated generation and
garbled many.

### 60 tokens generated instead of 10 (`honest2.py 60 808`, new seed, run once)

| | with memory | model alone |
|---|---|---|
| told fact comes back (60) | 96.7% | 3.3% |
| unrelated openings: generation differs from the model alone (60) | 80.0% | 0.0% |
| unrelated: tokens written before the first difference (median) | 22 of 60 | |
| unrelated: log probability per token, tokens 1-10 | -1.26 | -1.21 |
| unrelated: log probability per token, tokens 11-30 | -0.88 | -0.89 |
| unrelated: log probability per token, tokens 31-60 | -0.57 | -0.54 |
| told: log probability per token, tokens 1-10 | -2.25 | -1.16 |
| told: log probability per token, tokens 11-30 | -0.77 | -0.92 |
| told: log probability per token, tokens 31-60 | -0.30 | -0.48 |

The told fact came back 58 of 60 (the two misses are train platforms, `2`
written for 14 and for 10). Over 60 tokens most unrelated generations part from
the model's own at some point (48 of 60, typically around token 22) and stay as
logical as its own by its own measure at every stretch. After a told fact the
text carries on about it (`potter in Lusaka. He has a potter's wheel with a
radius of 12 feet.`). Both the model alone and the model with memory fall into
repeating a sentence, which greedy decoding on this small model does anyway;
one memory generation ran into a string of zeros.

Not tested yet: more than 60 told prompts, a larger model.

## Honest test, no gate, pull on routes (2026-11-23)

`honest.py` with `next_sim` (no cutoff, no reset, no threshold; scale 40, pull
3000 set on the demo's five sentences), seed 606, run once, 60 told prompts.

| asked | with memory | model alone |
|---|---|---|
| same start: told fact in the generation (60) | 100.0% | 1.7% |
| lead-in: told fact in the generation (60) | 16.7% | 0.0% |
| unrelated: generation differs from the model alone (60) | 100.0% | 0.0% |

On the demo's five told sentences the same settings recalled every prompt,
lead-ins included. With 60 told at once, the same start still works 60 of 60,
but after a lead-in the fact comes back 10 of 60 and the text is often broken
(`singer in TELUS. He has a car`): every one of about 900 places always pulls,
and their sum is twelve times what it was with five sentences.

## Honest test, gated carried sim (2026-11-23)

`honest.py` with the sim carried token by token (`next_carry`; THETA 0.85, kappa
100, c0 0.10 set on the demo sentences), seed 505, run once, 60 told prompts.

| asked | with memory | model alone |
|---|---|---|
| same start: told fact in the generation (60) | 100.0% | 0.0% |
| lead-in: told fact in the generation (60) | 100.0% | 0.0% |
| unrelated: generation changed (60) | 50.0% | 0.0% |

Lead-ins no longer matter: 60 of 60 (the windowed version below got 34 of 60),
every kind 10 of 10. The cost is on unrelated text: 30 of 60 story openings now
change (the windowed version changed 2), 6 of the 8 that end a sentence and 24
of the 52 cut mid-sentence. The changes come a few tokens in, where a short run
of ordinary words matches a told sentence and pulls its words in (`there was a
boy who was very young and he was a midwife`). Observation only, not a
requirement: a never-told subject of the same kind of sentence gets a told fact
30 of 30 times.

## Honest test of the demo, windowed sim (2026-11-23)

`honest.py`, as written in PROTOCOL.md, settings fixed at the demo's, seed 404,
run once. Qwen3-0.6B, 60 told prompts at once.

| asked | with memory | model alone |
|---|---|---|
| same start: told fact in the generation (60) | 100.0% | 0.0% |
| lead-in: told fact in the generation (60) | 56.7% | 0.0% |
| near miss: generation changed (30) | 93.3% | 0.0% |
| near miss: given a told fact (30) | 93.3% | 0.0% |
| unrelated: generation changed (60) | 3.3% | 0.0% |

Lead-in right out of 10 by kind: job 9, train 10, food 8, password 7, colour 0,
pet 0 (the two kinds whose told sentence starts with `The` / `My`).

- Told prompts do alter later generation: asked with the same start, the told
  fact comes back 60 of 60, and unrelated text is left alone (2 of 60 changed).
- It does not know whose memory it is: a subject that was never told gets
  someone else's fact 28 of 30 times (`Talia Jankowski works as a` -> `midwife
  in Ljubljana`, Dorian Rasmussen's).
- After a lead-in it recalls 34 of 60, and never for the kinds whose sentence
  starts differently once something is put in front of it.

Each line of the run is in `honest_results.json`.

## Test 1 of the kNN-LM comparison

## Test 1 (2026-11-23): FAIL

`run.py`, exactly as in PROTOCOL.md, run once. Qwen3-0.6B-4bit, 200 memories.

Calibration picked, for every memory system, a setting that does nothing:
kNN-LM lam 0.1 tau 0.3 (-5.0%), weighted kNN-LM beta 1 tau 0.03 (-5.0%), route
memory gamma 0.5 scale 1000 beta 100 (+0.0%). No setting of any of them scored
above zero on calibration.

Locked test, seed 202:

| system | score | lead-in | reworded | paraphrase | other person changed | unrelated changed |
|---|---|---|---|---|---|---|
| model alone | +0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| kNN-LM | -5.8% | 0.0% | 0.0% | 0.0% | 8.3% | 3.3% |
| kNN-LM, weighted like the route memory | -4.2% | 0.0% | 0.0% | 0.0% | 5.0% | 3.3% |
| route memory | +0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| (reference) all memories in the prompt | -78.6% | 26.7% | 28.3% | 6.7% | 100.0% | 98.3% |

Route memory minus the better kNN-LM: +4.2 points, bar +10: **FAIL**. Nothing
recalled anything: with 200 memories that all share one sentence frame, every
setting that recalled a person's job and city on calibration changed at least as
many control generations, so the best choice was to stay silent. Even with all
200 sentences in its prompt, this model gave the right job and city only 27%,
28% and 7% of the time.

The run did not keep the calibration numbers per setting, so it does not say
whether any setting recalls at all at this size. `run.py cal` reruns calibration
only and writes them to `calibration.json`.
