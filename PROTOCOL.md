# version_24: route memory on Qwen, whole generations

Written before the test ran (2026-11-23). The idea, its history on the small
model and every earlier result are in `openGPT/research/design.md`.

## What is tested

The route memory as built in `route.py` (a memory is which MLP units fire at every
token of a told sentence; every next token is pulled by every memory place with
weight `exp(-scale * delta)`, never switched off), moved to a real model and
scored on whole generations, against kNN-LM given the same chances.

## Checked before the test (`check.py`, plumbing only)

The hand-run pass matches the model (largest log-probability gap 0.065, 16-bit
arithmetic). Scoring only a short list of candidates exactly picked the same token
as scoring every token exactly, 24 of 24. One decision takes 15 ms on Qwen3-0.6B.
Model: **Qwen3-0.6B-4bit** (28 blocks, 3,072 units each, 86,016 per token), chosen
for speed so the full grid can run; Qwen3-4B is the follow-up if this passes.

## Data (made up, seeded)

- **Memories**: 200 at once. `Remember that {first} {last} works as a {job} in
  {city}.` Names from 40 first x 40 last names, 40 jobs, 40 cities, drawn at
  random, so memories share first names, last names, jobs and cities.
- **Probes** for a remembered person, 8 new tokens generated greedily, right when
  the text contains that person's job and city:
  - lead-in: `Yesterday I met {name}. {name} works as a`
  - reworded: `Everyone in town knows {name}, who works as a`
  - paraphrase: `What is {name}'s job? {name} is a`
- **Controls**, changed when any of the 8 tokens differs from the model alone:
  - other person: the same three sentences for a name with no memory (it may
    share a first or last name with remembered people)
  - unrelated: the first 8 words of a TinyStories story

## Systems

- model alone
- kNN-LM as published: key = final state, value = next token, `p = (1 - lam)
  p_model + lam * softmax(-d / tau)`; lam {0.1, 0.25, 0.5, 0.75, 0.9} x tau
  {0.03, 0.1, 0.3, 1} (distance in units of the typical distance between entries)
- kNN-LM weighted like the route memory: `score = log p + beta * sum exp(-d /
  tau)`, not normalised; beta {1, 3, 10, 30, 100} x the same tau
- route memory: gamma {0.5, 0.8} x scale {10, 30, 100, 300, 1000} x beta {1e2,
  1e3, 1e4, 1e5, 1e6, 1e7, 1e8}
- for reference only, not part of the bar: all 200 memory sentences put in the
  prompt (what retrieval into the context would give at best)

## Score and bar

Score = mean recall over the three probe kinds minus mean share changed over the
two control kinds. Every knob is picked on a calibration set (seed 101: its own
200 memories, 30 prompts of each kind) for the best score. The test set (seed
202: 200 new memories, 60 prompts of each kind) is run once with those settings.

**Pass bar**: the route memory's test score is at least 10 points above the
better of the two kNN-LMs.

## Honest test of the demo (written before running, 2026-11-23)

Chris: the question is whether prompts passed in alter later generation; test
that honestly, no gaming. Test 2 below was stopped and never produced a result.

`honest.py`. The memory exactly as in `demo.py`, settings fixed at the demo's
(gamma 0.8, scale 1000, delta0 0.12, chosen by hand on the demo's five
sentences). Nothing is tuned on this test; it is run once, seed 404.

**Told**: 60 sentences at once, 10 from each of six kinds, filled in at random:
a person's job and city, a thing's colour, a pet's name, a place's 4-digit
password, a person's favourite food, a train's platform. Each is one prompt; the
model reads it and replies 6 tokens; that pass is the memory.

**Asked**, 10 tokens generated greedily:
- same start (60): the told sentence cut just before the fact. Right when the
  generation contains the fact (job and city; colour; name; the number).
- lead-in (60): the same, after `Listen: `, `As I said before, ` or `Just so you
  know, `. Right the same way.
- near miss (30): the same kind of sentence about a subject that was never told
  (another person, thing, pet, place, train). Counted: changed from the model
  alone, and given a told fact of that kind.
- unrelated (60): the first 8 words of a TinyStories story. Counted: changed
  from the model alone.

Reported as those numbers, next to the model alone. No kNN-LM here: it would
need its own tuning, and the question is only whether told prompts alter later
generations correctly.

## Honest test, carried sim (written before running, 2026-11-23)

Chris: the goal is to not care what the lead-in is; the sim should carry token by
token through. Measured on the demo sentences: after a lead-in the first token
is a different token (`The` vs ` The`, 34% of units equal) while every token
after it keeps 90-96%. The windowed average let that one token sink the match.

Change (`Store.carry`, `next_carry`): the sim is carried along the told sentence
token by token. A token that matches (share of units equal above THETA 0.85)
adds its excess to what was carried from the token before; one that does not
drops it and the next token starts from zero. A place's weight is `exp(kappa *
(carried - c0))`. THETA 0.85, kappa 100, c0 0.10 were set by hand on the demo's
five sentences (`/tmp/carry24.py` printed their carried sims).

The same test as above, nothing tuned on it: `honest.py`, new seed 505, run
once. The numbers that answer the question are same start, lead-in and
unrelated; the near-miss count stays in the file as an observation only (Chris:
whose memory it is was never a requirement).

## Honest test, no gate (written before running, 2026-11-23)

Chris: stop gating, just test the model. The carried version above had three
rules of mine (a cutoff for "matching", a reset on a mismatch, a threshold for
full strength). All three are gone (`Store.carry_sim`, `next_sim`):

    P_t[n] = sim_t[n] ** scale * (1 + P_(t-1)[n-1])      sim = share of units equal
    score(token) = log p_model + pull * sum_places P[place - 1] * fit(token, place)

Every told token's raw sim is carried and multiplied along the run of tokens
before it, every run counts, every place always pulls. The only two numbers are
the scale (40) and the pull (3000), set by hand on the demo's five sentences
(`/tmp/sim24.py` tried scale 20/40/80 x pull 300/3000/30000 on them).

The same `honest.py`, nothing tuned on it, new seed 606, run once. Reported:
told fact back with the same start, told fact back after a lead-in, and how
many unrelated generations differ from the model alone.

## Honest test, surprise (written before running, 2026-11-24)

Chris, 2026-11-24: the nudging is proven; generation is never gated; there is no
"lead-in" (dropped, with the whose-memory count); the model should learn and
drift, not turn to gibberish. "It is like back prop all over again... we need
derivatives of an adjusted set of weights the new generations created."

That derivative, exactly: one backprop step on the output weights from every
told token, `dW = sum (told - predicted) key^T`, never written into the model
but applied beside it (`next_surprise`):

    scores = log p_model + eta * sum_places P[place] * (told - predicted)

P is the ungated carried sim as before (scale 40); told is 1 on the token told
at that place; predicted is the model's own distribution there, kept from the
told pass. A place where the model already said what it was told adds nothing.
eta 10 was set by hand on the demo's five sentences (`/tmp/sur24.py` tried 1,
3, 10, 30).

`honest.py`, nothing tuned on it, new seed 707, run once, 60 told prompts:
- told fact back: the told sentence cut before the fact, 10 tokens generated.
- unrelated story openings: how many generations differ from the model alone,
  and whether they stay logical, measured by the frozen model's own average
  log probability per token of what was written (gibberish scores far lower),
  with samples printed.

## The editing table on an 8B model (written before running, 2026-11-24)

Chris: move forward with the test on the 7B-parameter model. The result that
would make this stick is the standard knowledge-editing table.

- **Model**: LLaMA-3-8B-Instruct (the model of the MEMOIR table), MLX 4-bit
  (published numbers are full precision; said wherever they are compared).
- **Data**: zsRE (`zsre_mend_eval.json`, the ROME/EasyEdit file, 19,086 records).
  Test = records 0-999; calibration = records 1000-1999.
- **Told**: each edit is one prompt, `{question} {new answer}`, one forward pass,
  1,000 at once. Nothing is trained and no weights change.
- **Memory**: `next_surprise` exactly as on Qwen (ungated carried route sim,
  told minus predicted at the logits). Two engineering changes for size, neither
  a change of rule: the route sim is measured on a fixed random sample of 32,768
  of the 458,752 MLP units (standard error of a share about 0.002), and each
  place keeps the model's 64 most likely tokens as its prediction.
- **Scored** the way EasyEdit scores the published tables, teacher-forced token
  accuracy after all 1,000 edits are in:
  - reliability: the question, is each token of the new answer the top token;
  - generalization: the same for the benchmark's rephrased question;
  - locality: on the unrelated question with its answer, the share of top
    tokens that are the same as the model alone.
- **Settings**: the scale and step are per model, so they are picked on the
  calibration edits (scale {5, 10, 20, 40, 80} x step {3, 10, 30, 100, 300},
  best mean of the three scores with all 1,000 calibration edits in), then the
  test edits are run once. (The grid was first {20, 40, 80} x {3, 10, 30}; a
  plumbing run on 20 throwaway records, 5000-5039, picked its corner, so it was
  widened before the real run.)
- **Bar** (zsRE, T = 1000): MEMOIR 0.94 / 0.85 / 1.00, WISE 0.77 / 0.72 / 1.00,
  GRACE 0.93 / 0.08 / 1.00 (reliability / generalization / locality, from the
  WISE and MEMOIR papers). Expected weak point, said beforehand: generalization
  (a reworded question has never passed), and locality cannot be 1.00 because
  nothing is gated.

## The editing table, matched to the published setup (written before running, 2026-11-24)

Chris: match the setup. Read from the MEMOIR and EasyEdit code
(`examples/run_wise_editing.py`, `hparams/MEMOIR/llama3-8b.yaml`,
`easyeditor/evaluate/evaluate_utils.py`), what the published LLaMA-3-8B row is:

- model `meta-llama/Meta-Llama-3-8B-Instruct`, raw prompts (`use_chat_template:
  False`);
- edits = the first 1,000 records of `zsre_mend_edit.json` (1,037 records, from
  the WISE data archive), a different file from the one the first run used (350
  of its first 1,000 questions overlap);
- edit prompt `src`, new answer `alt`, rephrase `rephrase`, locality `loc` with
  `loc_ans`; all edits applied, then every record scored;
- score `test_prediction_acc`: tokenize `prompt + ' ' + target`, one forward
  pass, the top token at each target position against the target token, mean
  per record, mean over records; locality = the same top tokens before and
  after editing, share equal. (GRACE alone is scored by free generation.)

This run, `edit_test.py published`:
- the full-precision model (`mlx-community/Meta-Llama-3-8B-Instruct`, BF16, the
  same weights converted for MLX), raw prompts;
- the same 1,000 records in the same order, told as `src + ' ' + alt`;
- scoring is my line-for-line port of `test_prediction_acc` (the original needs
  a PyTorch model on a CUDA device, so it cannot be called on this machine);
- calibration = the first 1,000 records of `zsre_mend_train.json` that share no
  question, rephrase or subject with the edit file; grid scale {20, 30, 40, 60,
  80} x step {30, 100, 300, 1000, 3000} (extended upward because the 4-bit run
  picked the top step), same rule (best mean of the three), then the test once.

Everything else as in the run above (32,768 sampled units, top-64 predictions).

## The step curve, continued (written before running, 2026-11-24)

The matched run picked the largest step in its grid (scale 60, step 3000), and
on calibration generalization was still rising with the step at flat locality.
`edit_test.py published steps` continues the grid: scale {50, 60, 80, 100} x
step {3e3, 1e4, 3e4, 1e5, 3e5}, picked on the same calibration records by the
same rule (best mean of the three scores), then the same 1,000 test edits once.

This is the second time the test edits are scored. The setting is chosen from
calibration alone, and both test results are kept and reported (1.00 / 0.72 /
0.99 for the first).

## Sim on what differs (written before running, 2026-11-24)

`diagnose.py` (calibration only) showed the reworded questions fail at the first
answer token because the right memory is not found: most units of a route fire
the same for every question, so one token's sim barely separates questions
(right place 0.877, closest other 0.893). Chris agreed the fix to try: measure
the sim only on what differs.

Change, the only one (`Memory.set_center`, `shares`): before two routes are
compared, each unit's usual firing is removed; the sim is `(1 + cosine) / 2` of
what is left, so a unit that fires as usual counts for nothing and a unit that
departs from its usual counts in proportion. Two versions of "usual" are tried:
- `global`: the unit's mean over every told token;
- `token`: its mean over the told tokens that are the same token (a token told
  fewer than 5 times uses the global mean).
The carry, the surprise nudge, the sampled units and the scoring are unchanged.

`edit_test.py published centered`: both versions over scale {5, 10, 20, 40, 80,
160} x step {1e2, 1e3, 1e4, 1e5, 3e5} on the same calibration records; the best
mean of the three scores picks version and setting; then the same 1,000 test
edits once. Third scoring of those test edits (1.00 / 0.72 / 0.99, then 1.00 /
0.77 / 0.99); all three are kept and reported, and the choice comes from
calibration alone.

## Evidence wherever the words appear (written before running, 2026-11-24)

Removing shared firing gave +0.02 (global) or made it worse (per token). The
other cause from `diagnose.py` is untouched: the carry runs token against token
at the same offset, and 635 of 1,000 reworded questions keep only their last
token in place. Chris: try letting it follow the told sentence when the same
words appear at different positions.

Change (`Memory.weights`, plain sim): besides the carry, every told sentence
collects evidence from every token so far, at any position:

    a_t[n] = S_t[n]^scale / sum over all told tokens n' of S_t[n']^scale
    E[sentence] = sum over tokens so far t and the sentence's tokens n of a_t[n]

The division is what makes a word count by how particular it is: a word told in
a thousand sentences gives each a thousandth of itself, a word told in one gives
that sentence nearly all. A place's weight is its carried sim multiplied by E of
its sentence (`evidence-times`) or by 1 + E (`evidence-plus`). Nothing is gated;
the surprise nudge and the scoring are unchanged.

`edit_test.py published evidence`: both forms over scale {20, 40, 80} x step
{1e2 ... 1e7} on the calibration records. Only if the best calibration mean
beats the standing best (0.908, the centred sim) are the 1,000 test edits scored
(a fourth scoring; every result is kept and reported).

The comparison rows are corrected from here on: the MEMOIR paper reports the
other methods on this same model (LLaMA-3-8B, zsRE, T = 1000): MEMOIR 0.94 /
0.85 / 1.00, AlphaEdit 0.84 / 0.77 / 0.56, WISE 0.66 / 0.64 / 1.00, GRACE 1.00 /
0.37 / 1.00. The WISE and GRACE rows shown before were their LLaMA-2-7B numbers
from the WISE paper.

## Test 2: the bug in test 1, fixed (written before running, 2026-11-23)

Test 1 recalled nothing, and that was a bug in how I wrote the weight, found on
calibration data only (`/tmp/diag24.py` measures deltas on seed 101):

- For a lead-in probe the right person's place is at delta 0.074 (gamma 0.5) or
  0.100 (gamma 0.8); the closest other person's is 0.074 / 0.105, the median
  other 0.088 / 0.131; unrelated text is 0.20 away. With gamma 0.8 the right
  place is the closest of all 2,941 places in 90% of lead-in probes.
- So telling one person from 199 others needs a sharp scale (hundreds to
  thousands). But the weight was `beta * exp(-scale * delta)` with beta at most
  1e8: at scale 1000 and delta 0.07 that is 1e-32 x 1e8, no pull at all. Every
  setting in the grid was either too blunt to pick the person or too sharp to
  fire. The weighted kNN-LM had the same fault.

The fix is the same formula written so the grid can reach it, in logs: weight =
`exp(-scale * (delta - delta0))`, i.e. a place delta0 away pulls with a fixed
strength (1000), closer places exponentially more, farther ones exponentially
less. Nothing else changes. The same for the weighted kNN-LM:
`30 * exp(-(d - d0) / tau)`.

Grids: route memory gamma {0.5, 0.8, 0.95} x scale {100, 300, 1000, 3000} x
delta0 {0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20}; kNN-LM lam {0.1, 0.25,
0.5, 0.75, 0.9} x tau {0.001, 0.003, 0.01, 0.03, 0.1, 0.3}; weighted kNN-LM tau
{0.001, 0.003, 0.01, 0.03} x d0 {0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3}.

kNN-LM's grid is sharper than in test 1 for the same reason: on 12 throwaway
memories (seed 901) one person's entry is 0.0002 from its own prefix and other
people's are about 0.01 apart, so tau 0.03 and up blurs every person together
(it answered "bricklayer" for 10 of 12 people on their own sentences).

Same data recipe, score and bar. Calibration seed 101; the test is a new set,
seed 303, run once. Expected from the diagnostic and said here beforehand: the
lead-in probe can work; the reworded and paraphrase probes probably cannot yet
(the right place ranks 10th to 60th there).
