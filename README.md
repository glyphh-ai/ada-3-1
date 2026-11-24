# ada-3-1: route memory

A frozen language model learns from what it is told. Nothing is trained, no
weights change, and nothing is added to later prompts.

## How it works

1. **Told.** Every prompt told to the model is run once. For each token we keep
   its *route* (which MLP units fire in every block), the token that came next,
   and what the model itself predicted there.
2. **Similar.** While generating, the current token's route is compared with
   every stored token's route, and that similarity is carried token by token
   along the stored sentence, never gated:
   `P_t[n] = sim_t[n]^scale * (1 + P_(t-1)[n-1])`
3. **Nudged.** The next-token scores are pushed by the surprise of what was told:
   `scores = log p_model + step * sum_places P[place] * (told - predicted)`
   This is one backprop step on the output weights per told token, kept beside
   the weights instead of written into them. A place where the model already
   said what it was told adds nothing.

## Result so far

zsRE, 1,000 edits told at once, full-precision LLaMA-3-8B-Instruct, the edit set
and scoring of the published runs (`edit_test.py published`, run once):

| method | reliability | generalization | locality |
|---|---|---|---|
| route memory (this repo) | 1.00 | 0.72 | 0.99 |
| model alone | 0.31 | 0.30 | 1.00 |
| MEMOIR (published, LLaMA-3-8B) | 0.94 | 0.85 | 1.00 |
| WISE (published, LLaMA-2-7B) | 0.77 | 0.72 | 1.00 |
| GRACE (published, LLaMA-2-7B) | 0.93 | 0.08 | 1.00 |

Limits: scoring is a port of EasyEdit's `test_prediction_acc` (the original
needs CUDA); the other rows are published numbers, not reruns; the step setting
sat at the top of its grid, so the generalization curve is not finished.

## Run it (Apple silicon, MLX)

```
uv sync
uv run python demo.py "The sky is red." -- "The sky is"     # Qwen3-0.6B, tell then ask
uv run python honest2.py                                    # 60 told prompts, Qwen3-0.6B
uv run python edit_test.py published                        # the table above (16 GB model)
```

Models are read from the Hugging Face cache
(`mlx-community/Qwen3-0.6B-4bit`, `mlx-community/Meta-Llama-3-8B-Instruct`).

## Where things are

| file | what |
|---|---|
| `route.py` | the memory: routes, carried similarity, the surprise nudge (`next_surprise`) and the earlier variants |
| `edit_test.py` | the zsRE editing table on LLaMA-3-8B |
| `demo.py`, `honest.py`, `honest2.py` | tell-then-ask demo and its honest tests on Qwen3-0.6B |
| `run.py`, `check.py` | the earlier kNN-LM comparison and plumbing checks |
| `PROTOCOL.md` | every test, written down before it ran |
| `FINDINGS.md` | every result, including the failures |
| `reports/` | prior-art search (what is and is not new) |
| `history/opengpt-research/` | how the idea got here, on the small numpy GPT |
| `data/` | zsRE files used by the tests |
