"""edit_test.py — the editing table in PROTOCOL.md: 1,000 zsRE edits told to a
frozen 8B model, scored the EasyEdit way (reliability, generalization, locality).

Run:  uv run python edit_test.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask

import route as R

NAME = "Meta-Llama-3-8B-Instruct-4bit"
N_SAMPLE, TOPK, N_EDITS = 32768, 64, 1000
SCALES, ETAS = (5.0, 10.0, 20.0, 40.0, 80.0), (3.0, 10.0, 30.0, 100.0, 300.0)


class Big(R.Model):
    """The model with its route measured on a fixed random sample of MLP units."""

    def __init__(self, name=NAME, seed=0):
        super().__init__(name)
        idx = np.sort(np.random.default_rng(seed).choice(self.U, N_SAMPLE, replace=False))
        self.idx = [mx.array(idx[(idx >= l * self.H) & (idx < (l + 1) * self.H)] - l * self.H)
                    for l in range(len(self.layers))]
        self.bos = self.tok.bos_token_id

    def run(self, text):
        """One pass over the text -> (tokens, log p after every token (L, V), routes (L, N_SAMPLE) +1/-1)."""
        ids = self.encode(text)
        x = mx.array(np.array([self.bos] + ids, np.int32)[None])
        h = self.inner.embed_tokens(x)
        mask = create_attention_mask(h, None)
        signs = []
        for layer, ix in zip(self.layers, self.idx):
            h = h + layer.self_attn(layer.input_layernorm(h), mask, None)
            x2 = layer.post_attention_layernorm(h)
            g = layer.mlp.gate_proj(x2)
            h = h + layer.mlp.down_proj(nn.silu(g) * layer.mlp.up_proj(x2))
            signs.append(g[0][:, ix] > 0)
        out = self.inner.norm(h)[0]
        logits = (self.inner.embed_tokens.as_linear(out) if self.model.args.tie_word_embeddings
                  else self.model.lm_head(out)).astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        routes = mx.where(mx.concatenate(signs, -1), 1.0, -1.0).astype(mx.float32)
        mx.eval(lp, routes)
        return ids, lp[1:], routes[1:]          # the start-of-text token is not part of what was told


class Memory:
    """Every told token: its route, the token told next, and what the model predicted there."""

    def __init__(self, m, told):
        self.m = m
        routes, pos, vals, pidx, pval = [], [], [], [], []
        for text in told:
            ids, lp, r = m.run(text)
            routes.append(r)
            pos += list(range(len(ids)))
            top = mx.argpartition(-lp, TOPK, axis=-1)[:, :TOPK]
            mx.eval(top)
            top = np.array(top)
            p = np.exp(np.take_along_axis(np.array(lp), top, -1))
            # entry n predicts token n: told = ids[j], predicted = the model after token j-1
            vals += ids
            pidx.append(np.concatenate([top[:1] * 0, top[:-1]]))
            pval.append(np.concatenate([p[:1] * 0, p[:-1]]))
        self.M = mx.concatenate(routes, 0)
        mx.eval(self.M)
        self.pos, self.vals = np.array(pos), np.array(vals)
        self.pidx, self.pval = np.concatenate(pidx), np.concatenate(pval).astype(np.float64)
        self.has_prev = self.pos >= 1

    def shares(self, routes):
        """Share of sampled units equal: every given token against every told token, (T, N)."""
        return np.array((1 + routes @ self.M.T / N_SAMPLE) / 2, dtype=np.float64)

    def weights(self, S, scale):
        """Carried sim, no gate: P_t[n] = S_t[n]^scale * (1 + P_(t-1)[n-1]). Returns, for every
        position t, the weight of each place n (the sim carried up to told token n-1)."""
        P = np.zeros(S.shape[1])
        out = []
        for row in S:
            prev = np.where(self.has_prev, np.roll(P, 1), 0.0)
            P = row ** scale * (1.0 + prev)
            out.append(np.where(self.has_prev, np.roll(P, 1), 0.0))
        return out

    def shift(self, w):
        """sum_places w * (told - predicted), over the vocabulary."""
        s = np.zeros(self.m.V)
        np.add.at(s, self.vals, w)
        np.add.at(s, self.pidx.ravel(), -(w[:, None] * self.pval).ravel())
        return s


def records(lo, hi):
    return json.loads(Path("data/zsre_mend_eval.json").read_text())[lo:hi]


def evaluate(m, mem, recs, settings, label):
    """-> {setting: (reliability, generalization, locality)}, model alone (rel, gen)."""
    acc = {s: {"rel": [], "gen": [], "loc": []} for s in settings}
    alone = {"rel": [], "gen": []}
    t0 = time.time()
    for i, r in enumerate(recs):
        for kind, prompt, target in (("rel", r["src"], r["alt"]), ("gen", r["rephrase"], r["alt"]),
                                     ("loc", r["loc"], r["loc_ans"])):
            ids, lp, routes = m.run(f"{prompt} {target}")
            k = len(ids) - len(m.encode(prompt))
            if k <= 0:
                continue
            at = list(range(len(ids) - k - 1, len(ids) - 1))      # positions that predict the target tokens
            want = np.array(ids[-k:])
            lp_np = np.array(lp[mx.array(at)], dtype=np.float64)
            base = lp_np.argmax(-1)
            if kind != "loc":
                alone[kind].append(float((base == want).mean()))
            S = mem.shares(routes)
            for scale in sorted({s[0] for s in settings}):
                W = mem.weights(S, scale)
                shifts = [mem.shift(W[t]) for t in at]
                for eta in sorted({s[1] for s in settings if s[0] == scale}):
                    top = np.array([(lp_np[j] + eta * shifts[j]).argmax() for j in range(k)])
                    acc[(scale, eta)][kind].append(float((top == (base if kind == "loc" else want)).mean()))
        if (i + 1) % 100 == 0:
            print(f"  {label}: {i + 1}/{len(recs)} edits scored, {time.time() - t0:.0f}s", flush=True)
    return ({s: tuple(float(np.mean(v[k])) for k in ("rel", "gen", "loc")) for s, v in acc.items()},
            tuple(float(np.mean(alone[k])) for k in ("rel", "gen")))


def write(m, recs, label):
    t0 = time.time()
    mem = Memory(m, [f"{r['src']} {r['alt']}" for r in recs])
    print(f"  {label}: {len(recs)} edits told in one pass each, {len(mem.pos)} tokens, {time.time() - t0:.0f}s", flush=True)
    return mem


def main():
    m = Big(NAME)
    print(f"{NAME}: {len(m.layers)} blocks x {m.H} units = {m.U} per token, route sim on {N_SAMPLE} sampled units")

    if PUBLISHED:       # the published runs' own edit set; calibration shares no question or subject with it
        cal = json.loads(Path("data/ZsRE/zsre_calibration_from_train.json").read_text())[:N_EDITS]
    else:
        cal = records(OFFSET + N_EDITS, OFFSET + 2 * N_EDITS)
    grid = list(itertools.product(SCALES, ETAS))
    res, _ = evaluate(m, write(m, cal, "calibration"), cal, grid, "calibration")
    print("calibration (reliability, generalization, locality):")
    for s in grid:
        print(f"  scale {s[0]:g} step {s[1]:g}: " + " / ".join(f"{x:.3f}" for x in res[s]))
    best = max(grid, key=lambda s: np.mean(res[s]))
    print(f"picked: scale {best[0]:g}, step {best[1]:g}")
    tag = ("_published_steps" if STEPS else "_published") if PUBLISHED else ""
    Path(f"edit_calibration{tag}.json").write_text(json.dumps({f"{s[0]:g},{s[1]:g}": res[s] for s in grid}, indent=1))

    if PUBLISHED:
        lo = 0 if N_EDITS >= 1000 else 1000        # a plumbing run only touches records 1000+, outside the test
        test = json.loads(Path("data/ZsRE/zsre_mend_edit.json").read_text())[lo:lo + N_EDITS]
    else:
        test = records(OFFSET, OFFSET + N_EDITS)
    res, alone = evaluate(m, write(m, test, "test"), test, [best], "test")
    rel, gen, loc = res[best]
    print(f"\nzsRE, {N_EDITS} edits at once, {NAME}, run once (scale {best[0]:g}, step {best[1]:g}):")
    print("| method | reliability | generalization | locality |\n|---|---|---|---|")
    print(f"| route memory, surprise (this run) | {rel:.2f} | {gen:.2f} | {loc:.2f} |")
    print(f"| model alone (this run) | {alone[0]:.2f} | {alone[1]:.2f} | 1.00 |")
    print("| MEMOIR (published, LLaMA-3-8B) | 0.94 | 0.85 | 1.00 |")
    print("| WISE (published, LLaMA-2-7B) | 0.77 | 0.72 | 1.00 |")
    print("| GRACE (published, LLaMA-2-7B) | 0.93 | 0.08 | 1.00 |")
    Path(f"edit_results{tag}.json").write_text(json.dumps(
        {"model": NAME, "edits": N_EDITS, "scale": best[0], "step": best[1], "reliability": rel,
         "generalization": gen, "locality": loc, "alone_reliability": alone[0], "alone_generalization": alone[1]}, indent=1))


if __name__ == "__main__":
    OFFSET, PUBLISHED, STEPS = 0, False, False
    if sys.argv[1:2] == ["published"]:  # the published setup: full-precision model, the published runs' edit set
        PUBLISHED, NAME = True, "Meta-Llama-3-8B-Instruct"
        SCALES, ETAS = (20.0, 30.0, 40.0, 60.0, 80.0), (30.0, 100.0, 300.0, 1000.0, 3000.0)
        if sys.argv[2:3] == ["steps"]:  # the step curve continued past the first grid's top
            SCALES, ETAS, STEPS = (50.0, 60.0, 80.0, 100.0), (3e3, 1e4, 3e4, 1e5, 3e5), True
        elif len(sys.argv) > 2:         # plumbing only: first n calibration records, n records from the far end as "test"
            N_EDITS = int(sys.argv[2])
    elif len(sys.argv) > 1:             # smoke on records the real run never touches: edit_test.py <n edits>
        N_EDITS, OFFSET = int(sys.argv[1]), 5000
    main()
