"""route.py — route memory on Qwen3 (MLX).

A memory is the route of one pass over a told sentence: at every token, which MLP
units fire in every block (the sign of the SwiGLU gate). At every next token of a
new generation, every place in every memory gets a delta (how far the routes so
far are from that place's route history, the nearest tokens weighing most) and a
weight exp(-scale * delta), not normalised and never switched off. The pull on
each candidate token is the weighted sum of how well its first-block route fits
each place's next route:

    score(c) = log p_model(c) + beta * sum_places w * fit(c, place)

fit is linear in the place's route, so the sum is one matrix-vector product over
the whole vocabulary. Candidate routes are first taken without context (each
token alone through block 1, computed once), then the best SHORT candidates are
re-scored with their exact in-context block-1 route.
"""

import glob
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache

NAME = os.environ.get("QWEN", "Qwen3-0.6B-4bit")
LMAX = 8            # how many tokens back a route history looks (next_route only)
KAPPA = 30.0        # the scale: a token's sim is raised to this before it is carried (next_sim)
THETA = 0.85        # share of units equal above which a token counts as matching (next_carry)
SHORT = 256         # candidates re-scored exactly (plus the model's own top 64)


class Model:
    def __init__(self, name=NAME):
        path = glob.glob(os.path.expanduser(
            f"~/.cache/huggingface/hub/models--mlx-community--{name}/snapshots/*"))[0]
        self.model, self.tok = load(path)
        self.inner = self.model.model
        self.layers = self.inner.layers
        self.H = self.model.args.intermediate_size
        self.U = self.H * len(self.layers)
        self.V = self.model.args.vocab_size

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(list(ids))

    def new_caches(self):
        return [KVCache() for _ in self.layers]

    def forward(self, tokens, caches, every=False):
        """tokens: list of ids appended to caches -> (log p of the next token (V,),
        final hidden (L, d), routes (L, U) as +1/-1). every=True: log p after every
        token, (L, V)."""
        x = mx.array(np.array(tokens, np.int32)[None])
        h = self.inner.embed_tokens(x)
        mask = caches[0].make_mask(x.shape[1], return_array=False, window_size=None)
        signs = []
        for layer, c in zip(self.layers, caches):
            h = h + layer.self_attn(layer.input_layernorm(h), mask, c)
            x2 = layer.post_attention_layernorm(h)
            g = layer.mlp.gate_proj(x2)
            h = h + layer.mlp.down_proj(nn.silu(g) * layer.mlp.up_proj(x2))
            signs.append(g[0] > 0)
        out = self.inner.norm(h)
        last = out[0] if every else out[:, -1]
        logits = (self.inner.embed_tokens.as_linear(last) if self.model.args.tie_word_embeddings
                  else self.model.lm_head(last)).astype(mx.float32)
        logits = logits if every else logits[0]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        routes = mx.where(mx.concatenate(signs, -1), 1.0, -1.0).astype(mx.float32)
        hid = out[0].astype(mx.float32)
        mx.eval(lp, routes, hid, [c.keys for c in caches], [c.values for c in caches])
        return lp, hid, routes

    def block1_alone(self, chunk=8192):
        """Block-1 route of every vocabulary token on its own: (V, H) +1/-1."""
        layer, out = self.layers[0], []
        for k in range(0, self.V, chunk):
            h = self.inner.embed_tokens(mx.arange(k, min(k + chunk, self.V))[:, None])
            h = h + layer.self_attn(layer.input_layernorm(h), None, None)
            g = layer.mlp.gate_proj(layer.post_attention_layernorm(h))[:, 0]
            r = mx.where(g > 0, 1.0, -1.0).astype(mx.float32)
            mx.eval(r)
            out.append(r)
        return mx.concatenate(out, 0)

    def block1_exact(self, cache0, toks):
        """Block-1 route of each candidate token as the next token after cache0: (K, H)."""
        layer, K = self.layers[0], len(toks)
        c = KVCache()
        c.keys = mx.repeat(cache0.keys[..., :cache0.offset, :], K, axis=0)
        c.values = mx.repeat(cache0.values[..., :cache0.offset, :], K, axis=0)
        c.offset = cache0.offset
        h = self.inner.embed_tokens(mx.array(np.array(toks, np.int32))[:, None])
        h = h + layer.self_attn(layer.input_layernorm(h), None, c)
        g = layer.mlp.gate_proj(layer.post_attention_layernorm(h))[:, 0]
        return mx.where(g > 0, 1.0, -1.0).astype(mx.float32)


def copy_caches(caches):
    out = []
    for c in caches:
        n = KVCache()
        n.keys, n.values, n.offset = c.keys[..., :c.offset, :], c.values[..., :c.offset, :], c.offset
        out.append(n)
    return out


class Store:
    """The memories: every token's route, and for kNN-LM every token's final state."""

    def __init__(self, m, texts):
        self.m = m
        routes, keys, vals, pos, pred = [], [], [], [], []
        for t in texts:
            ids = m.encode(t)
            lp_all, hid, r = m.forward(ids, m.new_caches(), every=True)
            pred.append(mx.exp(lp_all[:-1]))                 # what the model predicted before each told token
            routes.append(r)
            hid = np.array(hid)
            keys.append(hid[:-1])
            vals += ids[1:]
            pos += list(range(len(ids)))
        self.M = mx.concatenate(routes, 0)                   # (N, U) every memory token's route
        mx.eval(self.M)
        self.pos = np.array(pos)
        self.place = np.where(self.pos >= 1)[0]              # places: tokens with a history
        self.nxt = self.M[mx.array(self.place)][:, :m.H]     # (P, H) block-1 route at the place
        mx.eval(self.nxt)
        self.keys, self.vals = np.concatenate(keys), np.array(vals)   # kNN-LM entries
        self.pred = mx.concatenate(pred, 0)                  # (P, V), row k belongs to place k; told token = vals[k]
        mx.eval(self.pred)
        k = self.keys[:: max(1, len(self.keys) // 300)][:300]
        d = ((k[:, None, :] - k[None, :, :]) ** 2).sum(-1)
        self.d_med = float(np.median(d[d > 0]))              # typical distance between entries

    def match(self, routes):
        """Share of units that match, each given route against every memory token: (n, N)."""
        return np.array((1 + routes @ self.M.T / self.m.U) / 2)

    def carry(self, C, row):
        """One token on: C[n] is the sim carried along the told sentence up to its token n.
        A token that matches (share of units equal above THETA) adds to what was carried
        from the token before; one that does not drops it, and the next starts from zero.
        So what stands in front of the matching words (a lead-in) costs nothing."""
        prev = np.where(self.pos >= 1, np.roll(C, 1), 0.0)
        return np.maximum(prev, 0.0) + row - THETA

    def carry_sim(self, P, row):
        """No gate of any kind. P[n] = every run of tokens ending at told token n, each
        run's token sims (to the power KAPPA) multiplied together, all runs added up:
            P_t[n] = sim_t[n] ** KAPPA * (1 + P_(t-1)[n-1])"""
        prev = np.where(self.pos >= 1, np.roll(P, 1), 0.0)
        return row ** KAPPA * (1.0 + prev)

    def deltas(self, hist, gamma):
        """hist[d] = match of the token d back against every memory token. One delta per place."""
        num = np.zeros(len(self.place))
        den = np.zeros(len(self.place))
        for d, h in enumerate(hist):
            ok = self.pos[self.place] - 1 - d >= 0
            num += gamma ** d * np.where(ok, h[np.maximum(self.place - 1 - d, 0)], 0.0)
            den += gamma ** d * ok
        return 1 - num / den


class State:
    """The model after a prompt plus some generated tokens; children are cached, so
    settings that write the same tokens share the passes."""

    def __init__(self, m, store, caches, lp, hid, hist, C=None, P=None):
        self.m, self.store, self.caches, self.C, self.P = m, store, caches, C, P
        self.lp, self.hid, self.hist, self.kids = np.array(lp, dtype=np.float64), np.array(hid), hist, {}
        self.lp_mx, self.dcache = lp, {}

    @classmethod
    def start(cls, m, store, tokens, caches=None):
        caches = m.new_caches() if caches is None else copy_caches(caches)
        lp, hid, routes = m.forward(tokens, caches)
        hist, C, P = [], None, None
        if store is not None:
            mt = store.match(routes)
            P = np.zeros(mt.shape[1])
            for row in mt:
                P = store.carry_sim(P, row)
            hist = [mt[i] for i in range(len(mt) - 1, max(len(mt) - 1 - LMAX, -1), -1)]
            C = np.zeros(mt.shape[1])
            for row in mt:                       # the sim is carried token by token through the prompt
                C = store.carry(C, row)
        return cls(m, store, caches, lp, hid[-1], hist, C, P)

    def child(self, tok):
        if tok not in self.kids:
            caches = copy_caches(self.caches)
            lp, hid, routes = self.m.forward([tok], caches)
            hist, C, P = [], None, None
            if self.store is not None:
                row = self.store.match(routes[-1:])[0]
                hist, C = ([row] + self.hist)[:LMAX], self.store.carry(self.C, row)
                P = self.store.carry_sim(self.P, row)
            self.kids[tok] = State(self.m, self.store, caches, lp, hid[-1], hist, C, P)
        return self.kids[tok]


def next_alone(s):
    return int(s.lp.argmax())


BETA = 1000.0       # pull of a place exactly delta0 away (fit differences are ~0.1, so ~100 logits)
BIG = 1e12          # past this the pull has long since swamped the model; capping changes nothing


def next_route(s, R0, scale, delta0, gamma, exact_all=False):
    """weight of a place = exp(-scale * (delta - delta0)): the same exp(-scale * delta)
    with its strength written as "a place delta0 away pulls with BETA", worked out in
    logs. (Written as beta * exp(-scale * delta) with beta up to 1e8, a sharp scale
    could never fire: at scale 1000 and delta 0.07 the weight is 1e-32.)"""
    if gamma not in s.dcache:
        s.dcache[gamma] = s.store.deltas(s.hist, gamma)
    return pull(s, R0, -scale * (s.dcache[gamma] - delta0), exact_all)


def next_carry(s, R0, kappa, c0, exact_all=False):
    """weight of a place = exp(kappa * (carried sim - c0)): the sim carried token by
    token up to the token before the place. c0 of carried sim pulls with BETA."""
    return pull(s, R0, kappa * (s.C[s.store.place - 1] - c0), exact_all)


def next_sim(s, R0, beta, exact_all=False):
    """Every place pulls with its carried sim times beta. No cutoff, no reset, no threshold."""
    return pull(s, R0, np.log(s.P[s.store.place - 1] + 1e-300) + np.log(beta / BETA), exact_all, shortcut=False)


def next_surprise(s, eta):
    """One backprop step on the output weights from every told token, kept beside the
    weights and applied to this context:

        scores = log p_model + eta * sum_places P[place] * (told - predicted)

    told is 1 on the token that was told at that place, predicted is what the model
    gave there. A place where the model already said what it was told adds nothing."""
    st = s.store
    w = s.P[st.place - 1]
    shift = -np.array(mx.array(w.astype(np.float32)) @ st.pred, dtype=np.float64)
    np.add.at(shift, st.vals, w)
    return int((s.lp + eta * shift).argmax())


def pull(s, R0, logw, exact_all=False, shortcut=True):
    st, m = s.store, s.m
    top_w = logw.max()
    rel = np.exp(logw - top_w)
    if shortcut and top_w < 700 and BETA * np.exp(top_w) * rel.sum() < 1e-3:   # no place pulls: the model's own token
        return int(s.lp.argmax())
    mult = min(BETA * np.exp(min(top_w, 700.0)), BIG)
    v = mx.array(rel.astype(np.float32)) @ st.nxt               # (H,) the weighted next route
    if exact_all:                                               # check only: every token exactly
        cand = np.arange(m.V)
        fit = np.concatenate([np.array(m.block1_exact(s.caches[0], cand[k:k + 4096].tolist()) @ v)
                              for k in range(0, m.V, 4096)])
    else:
        approx = s.lp_mx + mult * (R0 @ v) / (2 * m.H)
        top = np.array(mx.argpartition(-approx, SHORT)[:SHORT])
        cand = np.union1d(top, np.argpartition(-s.lp, 64)[:64])
        fit = np.array(m.block1_exact(s.caches[0], cand.tolist()) @ v)
    score = s.lp[cand] + mult * fit.astype(np.float64) / (2 * m.H)
    return int(cand[score.argmax()])


def next_knn(s, lam, tau):
    """kNN-LM as published: mix the model with a softmax over stored next tokens."""
    st = s.store
    d = ((st.keys - s.hid) ** 2).sum(1) / st.d_med
    a = np.exp(-(d - d.min()) / tau)
    a /= a.sum()
    p = (1 - lam) * np.exp(s.lp)
    np.add.at(p, st.vals, lam * a)
    return int(p.argmax())


def next_knn_scaled(s, tau, d0):
    """kNN-LM weighted the way the route memory is: exp(-(distance - d0) / tau), not
    normalised, added to the model's log probability (an entry d0 away adds 30)."""
    st = s.store
    d = ((st.keys - s.hid) ** 2).sum(1) / st.d_med
    score = s.lp.copy()
    np.add.at(score, st.vals, 30.0 * np.exp(np.minimum(-(d - d0) / tau, 30.0)))
    return int(score.argmax())


def generate(s, pick, n):
    out = []
    for _ in range(n):
        t = pick(s)
        out.append(t)
        s = s.child(t)
    return out



