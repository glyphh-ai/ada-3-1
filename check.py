"""check.py — the plumbing, before any test: the hand-run pass equals the model's,
the short list finds the same token as scoring every token exactly, and timing.

Run:  uv run python check.py
"""

import time

import mlx.core as mx
import numpy as np

import route as R

m = R.Model()
ids = m.encode("Remember that Mara Ellison works as a beekeeper in Lisbon.")
lp, hid, routes = m.forward(ids, m.new_caches())
ref = m.model(mx.array(np.array(ids)[None]))[0, -1].astype(mx.float32)
ref = ref - mx.logsumexp(ref)
print(f"{R.NAME}: {len(m.layers)} blocks, {m.H} units each, {m.U} per token")
print("largest gap, hand-run pass vs the model:", float(mx.abs(lp - ref).max()))
print("share of units firing:", float((routes > 0).mean()))

t = time.time()
R0 = m.block1_alone()
print(f"block-1 route of every token alone: {time.time() - t:.1f}s")

store = R.Store(m, ["Remember that Mara Ellison works as a beekeeper in Lisbon.",
                    "Remember that Tobin Hale works as a glassblower in Oslo.",
                    "Remember that Ines Okafor works as a locksmith in Quito."])
exact = np.array(m.block1_exact(R.State.start(m, store, ids[:6]).caches[0], [ids[6]]))[0]
print("block-1 route of one token: alone vs in context agree on",
      f"{float((np.array(R0[ids[6]]) == exact).mean()):.1%} of units")

same = n = 0
t = time.time()
for prompt in ["Yesterday I met Mara Ellison. Mara Ellison works as a",
               "What is Tobin Hale's job? Tobin Hale is a", "The weather today is"]:
    s = R.State.start(m, store, m.encode(prompt))
    for _ in range(4):
        for scale, beta in ((100, 0.10), (300, 0.12)):          # (scale, delta0)
            a = R.next_route(s, R0, scale, beta, 0.5)
            b = R.next_route(s, R0, scale, beta, 0.5, exact_all=True)
            same += a == b
            n += 1
        s = s.child(a)
print(f"short list picks the same token as every token scored exactly: {same}/{n} ({time.time() - t:.0f}s)")

s = R.State.start(m, store, m.encode("Yesterday I met Mara Ellison. Mara Ellison works as a"))
t = time.time()
for _ in range(20):
    R.next_route(s, R0, 100, 0.10, 0.5)
print(f"one route decision: {(time.time() - t) / 20 * 1000:.0f} ms")
t = time.time()
x = s
for k in range(20):
    x = x.child(1000 + k)
print(f"one model step: {(time.time() - t) / 20 * 1000:.0f} ms")
for scale, beta in ((100, 0.10), (300, 0.10), (1000, 0.12)):  # (scale, delta0)
    print(scale, beta, "->", repr(m.decode(R.generate(s, lambda z: R.next_route(z, R0, scale, beta, 0.5), 8))))
print("model alone ->", repr(m.decode(R.generate(s, R.next_alone, 8))))
