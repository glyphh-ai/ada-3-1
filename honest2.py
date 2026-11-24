"""honest2.py — the honest test of the surprise version, as written in PROTOCOL.md.

Run:  uv run python honest2.py      (seed 707, once)
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

import demo as D
import honest as H
import route as R
import run as RUN

SEED, N = 707, 10
if len(sys.argv) > 2:               # longer generations: honest2.py <tokens> <seed>
    N, SEED = int(sys.argv[1]), int(sys.argv[2])
PARTS = [(0, 10), (10, 30), (30, 60)]


def write(s, pick):
    """Generate N tokens; also the frozen model's own average log probability of them."""
    out, lps = [], []
    for _ in range(N):
        t = pick(s)
        lps.append(float(s.lp[t]))
        out.append(t)
        s = s.child(t)
    write.parts.append(lps)
    return out, float(np.mean(lps))


write.parts = []


def main():
    rng = random.Random(SEED)
    told, _ = H.build(rng)
    stories = [s.split() for s in RUN.STORIES.read_text().split("<|endoftext|>")]
    unrelated = [" ".join(s[:8]) for s in rng.sample([s for s in stories if len(s) >= 8], 60)]

    m = R.Model()
    memories = [t + m.decode(R.generate(R.State.start(m, None, m.encode(t)), R.next_alone, 6)) for _, t, _, _ in told]
    store = R.Store(m, memories)

    def ask(prompt):
        s = R.State.start(m, store, m.encode(prompt))
        mem, lp_mem = write(s, lambda z: R.next_surprise(z, D.ETA))
        alone, lp_alone = write(s, R.next_alone)
        same = next((i for i, (a, b) in enumerate(zip(mem, alone)) if a != b), N)
        return dict(prompt=prompt, memory=m.decode(mem), alone=m.decode(alone), changed=mem != alone,
                    lp_memory=lp_mem, lp_alone=lp_alone, first_difference=same,
                    lps_memory=write.parts[-2], lps_alone=write.parts[-1])

    facts, by_kind = [], {}
    for kind, _, cut, f in told:
        r = ask(cut)
        r.update(test="told", kind=kind, told=f, right=H.has(r["memory"], kind, f), alone_right=H.has(r["alone"], kind, f))
        by_kind.setdefault(kind, []).append(r["right"])
        facts.append(r)
    un = [dict(ask(u), test="unrelated") for u in unrelated]

    print(f"Qwen3-0.6B, {len(told)} told prompts at once, surprise version, settings fixed (scale {D.SIM_SCALE:g}, "
          f"step {D.ETA:g}), seed {SEED}, run once\n")
    print("| | with memory | model alone |\n|---|---|---|")
    print(f"| told fact comes back (60) | {np.mean([r['right'] for r in facts]):.1%} | {np.mean([r['alone_right'] for r in facts]):.1%} |")
    print(f"| unrelated openings: generation differs from the model alone (60) | {np.mean([r['changed'] for r in un]):.1%} | 0.0% |")
    print(f"| unrelated openings: model's own log probability per token of the text | "
          f"{np.mean([r['lp_memory'] for r in un]):.2f} | {np.mean([r['lp_alone'] for r in un]):.2f} |")
    ch = [r for r in un if r["changed"]]
    if ch:
        print(f"| the same, only the {len(ch)} that differ | {np.mean([r['lp_memory'] for r in ch]):.2f} | "
              f"{np.mean([r['lp_alone'] for r in ch]):.2f} |")
    if N > 10:
        print(f"| unrelated: tokens written before the first difference (median) | "
              f"{int(np.median([r['first_difference'] for r in un]))} of {N} | |")
        for name, rows in (("unrelated", un), ("told", facts)):
            for a, b in PARTS:
                if a < N:
                    print(f"| {name}: log probability per token, tokens {a + 1}-{min(b, N)} | "
                          f"{np.mean([np.mean(r['lps_memory'][a:b]) for r in rows]):.2f} | "
                          f"{np.mean([np.mean(r['lps_alone'][a:b]) for r in rows]):.2f} |")
    print(f"\nright out of 10, by kind: { {k: int(sum(v)) for k, v in by_kind.items()} }\n")
    for r in facts[:3] + [r for r in facts if not r["right"]][:3]:
        print(f"told {' / '.join(r['told'])}: {r['prompt']!r}\n   with memory: {r['memory']!r}\n   model alone: {r['alone']!r}")
    print()
    for r in (ch[:6] if ch else un[:3]):
        print(f"unrelated: {r['prompt']!r}\n   with memory: {r['memory']!r}  ({r['lp_memory']:.2f})\n   model alone: {r['alone']!r}  ({r['lp_alone']:.2f})")
    Path("honest2_results.json" if N == 10 else f"honest2_results_{N}tokens.json").write_text(
        json.dumps(facts + un, indent=1, default=float))


if __name__ == "__main__":
    main()
