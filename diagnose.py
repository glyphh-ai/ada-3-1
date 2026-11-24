"""diagnose.py — where the similarity is lost on reworded questions, and why.

Calibration records only (the test edits are never touched). All 1,000 calibration
edits are told; then every reworded question is scored at the picked setting and
each answer token is traced back to the place in memory that holds it.

Run:  uv run python diagnose.py
"""

import json
from pathlib import Path

import numpy as np

import edit_test as E

SCALE, STEP = 80.0, 300000.0
GROUPS = [("blocks 1-8", 0, 8), ("blocks 9-16", 8, 16), ("blocks 17-24", 16, 24), ("blocks 25-32", 24, 32)]


def main():
    m = E.Big("Meta-Llama-3-8B-Instruct")
    recs = json.loads(Path("data/ZsRE/zsre_calibration_from_train.json").read_text())[:1000]
    mem = E.Memory(m, [f"{r['src']} {r['alt']}" for r in recs])
    starts = np.concatenate([[0], np.cumsum([len(m.encode(f"{r['src']} {r['alt']}")) for r in recs])])
    owner = np.repeat(np.arange(len(recs)), np.diff(starts))
    per_block = [len(ix) for ix in m.idx]
    edges = np.concatenate([[0], np.cumsum(per_block)])
    M_np = None

    rows = []          # one per answer token of every reworded question
    layer = {g[0]: [] for g in GROUPS}
    layer["all blocks"] = []
    for i, r in enumerate(recs):
        told_ids = m.encode(f"{r['src']} {r['alt']}")
        ids, lp, routes = m.run(f"{r['rephrase']} {r['alt']}")
        k = len(ids) - len(m.encode(r["rephrase"]))
        kt = len(told_ids) - len(m.encode(r["src"]))
        if k <= 0 or k != kt:
            continue
        at = list(range(len(ids) - k - 1, len(ids) - 1))
        want = ids[-k:]
        lp_np = np.array(lp[E.mx.array(at)], dtype=np.float64)
        S = mem.shares(routes)
        W = mem.weights(S, SCALE)
        # how many closing tokens of the reworded question are the same tokens as the told question's
        q_new, q_old = ids[:-k], told_ids[:-k]
        run = 0
        while run < min(len(q_new), len(q_old)) and q_new[-1 - run] == q_old[-1 - run]:
            run += 1
        for j, t in enumerate(at):
            right = starts[i] + len(told_ids) - k + j          # the place that holds this answer token
            w = W[t]
            shift = mem.shift(w)
            top = int((lp_np[j] + STEP * shift).argmax())
            best_wrong = int(np.argmax(np.where(owner == i, -1.0, w)))
            rows.append(dict(j=j, ok=top == want[j], alone_ok=int(lp_np[j].argmax()) == want[j], run=run,
                             w_right=float(w[right]), w_wrong=float(w[best_wrong]), share=float(w[right] / w.sum()),
                             rank=int((w > w[right]).sum()) + 1,
                             wrong_same_token=bool(mem.vals[best_wrong] == want[j]),
                             sim_last=float(S[t][right - 1])))
            if j == 0:      # the last token of the question against the told question's last token, block by block
                q, n = np.array(routes[t]), right - 1
                if M_np is None:
                    M_np = np.array(mem.M)
                for name, a, b in GROUPS + [("all blocks", 0, 32)]:
                    c0, c1 = edges[a], edges[b]
                    s = (1 + M_np[:, c0:c1] @ q[c0:c1] / (c1 - c0)) / 2
                    others = np.where(owner == i, -1.0, s)
                    layer[name].append((float(s[n]), float(others.max()), int((others > s[n]).sum()) + 1))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(recs)} reworded questions traced", flush=True)

    R = rows
    first = [x for x in R if x["j"] == 0]
    later = [x for x in R if x["j"] > 0]
    pct = lambda xs: f"{np.mean(xs):.1%}" if len(xs) else "n/a"
    print(f"\ncalibration, {len(first)} reworded questions, {len(R)} answer tokens, scale {SCALE:g}, step {STEP:g}")
    print("\n1. Which answer token fails\n| answer token | tokens | right with memory | right, model alone |\n|---|---|---|---|")
    for name, xs in (("first", first), ("second", [x for x in R if x["j"] == 1]), ("third or later", [x for x in R if x["j"] >= 2])):
        print(f"| {name} | {len(xs)} | {pct([x['ok'] for x in xs])} | {pct([x['alone_ok'] for x in xs])} |")

    print("\n2. First answer token: did memory find the right place?\n| | questions | answer token right |\n|---|---|---|")
    found = [x for x in first if x["rank"] == 1]
    lost = [x for x in first if x["rank"] > 1]
    print(f"| right place has the largest weight | {len(found)} ({len(found) / len(first):.1%}) | {pct([x['ok'] for x in found])} |")
    print(f"| another memory's place is larger | {len(lost)} ({len(lost) / len(first):.1%}) | {pct([x['ok'] for x in lost])} |")
    fails = [x for x in first if not x["ok"]]
    print(f"\nof the {len(fails)} first-token failures: right place not the largest in {pct([x['rank'] > 1 for x in fails])}; "
          f"median rank of the right place {int(np.median([x['rank'] for x in fails]))} of {len(mem.pos)}; "
          f"median share of all weight on the right place {np.median([x['share'] for x in fails]):.3f} "
          f"(successes: {np.median([x['share'] for x in first if x['ok']]):.3f})")
    print(f"later answer tokens: right place largest in {pct([x['rank'] == 1 for x in later])}")

    print("\n3. First answer token, by how many closing tokens of the question are unchanged\n"
          "| unchanged closing tokens | questions | answer token right | right place largest | sim of the last question token |\n|---|---|---|---|---|")
    for lo, hi, name in ((0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 99, "7 or more")):
        xs = [x for x in first if lo <= x["run"] <= hi]
        if xs:
            print(f"| {name} | {len(xs)} | {pct([x['ok'] for x in xs])} | {pct([x['rank'] == 1 for x in xs])} | "
                  f"{np.mean([x['sim_last'] for x in xs]):.3f} |")

    print("\n4. Last question token against the told question's last token, one token, no carry\n"
          "| blocks | sim to the right place | sim to the closest other memory | right place is the closest |\n|---|---|---|---|")
    for name, v in layer.items():
        v = np.array(v)
        print(f"| {name} | {v[:, 0].mean():.3f} | {v[:, 1].mean():.3f} | {np.mean(v[:, 2] == 1):.1%} |")
    Path("diagnose_results.json").write_text(json.dumps(rows, default=float))


if __name__ == "__main__":
    main()
