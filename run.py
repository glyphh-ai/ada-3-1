"""run.py — the test in PROTOCOL.md: route memory vs kNN-LM on Qwen, whole generations.

Run:  uv run python run.py
"""

import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

import route as R

N_MEM, N_NEW = 200, 8
FIRST = "Mara Tobin Ines Calla Dorian Esme Farid Greta Hollis Imani Jarek Keiko Lorcan Mirela Nadia Osric Petra Quentin Rania Soren Talia Ulric Vesna Wendell Xiomara Yusuf Zelda Anders Bettina Caspian Delphine Emeric Fenna Gideon Henrike Ignacio Jorah Katya Leopold Marisol".split()
LAST = "Ellison Hale Okafor Brandt Castellan Dvorak Easton Fairweather Galloway Hartmann Ibarra Jankowski Kessler Lindqvist Marchetti Novak Oyelaran Pemberton Quist Rasmussen Sandoval Thackeray Underhill Vasquez Whitlock Yamada Zielinski Abernathy Blackwood Cardenas Delacroix Engstrom Fitzroy Grimaldi Holloway Iversen Jaramillo Kowalczyk Lachance Montague".split()
JOBS = "beekeeper glassblower locksmith cartographer blacksmith florist plumber librarian tailor baker carpenter mechanic surveyor translator welder botanist jeweler midwife paramedic electrician fisherman shepherd potter roofer upholsterer watchmaker bricklayer chimney-sweep dentist pharmacist geologist archivist bookbinder cobbler stonemason violinist puppeteer tanner brewer falconer".split()
CITIES = "Lisbon Oslo Quito Nairobi Hanoi Tbilisi Montevideo Riga Dakar Lima Zagreb Tunis Havana Tallinn Bogota Accra Jakarta Vilnius Rabat Asuncion Sofia Kampala Manila Bratislava Amman Reykjavik Lusaka Belgrade Muscat Tashkent Windhoek Sarajevo Colombo Yerevan Kigali Ljubljana Santiago Kathmandu Minsk Algiers".split()
PROBES = {
    "lead-in": "Yesterday I met {n}. {n} works as a",
    "reworded": "Everyone in town knows {n}, who works as a",
    "paraphrase": "What is {n}'s job? {n} is a",
}
STORIES = Path.home() / "development/openGPT/data/tinystories.txt"

KNN = list(itertools.product((0.1, 0.25, 0.5, 0.75, 0.9), (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)))  # lam, tau
KNN_SCALED = list(itertools.product((0.001, 0.003, 0.01, 0.03),                                    # tau, d0
                                    (0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3)))
ROUTE = list(itertools.product((0.5, 0.8, 0.95), (100, 300, 1000, 3000),                           # gamma, scale, delta0
                               (0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20)))


def make_set(seed, n_each):
    rng = random.Random(seed)
    names = rng.sample([f"{a} {b}" for a in FIRST for b in LAST], N_MEM + n_each)
    people = {n: (rng.choice(JOBS), rng.choice(CITIES)) for n in names[:N_MEM]}
    memories = [f"Remember that {n} works as a {j} in {c}." for n, (j, c) in people.items()]
    prompts = []                                    # (kind, text, job, city)
    for kind, tpl in PROBES.items():
        for n in rng.sample(names[:N_MEM], n_each):
            prompts.append((kind, tpl.format(n=n), *people[n]))
    tpls = list(PROBES.values())
    for i, n in enumerate(names[N_MEM:]):
        prompts.append(("other person", tpls[i % 3].format(n=n), None, None))
    stories = [s.split() for s in STORIES.read_text().split("<|endoftext|>")]
    for s in rng.sample([s for s in stories if len(s) >= 8], n_each):
        prompts.append(("unrelated", " ".join(s[:8]), None, None))
    return memories, prompts


def run_set(m, R0, memories, prompts, systems, label):
    """systems: {name: pick(state)} -> {name: [generated tokens per prompt]}"""
    store = R.Store(m, memories)
    out = {k: [] for k in systems}
    t = time.time()
    for i, (_, text, _, _) in enumerate(prompts):
        root = R.State.start(m, store, m.encode(text))
        for k, pick in systems.items():
            out[k].append(R.generate(root, pick, N_NEW))
        if (i + 1) % 30 == 0:
            print(f"  {label}: {i + 1}/{len(prompts)} prompts, {time.time() - t:.0f}s", flush=True)
    return out


def score(m, prompts, gen, alone):
    rec = {k: [] for k in PROBES}
    chg = {"other person": [], "unrelated": []}
    for (kind, _, job, city), g, a in zip(prompts, gen, alone):
        if kind in rec:
            text = m.decode(g).lower()
            rec[kind].append(job in text and city.lower() in text)
        else:
            chg[kind].append(g != a)
    rec = {k: float(np.mean(v)) for k, v in rec.items()}
    chg = {k: float(np.mean(v)) for k, v in chg.items()}
    return float(np.mean(list(rec.values())) - np.mean(list(chg.values()))), rec, chg


def systems_for(R0, knn, scaled, route):
    s = {("alone",): R.next_alone}
    s.update({("knn", l, t): (lambda z, l=l, t=t: R.next_knn(z, l, t)) for l, t in knn})
    s.update({("scaled", t, d0): (lambda z, t=t, d0=d0: R.next_knn_scaled(z, t, d0)) for t, d0 in scaled})
    s.update({("route", g, sc, d0): (lambda z, g=g, sc=sc, d0=d0: R.next_route(z, R0, sc, d0, g)) for g, sc, d0 in route})
    return s


def main(cal_seed=101, test_seed=303, n_cal=30, n_test=60):
    m = R.Model()
    R0 = m.block1_alone()

    memories, prompts = make_set(cal_seed, n_cal)
    gen = run_set(m, R0, memories, prompts, systems_for(R0, KNN, KNN_SCALED, ROUTE), "calibration")
    best = {}
    for k, g in gen.items():
        sc = score(m, prompts, g, gen[("alone",)])[0]
        if k[0] not in best or sc > best[k[0]][0]:
            best[k[0]] = (sc, k)
    for fam, (sc, k) in best.items():
        print(f"calibration pick, {fam}: {k[1:]} ({sc:+.1%})")
    # every setting's calibration numbers, to read the trade-off afterwards
    Path("calibration.json").write_text(json.dumps(
        [{"setting": list(k), **dict(zip(("score", "recall", "changed"), score(m, prompts, g, gen[("alone",)])))}
         for k, g in gen.items()], indent=1))
    if "cal" in sys.argv[1:]:
        return

    memories, prompts = make_set(test_seed, n_test)
    picks = [best[f][1] for f in ("knn", "scaled", "route")]
    sysm = systems_for(R0, [k[1:] for k in picks if k[0] == "knn"], [k[1:] for k in picks if k[0] == "scaled"],
                       [k[1:] for k in picks if k[0] == "route"])
    gen = run_set(m, R0, memories, prompts, sysm, "test")

    # reference: every memory sentence in the prompt
    prefix = m.new_caches()
    m.forward(m.encode("\n".join(memories) + "\n"), prefix)
    ctx = [R.generate(R.State.start(m, None, m.encode(text), prefix), R.next_alone, N_NEW)
           for _, text, _, _ in prompts]

    names = {"alone": "model alone", "knn": "kNN-LM", "scaled": "kNN-LM, weighted like the route memory",
             "route": "route memory"}
    rows = {names[k[0]]: score(m, prompts, g, gen[("alone",)]) for k, g in gen.items()}
    rows["(reference) all memories in the prompt"] = score(m, prompts, ctx, gen[("alone",)])
    print(f"\nlocked test, seed {test_seed}, run once ({N_MEM} memories, {n_test} prompts of each kind, 8 tokens generated):")
    print("| system | score | lead-in | reworded | paraphrase | other person changed | unrelated changed |")
    print("|---|---|---|---|---|---|---|")
    for k, (sc, rec, chg) in rows.items():
        print(f"| {k} | {sc:+.1%} | " + " | ".join(f"{v:.1%}" for v in list(rec.values()) + list(chg.values())) + " |")
    gap = rows["route memory"][0] - max(rows["kNN-LM"][0], rows["kNN-LM, weighted like the route memory"][0])
    print(f"route memory minus the better kNN-LM: {gap:+.1%}; bar +10.0%: {'PASS' if gap >= 0.10 else 'FAIL'}")

    route_key = [k for k in gen if k[0] == "route"][0]
    knn_key = [k for k in gen if k[0] == "knn"][0]
    print()
    for kind in list(PROBES) + ["other person", "unrelated"]:
        i = [j for j, p in enumerate(prompts) if p[0] == kind][0]
        _, text, job, city = prompts[i]
        told = f"{job} in {city}" if job else "(no memory)"
        print(f"{kind}: {text}\n  told: {told}\n  model alone: {m.decode(gen[('alone',)][i])!r}"
              f"\n  route memory: {m.decode(gen[route_key][i])!r}\n  kNN-LM: {m.decode(gen[knn_key][i])!r}")
    Path("results.json").write_text(json.dumps(
        {"picks": {f: list(best[f][1]) for f in best},
         "test": {k: {"score": sc, "recall": rec, "changed": chg} for k, (sc, rec, chg) in rows.items()}}, indent=1))


if __name__ == "__main__":
    main()
