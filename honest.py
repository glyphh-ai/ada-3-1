"""honest.py — the honest test of the demo, exactly as written in PROTOCOL.md.

Run:  uv run python honest.py      (seed 404, once)
"""

import json
import random
import re
from pathlib import Path

import numpy as np

import demo as D
import route as R
import run as RUN

SEED, PER_KIND, N = 606, 10, 10     # 404: windowed version, 505: gated carried version
COLORS = "red green purple orange yellow pink brown black white silver turquoise maroon".split()
THINGS = ["front door", "kitchen wall", "old bicycle", "garden fence", "mailbox", "ceiling", "bedroom carpet",
          "school bus", "coffee mug", "umbrella", "notebook", "teapot", "bathtub", "rocking chair", "lampshade"]
PETS = "dog cat parrot rabbit hamster turtle horse goat lizard ferret goldfish pony kitten puppy canary".split()
PETNAMES = "Biscuit Pickle Waffles Mango Juniper Pepper Clover Ziggy Noodle Maple Tofu Sprout".split()
PLACES = "garage attic basement office shed cabin studio workshop boathouse greenhouse locker storeroom cellar barn vault".split()
FOODS = ["mushroom soup", "pickled herring", "lemon tart", "fried plantain", "barley stew", "almond cake", "roasted beets",
         "cold noodles", "plum jam", "garlic bread", "rice pudding", "lentil curry", "smoked trout", "pumpkin pie", "onion rings"]
LEADS = ["Listen: ", "As I said before, ", "Just so you know, "]


def build(rng):
    """-> told [(kind, sentence, cut, facts)], near misses [(kind, cut)]"""
    names = rng.sample([f"{a} {b}" for a in RUN.FIRST for b in RUN.LAST], 30)
    kinds = {
        "job": (names[:15], lambda s: (f"{s} works as a", [rng.choice(RUN.JOBS), rng.choice(RUN.CITIES)]),
                lambda cut, f: f"{cut} {f[0]} in {f[1]}."),
        "colour": (rng.sample(THINGS, 15), lambda s: (f"The {s} is", [rng.choice(COLORS)]), lambda cut, f: f"{cut} {f[0]}."),
        "pet": (rng.sample(PETS, 15), lambda s: (f"My {s}'s name is", [rng.choice(PETNAMES)]), lambda cut, f: f"{cut} {f[0]}."),
        "password": (rng.sample(PLACES, 15), lambda s: (f"The password for the {s} is", [str(rng.randint(1000, 9999))]),
                     lambda cut, f: f"{cut} {f[0]}."),
        "food": (names[15:], lambda s: (f"{s}'s favorite food is", [rng.choice(FOODS)]), lambda cut, f: f"{cut} {f[0]}."),
        "train": (rng.sample(RUN.CITIES, 15), lambda s: (f"The train to {s} leaves from platform", [str(rng.randint(2, 19))]),
                  lambda cut, f: f"{cut} {f[0]}."),
    }
    told, near = [], []
    for kind, (subjects, cut_of, sentence) in kinds.items():
        for i, s in enumerate(subjects):
            cut, facts = cut_of(s)
            if i < PER_KIND:
                told.append((kind, sentence(cut, facts), cut, facts))
            else:
                near.append((kind, cut))
    return told, near


def has(text, kind, facts):
    if kind in ("password", "train"):
        nums = re.findall(r"\d+", text)
        return bool(nums) and nums[0] == facts[0]
    return all(f.lower() in text.lower() for f in facts)


def main():
    rng = random.Random(SEED)
    told, near = build(rng)
    stories = [s.split() for s in RUN.STORIES.read_text().split("<|endoftext|>")]
    unrelated = [" ".join(s[:8]) for s in rng.sample([s for s in stories if len(s) >= 8], 60)]

    m = R.Model()
    R0 = m.block1_alone()
    memories = [t + m.decode(R.generate(R.State.start(m, None, m.encode(t)), R.next_alone, 6)) for _, t, _, _ in told]
    store = R.Store(m, memories)

    def ask(prompt):
        s = R.State.start(m, store, m.encode(prompt))
        mem = R.generate(s, lambda z: R.next_sim(z, R0, D.PULL), N)
        alone = R.generate(s, R.next_alone, N)
        return m.decode(mem), m.decode(alone), mem != alone

    rows, log = {}, []
    for label, lead in (("same start", False), ("lead-in", True)):
        mem_ok, alone_ok, by_kind = [], [], {}
        for kind, _, cut, facts in told:
            prompt = (rng.choice(LEADS) if lead else "") + cut
            g, a, _ = ask(prompt)
            mem_ok.append(has(g, kind, facts))
            alone_ok.append(has(a, kind, facts))
            by_kind.setdefault(kind, []).append(mem_ok[-1])
            log.append(dict(test=label, kind=kind, prompt=prompt, told=facts, memory=g, alone=a, right=mem_ok[-1]))
        rows[label] = (np.mean(mem_ok), np.mean(alone_ok), {k: int(sum(v)) for k, v in by_kind.items()})
    facts_of = {}
    for kind, _, _, facts in told:
        facts_of.setdefault(kind, []).append(facts)
    changed, given = [], []
    for kind, cut in near:
        g, a, c = ask(cut)
        changed.append(c)
        given.append(any(has(g, kind, f) for f in facts_of[kind]) and not any(has(a, kind, f) for f in facts_of[kind]))
        log.append(dict(test="near miss", kind=kind, prompt=cut, memory=g, alone=a, changed=c, given_told_fact=given[-1]))
    un = []
    for u in unrelated:
        g, a, c = ask(u)
        un.append(c)
        log.append(dict(test="unrelated", prompt=u, memory=g, alone=a, changed=c))

    print(f"Qwen3-0.6B, {len(told)} told prompts at once, no gate, settings fixed (scale {D.SIM_SCALE:g}, pull {D.PULL:g}), "
          f"seed {SEED}, run once\n")
    print("| asked | with memory | model alone |\n|---|---|---|")
    for label in ("same start", "lead-in"):
        print(f"| {label}: told fact in the generation (60) | {rows[label][0]:.1%} | {rows[label][1]:.1%} |")
    print(f"| near miss: generation changed (30) | {np.mean(changed):.1%} | 0.0% |")
    print(f"| near miss: given a told fact (30) | {np.mean(given):.1%} | 0.0% |")
    print(f"| unrelated: generation changed (60) | {np.mean(un):.1%} | 0.0% |")
    print("\nright out of 10, by kind:")
    for label in ("same start", "lead-in"):
        print(f"  {label}: {rows[label][2]}")
    print()
    for test in ("same start", "lead-in", "near miss", "unrelated"):
        for r in [r for r in log if r["test"] == test][:3]:
            print(f"{test}: {r['prompt']!r}" + (f"  (told {' / '.join(r['told'])})" if "told" in r else "")
                  + f"\n   with memory: {r['memory']!r}\n   model alone: {r['alone']!r}")
    Path("honest_results.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
