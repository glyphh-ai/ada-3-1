"""demo.py — tell Qwen things in prompts, then see later generations change.

Each told prompt is run once; the route of that pass (prompt plus the model's own
reply) is the memory. Nothing is trained and nothing is added to later prompts.

Run:  uv run python demo.py
      uv run python demo.py "The sky is red." -- "The sky is"
"""

import sys

import route as R

SIM_SCALE, ETA, N = 40.0, 10.0, 8      # the carried sim's scale; the step size of the surprise (R.next_surprise)
R.KAPPA = SIM_SCALE
PULL = 3000.0                          # the earlier pull on routes (R.next_sim), kept for the record
KAPPA, C0 = 100, 0.10              # the gated carried version (R.next_carry), kept for the record
GAMMA, SCALE, DELTA0 = 0.8, 1000, 0.12   # the earlier windowed version (R.next_route), kept for the record

TOLD = ["The sky is red.", "Mara Ellison works as a beekeeper in Lisbon.",
        "My dog's name is Biscuit and he loves carrots.", "The password for the garage is 4471.",
        "Once upon a day, there was a little dog named Max."]
ASKED = ["The sky is", "Mara Ellison works as a", "My dog's name is", "The password for the garage is", "Once upon a",
         "I looked up and saw that the sky is", "Yesterday I met Mara Ellison. Mara Ellison works as a",
         "Listen: The sky is", "Just so you know, my dog's name is",
         "The capital of France is", "She opened the door and", "Tobin Hale works as a", "The grass is"]


def main(told, asked):
    m = R.Model()
    R0 = m.block1_alone()
    memories = []
    for t in told:                      # prompt 1: the model reads it and replies; the whole pass is kept
        reply = m.decode(R.generate(R.State.start(m, None, m.encode(t)), R.next_alone, 6))
        memories.append(t + reply)
        print(f"told: {t}")
    store = R.Store(m, memories)
    print("\n| prompt | with memory | model alone |\n|---|---|---|")
    for a in asked:
        s = R.State.start(m, store, m.encode(a))
        mem = m.decode(R.generate(s, lambda z: R.next_surprise(z, ETA), N))
        alone = m.decode(R.generate(s, R.next_alone, N))
        print(f"| {a} | {mem.strip()} | {alone.strip()} |")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--" in args:
        main(args[:args.index("--")], args[args.index("--") + 1:])
    else:
        main(TOLD, ASKED)
