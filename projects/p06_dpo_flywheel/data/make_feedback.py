"""Generate the committed demo feedback log `feedback.jsonl` (~400 events, seeded).

Run: python -m projects.p06_dpo_flywheel.data.make_feedback
Mix: ~30% thumbs-down (some with corrections), ~8% prompts with PII, ~5% duplicate event_ids,
~3% spam, a few empty responses / too-short prompts, user_ids drawn so one heavy user hits the cap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TEMPLATES = {
    "billing": (["Why was I charged twice for order {n}?", "I was charged twice for order {n}, why?",
                 "Order {n} shows two charges on my card. Why?"],
                "I'm sorry about the double charge on order {n}. The second line is a temporary authorisation hold that your bank releases within 3-5 business days; you will only be billed once. If it is still there after 5 days, reply here and I will escalate it.",
                ["Please contact support.", "I can't help with billing questions.", "Charges are final."]),
    "shipping": (["Where is my order {n}? It's been {d} days.", "Order {n} still hasn't arrived after {d} days",
                  "Can you check the status of order {n}? Waiting {d} days now."],
                 "Order {n} left our kitchen and is with the courier; the tracking link in your confirmation email updates every 15 minutes. Since it has been {d} days, I have also raised a trace with the courier and you will get an update within 24 hours.",
                 ["It will arrive soon.", "Not sure, try again later.", "I don't know where your order is."]),
    "account": (["How do I reset my password?", "how do i reset my password", "I forgot my password, how can I reset it?"],
                "Open the app, tap Account > Security > Reset password, and we will email you a one-time link that is valid for 30 minutes. If the email does not arrive, check your spam folder or reply here and I can trigger it manually.",
                ["Go to settings.", "I can't help with account issues.", "Call us."]),
    "product": (["Can I substitute the sauce on the {p} pizza?", "Is it possible to swap the sauce on the {p}?",
                 "Can the {p} pizza come with a different sauce?"],
                "Yes. On the {p} you can switch the base sauce to marinara, garlic cream or BBQ at no extra cost: tap Customise on the item, choose Sauce, and pick the one you want before adding it to the cart.",
                ["No.", "Sauces cannot be changed.", "I'm unable to assist with menu questions."]),
    "refund": (["I want a refund for order {n}, the food arrived cold.", "Order {n} arrived cold, I'd like a refund",
                "Refund please for order {n} - everything was cold"],
               "I'm sorry order {n} arrived cold. I have issued a full refund to your original payment method; it will show within 5-7 business days. I have also added a 20% credit to your account for the next order.",
               ["Refunds are not possible.", "Please contact support.", "Cold food is normal in winter."]),
}
PIZZAS = ["Margherita", "Pepperoni Feast", "Veggie Supreme", "BBQ Chicken"]
PII_SNIPPETS = [" My email is {u}@example.com.", " Call me on 555-{a}-{b}.", " Card used: 4111{c}.",
                " Reach me at +1 (415) {a}-{b} or {u}@gmail.com."]
SPAM = ["Buy followers now click here http://bit.ly/xyz", "CRYPTO GIVEAWAY free money for everyone",
        "casino bonus work from home $$$"]


def make_feedback(n: int = 400, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    tags = list(TEMPLATES)
    orders = [str(40000 + int(x)) for x in rng.integers(0, 60, size=25)]
    users = [f"u{i:03d}" for i in range(60)]
    events: list[dict] = []
    for i in range(n):
        tag = tags[int(rng.integers(0, len(tags)))]
        prompts, good, bads = TEMPLATES[tag]
        fill = {"n": orders[int(rng.integers(0, len(orders)))], "d": int(rng.integers(2, 9)),
                "p": PIZZAS[int(rng.integers(0, len(PIZZAS)))]}
        prompt = prompts[int(rng.integers(0, len(prompts)))].format(**fill)
        down = rng.random() < 0.30
        user = users[0] if rng.random() < 0.12 else users[int(rng.integers(1, len(users)))]
        ev = {"event_id": f"ev{i:04d}", "ts": f"2026-10-{1 + i // 20:02d}T{(i % 20):02d}:{int(rng.integers(0, 60)):02d}:00Z",
              "conversation_id": f"c{int(rng.integers(0, 150)):04d}", "user_id": user, "prompt": prompt,
              "response": bads[int(rng.integers(0, len(bads)))].format(**fill) if down else good.format(**fill),
              "rating": "down" if down else "up", "tags": [tag],
              "tenant": "acme" if rng.random() < 0.6 else ("globex" if rng.random() < 0.6 else "initech")}
        if down and rng.random() < 0.40:
            ev["correction"] = good.format(**fill)
        if rng.random() < 0.08:
            ev["prompt"] += PII_SNIPPETS[int(rng.integers(0, len(PII_SNIPPETS)))].format(
                u=f"user{int(rng.integers(1, 999))}", a=int(rng.integers(100, 999)), b=int(rng.integers(1000, 9999)),
                c=int(rng.integers(10 ** 11, 10 ** 12 - 1)))
        r = rng.random()
        if r < 0.03:
            ev["prompt"] = SPAM[int(rng.integers(0, len(SPAM)))]
        elif r < 0.05:
            ev["response"] = ""
        elif r < 0.065:
            ev["prompt"] = "hi"
        events.append(ev)
    # exact duplicates (same event_id) interleaved, ~5%
    for j in range(n // 20):
        events.insert(int(rng.integers(0, len(events))), dict(events[int(rng.integers(0, n))]))
    return events


if __name__ == "__main__":
    out = Path(__file__).with_name("feedback.jsonl")
    evs = make_feedback()
    with open(out, "w", encoding="utf-8") as f:
        for e in evs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(evs)} events to {out}")
