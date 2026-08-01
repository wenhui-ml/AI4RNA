#!/usr/bin/env python3
"""R10 (reviewer Q3): the six-model table under the conservative accounting.

Each invented operator is credited with its panel-unanimous count over the 60-target pool, but four of the six
runs contain per-target timeouts, and a timeout is not a solve. The question is what the table looks like if
timeouts are charged as failures against the full 60 rather than dropped from the denominator -- that is,
whether the qualitative claim (every model clears the gamer) and the apparent ordering survive the change.

Both columns are printed. The paper ranks by neither: these are single temperature-0.4 draws with no seed
replication, and the two models later replicated span 10-18 on this metric across seeds, a range wider than
most gaps in the table. Offline over the stored per-model results.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [("anthropic_claudeopus48", "Claude-Opus-4.8"), ("deepseek_deepseekv4pro", "DeepSeek-v4-pro"),
          ("google_gemini31propreview", "Gemini-3.1-pro"), (None, "GLM-5.2"),
          ("moonshotai_kimik3", "Kimi-k3"), ("openai_gpt56sol", "GPT-5.6-sol")]


def main():
    print(f"  {'model':18} {'completed':>10} {'timeout':>8} {'unan':>6} {'/completed':>11} {'/60 charged':>12}")
    rows = []
    for folder, name in sorted(MODELS, key=lambda x: x[1]):
        p = (os.path.join(HERE, "invent_operator", folder, "result.json") if folder
             else os.path.join(HERE, "invent_operator", "result.json"))
        if not os.path.exists(p):
            print(f"  {name:18} (missing)")
            continue
        ho = json.load(open(p)).get("heldout") or []
        done = [r for r in ho if not r.get("error")]
        unan = sum(1 for r in done if r.get("pkiss") and r.get("probknot") and r.get("shapeknots"))
        rows.append((name, len(done), len(ho) - len(done), unan))
        print(f"  {name:18} {len(done):10d} {len(ho)-len(done):8d} {unan:6d} "
              f"{unan/max(1,len(done)):11.3f} {unan/60:12.3f}")
    print("\n  references: gamer 1/60 = 0.017 | CF floor 0/60 | hand-built panel_auto 3-8/60")
    if rows:
        c = sum(1 for _, _, _, u in rows if u / 60 > 1 / 60)
        print(f"  models clearing the gamer with timeouts charged as failures: {c}/{len(rows)}")
        best_c = max(rows, key=lambda r: r[3] / max(1, r[1]))[0]
        best_g = max(rows, key=lambda r: r[3])[0]
        print(f"  top model by /completed: {best_c}; by /60 charged: {best_g}. The ordering flips under the")
        print("  accounting change, which is the clearest statement of why there is no ordering here.")


if __name__ == "__main__":
    main()
