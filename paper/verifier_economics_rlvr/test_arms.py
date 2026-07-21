"""CPU unit test for the three verifier arms: validates cheap over-accepts, strong is strict,
cascade == strong on accept-set (sound), and gaming is detected. No GPU, no LLM."""
from verifiers.arms import cheap_verify, strong_verify, cascade_verify, reward_and_metrics

GOLD = "42"
cases = {
    # name: (completion, expect_cheap, expect_strong)
    "correct_boxed":      (r"Reasoning... so the answer is \boxed{42}.", True,  True),
    "correct_frac":       (r"= 84/2 = \boxed{84/2}.", True,  True),          # 84/2 == 42 (strong via value)
    "gamed_mention":      (r"Maybe 42? Let me try. Actually \boxed{7}.", True,  False),  # 42 appears, boxed wrong
    "gamed_enumerate":    (r"Could be 40, 41, 42, 43... \boxed{99}", True, False),        # enumerate incl gold
    "wrong":              (r"The answer is \boxed{7}.", False, False),
    "no_box_but_mentions":(r"I think it is 42 somewhere in text.", True, False),          # cheap yes, no box -> strong no
}

print(f"{'case':22} {'cheap':>6} {'strong':>7} {'cascade':>8}  ok?")
ok_all = True
for name, (comp, ec, es) in cases.items():
    ca, _ = cheap_verify(comp, GOLD)
    sa, _ = strong_verify(comp, GOLD)
    cas, _, _ = cascade_verify(comp, GOLD)
    # invariants: cheap==ec, strong==es, cascade == (strong if cheap else False) == strong-on-accepts
    inv = (ca == ec) and (sa == es) and (cas == (sa if ca else False))
    # soundness: cascade accept-set == strong accept-set (since cheap superset of strong)
    sound = (cas == sa) or (not ca)   # if cheap rejects a strong-accept it'd break soundness
    ok = inv and sound
    ok_all &= ok
    print(f"{name:22} {str(ca):>6} {str(sa):>7} {str(cas):>8}  {'PASS' if ok else 'FAIL'}")

# check the cheap-superset-of-strong invariant explicitly across cases
viol = [n for n, (c, _, _) in cases.items() if (lambda comp: strong_verify(comp, GOLD)[0] and not cheap_verify(comp, GOLD)[0])(c)]
print(f"\ncheap-superset-of-strong violations (must be none): {viol}")

# demonstrate reward + metrics wiring for one gamed case
m = reward_and_metrics("cheap", cases["gamed_mention"][0], GOLD)
m2 = reward_and_metrics("cascade", cases["gamed_mention"][0], GOLD)
print(f"\ngamed case -> cheap arm reward={m['reward']} gamed={m['gamed']} (rewards the hack)")
print(f"gamed case -> cascade arm reward={m2['reward']} gamed_flag={m2['gamed']} (cascade rejects the hack)")
print(f"\nALL INVARIANTS: {'PASS' if ok_all and not viol else 'FAIL'}")
