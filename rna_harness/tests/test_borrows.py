"""Tests for the OpenScience-borrowed utilities: guards (E4), prereg (S5), transfer_gate (S2/S4)."""
from rna_harness.experiments.guards import BudgetGuard, IterationGuard, in_main_process, guard_main_only
from rna_harness.experiments.prereg import Prereg, verdict
from rna_harness.meta.transfer_gate import TransferLedger


# ---- E4 guards ----
def test_budget_guard_calls():
    g = BudgetGuard(max_calls=3)
    assert g.ok()
    g.spend(3)
    assert not g.ok() and "call budget" in g.reason()


def test_budget_guard_remaining():
    g = BudgetGuard(max_calls=10)
    g.spend(4)
    assert g.remaining_calls() == 6


def test_iteration_guard_max():
    g = IterationGuard(max_iters=2)
    assert g.advance("a")[0] is True
    assert g.advance("b")[0] is True
    cont, reason = g.advance("c")
    assert cont is False and "max_iters" in reason


def test_iteration_guard_doom_loop():
    g = IterationGuard(max_iters=99, stall_patience=2)
    assert g.advance("x")[0] is True      # 1st x
    assert g.advance("x")[0] is True      # 2nd x
    cont, reason = g.advance("x")         # 3rd identical -> stall_patience(2)+1 identical
    assert cont is False and "doom loop" in reason


def test_iteration_guard_progress_ok():
    g = IterationGuard(max_iters=99, stall_patience=2)
    for s in ["a", "a", "b", "b", "c"]:   # never 3-in-a-row identical
        assert g.advance(s)[0] is True


def test_main_process_guard():
    assert in_main_process() is True
    guard_main_only()                      # must not raise from the main/test process


def test_budget_guard_bounded_flag():
    assert BudgetGuard(max_calls=5).bounded is True
    assert BudgetGuard(wall_s=1.0).bounded is True
    assert BudgetGuard().bounded is False     # neither limit => unbounded (footgun, flagged)


def test_iteration_guard_mutable_state_with_repr_default():
    # a dict mutated IN PLACE across iters: repr-default key must see the change (no false doom-loop)
    g = IterationGuard(max_iters=99, stall_patience=2)
    st = {"bp": 5}
    assert g.advance(st)[0] is True
    st["bp"] = 4
    assert g.advance(st)[0] is True
    st["bp"] = 3
    assert g.advance(st)[0] is True           # changed every time -> never a doom loop


def test_iteration_guard_mutable_state_stall_detected():
    g = IterationGuard(max_iters=99, stall_patience=2)
    st = {"bp": 5}
    assert g.advance(st)[0] is True
    assert g.advance(st)[0] is True           # unchanged value
    cont, reason = g.advance(st)              # 3rd identical value -> doom loop
    assert cont is False and "doom loop" in reason


# ---- S5 prereg ----
def test_verdict_null_within_noise():
    p = Prereg(name="mem", hypothesis="memory raises solve_rate", metric="solve_rate",
               direction="higher", baseline=0.58, noise_floor=0.05, min_effect=0.08, n=42)
    v = verdict(p, 0.60)                    # +0.02 < noise_floor 0.05
    assert v["verdict"] == "null"


def test_verdict_supported():
    p = Prereg(name="mem", hypothesis="h", metric="solve_rate", direction="higher",
               baseline=0.58, noise_floor=0.05, min_effect=0.08, n=42)
    v = verdict(p, 0.70)                     # +0.12 >= min_effect 0.08 and > noise
    assert v["verdict"] == "supported"


def test_verdict_refuted_wrong_direction():
    p = Prereg(name="mem", hypothesis="h", metric="mean_bp", direction="lower",
               baseline=5.0, noise_floor=0.3, min_effect=1.0, n=42)
    v = verdict(p, 6.0)                       # went UP by 1.0 but we needed it LOWER
    assert v["verdict"] == "refuted"


def test_verdict_insufficient_right_direction_small():
    # right direction, above noise, but below the pre-committed min_effect => 'insufficient' (not refuted)
    p = Prereg(name="mem", hypothesis="h", metric="solve_rate", direction="higher",
               baseline=0.58, noise_floor=0.03, min_effect=0.10, n=42)
    v = verdict(p, 0.63)                       # +0.05: >noise 0.03, <min_effect 0.10, right direction
    assert v["verdict"] == "insufficient"


def test_prereg_save_load(tmp_path):
    p = Prereg(name="m", hypothesis="h", metric="solve_rate", direction="higher",
               baseline=0.5, noise_floor=0.05, min_effect=0.1, n=10)
    fp = str(tmp_path / "prereg.json")
    p.save(fp)
    q = Prereg.load(fp)
    assert q.baseline == 0.5 and q.min_effect == 0.1 and q.created == p.created


# ---- S2/S4 transfer_gate ----
def test_transfer_learned_true():
    led = TransferLedger()
    # skill "A" used in 3 distinct episodes, all solved; not-A episodes fail -> reuse>=3, lift>0
    for i in range(3):
        led.record_episode(f"e{i}", items_used=["A"], solved=True)
    led.record_episode("e3", items_used=[], solved=False)
    led.record_episode("e4", items_used=[], solved=False)
    assert led.reuse("A") == 3
    assert led.lift("A") > 0
    assert led.is_learned("A", min_reuse=3, min_lift=0.0) is True


def test_transfer_not_learned_low_reuse():
    led = TransferLedger()
    led.record_episode("e0", items_used=["B"], solved=True)   # only 1 reuse
    led.record_episode("e1", items_used=[], solved=False)
    assert led.is_learned("B", min_reuse=3) is False


def test_transfer_not_learned_no_lift():
    led = TransferLedger()
    # "C" reused 3x but solve-rate with == without (no lift) -> null
    for i in range(3):
        led.record_episode(f"w{i}", items_used=["C"], solved=(i == 0))   # 1/3 solved with C
    for i in range(3):
        led.record_episode(f"n{i}", items_used=[], solved=(i == 0))      # 1/3 solved without C
    assert led.reuse("C") == 3
    assert led.lift("C") == 0.0
    assert led.is_learned("C", min_reuse=3, min_lift=0.0) is False


def test_transfer_save_load(tmp_path):
    led = TransferLedger()
    led.record_episode("e0", ["A"], True)
    fp = str(tmp_path / "led.json")
    led.save(fp)
    led2 = TransferLedger.load(fp)
    assert led2.reuse("A") == 1 and led2.episodes["e0"]["solved"] is True
