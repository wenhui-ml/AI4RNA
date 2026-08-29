#!/usr/bin/env python3
"""M39: the leakage audit run against the model's ACTUAL training sequences, and bounded.

Declaration m39_exposure_tiers_declaration.json, frozen and pushed BEFORE this file existed.
Offline after one 526 MB public download. $0 -- no API, no GPU.

C7 is the ledger's leakage channel, the one a reader demands first, and it is priced empty. Its
weakness was never the statistic: M26d streamed its comparison sequences from RMDB, a PROXY for
what RibonanzaNet saw. The competition's own train_data_QUICK_START.csv is obtainable from a
public mirror, so the audit can be run against the thing itself.

THE FILE IS A SUBSET (167,808 unique sequences against a full corpus of roughly two million), so
every count here is a LOWER BOUND on exposure. A design found here IS exposed; a design not found
here may still be in the part we do not have. Declaration K3 makes that a wording rule binding on
the manuscript, and this script prints it next to every count.

No k-mer index. M26d's K0 existed because an inverted index recorded "no candidate found" as
"identity 0.0". Every query here is compared against EVERY training sequence of sufficient length,
exhaustively and vectorised, so a zero is a measured zero and that failure mode cannot occur.
"""
from __future__ import annotations

import collections, csv, json, os, random, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26
import m26d_clean_subset as M26D

TRAIN = "/data/tmp/ribonanza/train_data_QUICK_START.csv"
DECL = os.path.join(HERE, "m39_exposure_tiers_declaration.json")
OUTJ = os.path.join(HERE, "m39_exposure_tiers.json")
D = json.load(open(DECL))
THR = D["constants"]["THRESH"]
TIERS = D["constants"]["TIERS"]
SEED = D["constants"]["SEED"]
NBOOT = D["constants"]["NBOOT"]
MIN_CLEAN = D["constants"]["MIN_CLEAN_INTERVENTIONS"]
CODE = {c: i for i, c in enumerate("ACGUTN")}
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def enc(s):
    return np.frombuffer(s.replace("T", "U").encode(), dtype=np.uint8)


# ------------------------------------------------------------------ exhaustive max identity
def build_bank(seqs):
    """training sequences grouped by length, as dense uint8 arrays. No index, no candidates."""
    by = collections.defaultdict(list)
    for i, s in enumerate(seqs):
        by[len(s)].append(i)
    bank = {}
    for L, idx in by.items():
        A = np.empty((len(idx), L), dtype=np.uint8)
        for r, i in enumerate(idx):
            A[r] = enc(seqs[i])
        bank[L] = (A, np.asarray(idx))
    return bank


def max_identity(query, bank):
    """(best identity, index of the training sequence that gave it). Exhaustive over every
    training sequence long enough to contain an equal-length substring."""
    q = enc(query); Lq = len(q)
    best, who = 0, -1
    for L, (A, idx) in bank.items():
        if L < Lq:
            continue                      # cannot contain an equal-length substring
        for o in range(L - Lq + 1):
            m = (A[:, o:o + Lq] == q).sum(axis=1)
            k = int(m.argmax()); v = int(m[k])
            if v > best:
                best, who = v, int(idx[k])
    return best / Lq, who


def max_identity_slow(query, seqs):
    """检查2 对拍: the same number, pure Python, no numpy, no shared code path."""
    Lq = len(query); best = 0.0
    for s in seqs:
        if len(s) < Lq:
            continue
        for o in range(len(s) - Lq + 1):
            w = s[o:o + Lq]
            m = sum(1 for x, y in zip(query, w) if x == y)
            if m / Lq > best:
                best = m / Lq
    return best


def dinuc_shuffle(s, rng):
    """Altschul-Erikson: preserves every dinucleotide count. A mononucleotide shuffle would be a
    weaker control, because identity to a training sequence is partly a composition statistic."""
    edges = collections.defaultdict(list)
    for a, b in zip(s, s[1:]):
        edges[a].append(b)
    last = s[-1]; verts = list(edges)
    for _try in range(200):
        lastedge = {v: rng.choice(edges[v]) for v in verts if v != last}
        ok = True
        for v in verts:
            if v == last:
                continue
            seen, u = set(), v
            while u != last:
                if u in seen or u not in lastedge:
                    ok = False; break
                seen.add(u); u = lastedge[u]
            if not ok:
                break
        if ok:
            break
    else:
        return None
    new = {}
    for v in verts:
        e = list(edges[v])
        if v != last:
            e.remove(lastedge[v])
        rng.shuffle(e)
        if v != last:
            e.append(lastedge[v])
        new[v] = e
    out = [s[0]]; v = s[0]; ix = collections.defaultdict(int)
    for _ in range(len(s) - 1):
        b = new[v][ix[v]]; ix[v] += 1; out.append(b); v = b
    return "".join(out)


def tier_of(x):
    if x < TIERS[0]:
        return 0
    if x < TIERS[1]:
        return 1
    if x < TIERS[2]:
        return 2
    return 3


def save(OUT):
    OUT["log"] = LOG
    json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False)
    open(OUTJ.replace(".json", ".log"), "w").write("\n".join(LOG) + "\n")


def main():
    t0 = time.time()
    log("=== M39:把 C7 的审计从代理换成模型真正的训练序列,并且给出界 ===")
    log("  声明已冻结并推送于本文件存在之前。一次 526 MB 公开下载之后全离线,$0。")
    log("  **这个文件是子集(16.8 万条 / 全集约两百万),所以每一个计数都是曝光的下界。**")
    log("  在这里找到 = 确实曝光;没找到 ≠ 没曝光。声明 K3 把这条写成论文的措辞规则。\n")
    OUT = {"declaration": os.path.basename(DECL), "constants": D["constants"],
           "bound_direction": "every count is a LOWER BOUND on exposure"}

    # ---------------- the training corpus
    csv.field_size_limit(10 ** 9)
    seq2lib = collections.defaultdict(set)
    for d in csv.DictReader(open(TRAIN)):
        lib = d["dataset_name"]
        for tag in ("_2A3", "_DMS"):          # the chemistry can sit at the END or in the MIDDLE
            lib = lib.replace(tag, "")
        lib = lib.strip("_").replace("__", "_")
        seq2lib[d["sequence"]].add(lib)
    seqs = sorted(seq2lib)
    log(f"  训练唯一序列 {len(seqs)},来源库 {len({l for v in seq2lib.values() for l in v})} 个 "
        f"({time.time()-t0:.0f}s)")
    OUT["n_train_unique"] = len(seqs)
    OUT["n_train_libraries"] = len({l for v in seq2lib.values() for l in v})

    bank = build_bank(seqs)
    log(f"  按长度分组 {len(bank)} 组")

    # ---------------- the queries
    recs, _dr = M26.build()
    win = {}
    for d in csv.DictReader(open(M26.SRC)):
        try:
            ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
        except (ValueError, KeyError):
            continue
        win[(d["puzzle"], d["method"])] = (ss - 1, se)
    zc = np.load(os.path.join(HERE, "m26c_pneural_cache.npz"), allow_pickle=True)
    pn = {k: v for k, v in zip(zc["keys"], zc["mats"])}
    miss = [r for r in recs if r["mut_seq"] not in pn]
    if miss:
        log(f"  **神经缓存缺 {len(miss)} 条 ⇒ 作废**")
        OUT["verdict"] = "void_neural_cache_incomplete"; save(OUT); return 1
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
    del pn, zc

    q = {}
    for r in recs:
        k = r["design"]
        if k in win and k not in q:
            lo, hi = win[k]
            q[k] = r["wt_seq"][lo:hi]
    QL = sorted({len(v) for v in q.values()})
    log(f"  设计单元 {len(q)},设计窗长度 {QL}")

    # THE DECLARATION IS WRONG HERE AND IS NOT BEING EDITED. Its length_note assumed a 177 nt
    # design window, which is the modal length of the TRAINING file, not of our queries. The
    # windows are {QL[0]}-{QL[-1]} nt. The error is in our favour: a shorter query is eligible to
    # match MORE training sequences, so the comparison set is larger than the declaration
    # anticipated and the bound is tighter, not looser. Recorded, not corrected in place.
    def eligible(Lq):
        return int(sum(A.shape[0] for L, (A, _i) in bank.items() if L >= Lq))
    el = {Lq: eligible(Lq) for Lq in QL}
    log(f"  可参与比对的训练序列(按查询长度):最短查询 {QL[0]} nt -> {el[QL[0]]} 条,"
        f"最长查询 {QL[-1]} nt -> {el[QL[-1]]} 条,共 {len(seqs)} 条")
    log("  **声明的 length_note 写的是 177 nt 设计窗,那是训练文件的众数长度,不是我们查询的长度。**")
    log("  不改声明。这个错误的方向对我们有利:查询更短 ⇒ 可比对的训练序列更多 ⇒ 界更紧。")
    OUT["n_designs"] = len(q)
    OUT["query_lengths"] = QL
    OUT["eligible_train_by_query_length"] = {str(k): v for k, v in el.items()}
    OUT["declaration_erratum"] = ("length_note assumed a 177 nt design window; the actual windows "
                                  f"are {QL[0]}-{QL[-1]} nt. The declaration is left unedited. The "
                                  "error enlarges the comparison set, so the reported bound is "
                                  "tighter than the declaration anticipated.")

    # ---------------- V: verification, before anything is read
    log("\n=== V 验证(在读任何结果之前判)===")
    rng = random.Random(SEED)
    ks = sorted(q)
    smp = [ks[i] for i in random.Random(SEED).sample(range(len(ks)), min(6, len(ks)))]

    self_ok = all(abs(max_identity(q[k], build_bank([q[k]]))[0] - 1.0) < 1e-12 for k in smp)
    log(f"  检查1 零点恒等式:序列对自己 = 1.0,{len(smp)} 条  {'通过' if self_ok else '**失败**'}")

    sub_idx = random.Random(SEED + 1).sample(range(len(seqs)), min(3000, len(seqs)))
    sub = [seqs[i] for i in sub_idx]
    sub_bank = build_bank(sub)
    worst = 0.0
    for k in smp:
        a = max_identity(q[k], sub_bank)[0]
        b = max_identity_slow(q[k], sub)
        worst = max(worst, abs(a - b))
    obw_ok = worst < 1e-12
    log(f"  检查2 对拍(numpy 向量化 vs 纯 Python,同一 {len(sub)} 条子集,{len(smp)} 个查询):"
        f"最大偏差 {worst:.3e}  {'通过' if obw_ok else '**失败**'}")

    sh = []
    for k in ks:
        z = dinuc_shuffle(q[k], random.Random(SEED + hash(k) % 9973))
        if z:
            sh.append(max_identity(z, bank)[0])
    sh = np.asarray(sh)
    neg_ok = len(sh) > 0 and float(sh.max()) < THR
    log(f"  检查3 阴性对照(双核苷酸保持打乱,{len(sh)} 条):最大同一性 {sh.max():.4f},"
        f"中位 {np.median(sh):.4f}  阈值 {THR}  {'通过' if neg_ok else '**失败**'}")
    ok = bool(self_ok and obw_ok and neg_ok)
    OUT["V_verification"] = {"self_identity": bool(self_ok), "obweight_max": worst,
                             "shuffle_max": float(sh.max()) if len(sh) else None,
                             "shuffle_median": float(np.median(sh)) if len(sh) else None,
                             "pass": ok}
    log(f"  → {'验证通过,可以读数' if ok else '**K0 触发:本次运行作废**'}")
    if not ok:
        OUT["verdict"] = "void_K0_verification_failed"; save(OUT); return 1

    # ---------------- E1 + E2: exposure, and exposure graded
    log("\n=== E1/E2 曝光,以及分级曝光(阈值与分级边界由声明钉死)===")
    best, who = {}, {}
    for n, k in enumerate(ks, 1):
        best[k], who[k] = max_identity(q[k], bank)
    exposed = {k for k, v in best.items() if v >= THR}
    tiers = collections.Counter(tier_of(v) for v in best.values())
    log(f"  **曝光设计单元 {len(exposed)}/{len(q)} = {100*len(exposed)/len(q):.1f}%**  "
        f"(M26d 用 RMDB 代理时是 14/158 = 8.9%)")
    names = {0: f"T0 <{TIERS[0]}", 1: f"T1 [{TIERS[0]},{TIERS[1]})",
             2: f"T2 [{TIERS[1]},{TIERS[2]})", 3: f"T3 >={TIERS[2]} 逐字"}
    for t in range(4):
        log(f"    {names[t]:22s} {tiers.get(t,0):4d} 个设计单元")
    lib = collections.Counter()
    for k in ks:
        if best[k] >= THR and who[k] >= 0:
            for l in seq2lib[seqs[who[k]]]:
                lib[l] += 1
    log("  曝光归因到的来源库:")
    for l, c in lib.most_common(8):
        log(f"    {c:4d}  {l}")
    OUT["E1"] = {"n_exposed_designs": len(exposed), "n_designs": len(q),
                 "frac": round(len(exposed) / len(q), 4),
                 "m26d_proxy_count": 14, "m26d_proxy_frac": 0.0886}
    OUT["E2"] = {"tiers": {names[t]: tiers.get(t, 0) for t in range(4)},
                 "libraries": dict(lib.most_common()),
                 "identity_by_design": {f"{k[0]}|{k[1]}": round(best[k], 5) for k in ks}}

    for r in recs:
        r["_tier"] = tier_of(best.get(r["design"], 0.0))
    clean = [r for r in recs if r["design"] not in exposed]
    dirty = [r for r in recs if r["design"] in exposed]
    log(f"  未曝光干预 {len(clean)},曝光干预 {len(dirty)}")
    OUT["n_clean"] = len(clean); OUT["n_dirty"] = len(dirty)

    if len(clean) < MIN_CLEAN:
        log(f"\n  **K1 触发:未曝光干预 {len(clean)} < {MIN_CLEAN} ⇒ E3/E4 作废,只报 E1/E2**")
        OUT["verdict"] = "K1_clean_subset_too_small_only_exposure_reported"; save(OUT); return 0

    # ---------------- E3: the clean/exposed skill contrast, WITH an interval
    log("\n=== E3 未曝光 vs 曝光的技能对比(用 M26d 的 arm,逐条并列;差值给区间)===")
    res = {}
    for pkey, mname in (("nmut", "RibonanzaNet"), ("pmut", "ViennaRNA")):
        log(f"\n  --- {mname} ---")
        for tag, rs in (("all", recs), ("clean", clean), ("exposed", dirty)):
            res[f"{mname}_{tag}"] = M26D.arm(rs, f"{mname}/{tag}", pkey, log, SEED)
    OUT["E3_arms"] = res

    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    pzc = collections.defaultdict(list); pzd = collections.defaultdict(list)
    for r in clean:
        pzc[r["puzzle"]].append(r)
    for r in dirty:
        pzd[r["puzzle"]].append(r)
    E3d = {}
    for pkey, mname in (("nmut", "RibonanzaNet"), ("pmut", "ViennaRNA")):
        g = np.random.default_rng(SEED + 3); dif = []
        pc, pd = sorted(pzc), sorted(pzd)
        for _ in range(NBOOT):
            a = [r for i in g.integers(0, len(pc), len(pc)) for r in pzc[pc[i]]]
            b = [r for i in g.integers(0, len(pd), len(pd)) for r in pzd[pd[i]]]
            sa, _ = M26.slope_in(a, sb, pkey); sbv, _ = M26.slope_in(b, sb, pkey)
            if sa is not None and sbv is not None:
                dif.append(sa - sbv)
        dif = np.sort(np.asarray(dif))
        ci = [round(float(dif[int(0.025 * len(dif))]), 5),
              round(float(dif[int(0.975 * len(dif))]), 5)] if len(dif) >= 20 else [None, None]
        E3d[mname] = {"clean_minus_exposed_ci": ci, "n_boot": len(dif),
                      "ci_contains_zero": bool(ci[0] is not None and ci[0] <= 0 <= ci[1])}
        log(f"  {mname:14s} 未曝光 − 曝光 的自助区间 {ci}"
            f"{'  含 0 ⇒ 读不出差别' if E3d[mname]['ci_contains_zero'] else '  不含 0'}")
    OUT["E3_difference_ci"] = E3d

    # ---------------- E4: THE memorisation test M26d could not run
    log("\n=== E4 技能随曝光等级怎么走 —— 记忆假说的直接检验 ===")
    log("  记忆假说预测:等级越高,技能越高。")
    E4 = {}
    for pkey, mname in (("nmut", "RibonanzaNet"), ("pmut", "ViennaRNA")):
        row = {}
        for t in range(4):
            rs = [r for r in recs if r["_tier"] == t]
            if len(rs) < 50:
                row[f"T{t}"] = {"n": len(rs), "slope": None, "note": "n<50, not read"}
                log(f"  {mname:14s} T{t}: n={len(rs)} 太少,不读")
                continue
            s, _n = M26.slope_in(rs, sb, pkey)
            pzt = collections.defaultdict(list)
            for r in rs:
                pzt[r["puzzle"]].append(r)
            pk = sorted(pzt); g = np.random.default_rng(SEED + 40 + t); bs = []
            for _ in range(NBOOT):
                a = [x for i in g.integers(0, len(pk), len(pk)) for x in pzt[pk[i]]]
                v, _ = M26.slope_in(a, sb, pkey)
                if v is not None:
                    bs.append(v)
            bs = np.sort(np.asarray(bs))
            ci = [round(float(bs[int(0.025 * len(bs))]), 5),
                  round(float(bs[int(0.975 * len(bs))]), 5)] if len(bs) >= 20 else [None, None]
            row[f"T{t}"] = {"n": len(rs), "slope": round(s, 5) if s is not None else None, "ci": ci}
            log(f"  {mname:14s} T{t}: n={len(rs):5d}  斜率 {s:+.5f}  CI {ci}")
        E4[mname] = row
    OUT["E4_skill_by_tier"] = E4

    def monotone(row):
        v = [row[f"T{t}"]["slope"] for t in range(4) if row[f"T{t}"].get("slope") is not None]
        return len(v) >= 2 and all(x <= y for x, y in zip(v, v[1:]))
    mono = {m: monotone(E4[m]) for m in E4}
    log(f"  技能随等级单调上升?{mono}")
    OUT["E4_monotone_rising"] = mono

    # ---------------- staked, in the declaration's own words
    log("\n=== 声明里下的注,按声明自己的措辞判 ===")
    nslope = res.get("RibonanzaNet_clean", {}) or {}
    ns = nslope.get("slope")
    st = {
      "1": {"text": D["staked"]["1"], "pass": bool(len(exposed) > 14),
            "detail": f"真语料 {len(exposed)}/{len(q)} vs RMDB 代理 14/158"},
      "2": {"text": D["staked"]["2"], "pass": bool(tiers.get(3, 0) > 0),
            "detail": f"T3(逐字)设计单元 {tiers.get(3,0)} 个"},
      "3": {"text": D["staked"]["3"], "pass": not any(mono.values()),
            "detail": f"单调上升 = {mono}"},
      "4": {"text": D["staked"]["4"],
            "pass": bool(ns is not None and 0.15 <= ns <= 0.27),
            "detail": f"未曝光子集神经斜率 {ns}"},
    }
    for k in sorted(st):
        log(f"  下注{k} {'成立' if st[k]['pass'] else '**未成立**'} —— {st[k]['detail']}")
        log(f"        『{st[k]['text']}』")
    OUT["staked"] = st

    k2 = any(not E3d[m]["ci_contains_zero"] and (E3d[m]["clean_minus_exposed_ci"][1] or 0) < 0
             for m in E3d) or any(mono.values())
    if k2:
        OUT["verdict"] = "K2_exposure_raises_skill_C7_carries"
        log("\n  **K2 触发:技能随曝光上升 ⇒ C7 不再是空的,账本那一行要翻。**")
    else:
        OUT["verdict"] = "C7_survives_against_the_real_training_corpus_as_a_bound"
    log(f"\n判决:{OUT['verdict']}")
    log("  措辞规则(K3):不许写『不在训练集里』或『没有泄漏』。只许写"
        "『在我们能拿到的那部分训练语料里没有找到』,并把那部分有多大写出来。")
    log(f"用时 {time.time()-t0:.0f}s")
    save(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
