#!/usr/bin/env python3
"""M55: a second transformer, so the architecture axis stops being one model.

Declaration m55_second_transformer_declaration.json, frozen and pushed BEFORE this file existed.
Offline after one download, $0 -- no API and no external inference service.

m35's architecture axis is literally RibonanzaNet minus EternaFold. RNA-FM shares neither the
backbone nor the training corpus and ships a head its own authors trained, so the axis can be
recomputed with a different transformer and nothing is fitted here.

Primary head fixed in the declaration: SS/RNA-FM-ResNet_bpRNA.pth. The other three released heads
run as the declared sensitivity and the primary is not re-chosen.

Estimator, rank agreement and bootstrap: M53's, written there, verified here by V3.
"""
from __future__ import annotations

import collections, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

DECL = os.path.join(HERE, "m55_second_transformer_declaration.json")
OUTJ = os.path.join(HERE, "m55_second_transformer.json")
PCACHE = os.path.join(HERE, "m55_rnafm_cache.json")
RCACHE = os.path.join(HERE, "m55_recs_cache.json")
D = json.load(open(DECL))
SEED, NBOOT, NWORK = 20260901, 2000, 48

RNAFM = "/mnt/beegfs/code/RNA-FM"
WDIR = "/mnt/beegfs/weights/wenhui/rnafm"
HEADS = {"bpRNA": f"{WDIR}/SS/RNA-FM-ResNet_bpRNA.pth",
         "RNAStralign": f"{WDIR}/SS/RNA-FM-ResNet_RNAStralign.pth",
         "PDB-All": f"{WDIR}/SS/RNA-FM-ResNet_PDB-All.pth",
         "PDB-TR1": f"{WDIR}/SS/RNA-FM-ResNet_PDB-TR1.pth"}
SRC = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
PRIMARY = "bpRNA"          # fixed in the declaration, before any number existed

PUB = {"rf": 0.0539, "nmut": 0.0605, "pmut": -0.0304, "emut": -0.00569}
PUB_RANK = {"RibonanzaNet": 0.32947, "ViennaRNA": 0.25570, "EternaFold": 0.27649}
PUB_DEPTH = 0.32252
CAN = {"AU", "UA", "GC", "CG", "GU", "UG"}
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def _plain(o):
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    raise TypeError(repr(o))


def save(OUT):
    OUT["log"] = LOG
    json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False, default=_plain)
    log(f"\n写出 {OUTJ}")


# --------------------------------------------------------------- RNA-FM inference
_M = {"model": None, "bc": None, "head": None}


def _init(head_path, random_head):
    import torch
    sys.path.insert(0, RNAFM); sys.path.insert(0, os.path.join(RNAFM, "redevelop"))
    from model.baseline import Baseline
    torch.set_num_threads(2)
    m = Baseline(backbone_name="rna-fm", seqwise_predictor_name="none",
                 elewise_predictor_name="none",
                 pairwise_predictor_name="pc-resnet_1_sym_first:r-ss",
                 backbone_frozen=1, backbone_random_initialization=0)
    if not random_head:
        m.load_param("overall", head_path)
    m.eval()
    _M["model"] = m
    _M["bc"] = m.backbone_alphabet.get_batch_converter()


def _score(job):
    """(seq, [(i,j)...], want_top) -> (seq, {(i,j): p}, top pairs or None).

    The head emits an L x L logit map already trimmed to the sequence, so map index i is sequence
    position i. V4 is what checks that rather than this comment."""
    import torch
    seq, pairs, want_top = job
    _l, _s, toks = _M["bc"]([("s", seq)])
    with torch.no_grad():
        p = torch.sigmoid(_M["model"]({"token": toks})["r-ss"])[0].float().numpy()
    p = 0.5 * (p + p.T)
    got = {f"{i},{j}": float(p[i, j]) for i, j in pairs if i < len(seq) and j < len(seq)}
    top = None
    if want_top:
        L = len(seq); k = max(L // 2, 1)
        iu = np.triu_indices(L, k=4)
        v = p[iu]
        idx = np.argsort(-v)[:k]
        top = [[int(iu[0][t]), int(iu[1][t]), float(v[t])] for t in idx]
    return seq, got, top


def run_head(tag, jobs, random_head=False):
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time(); out = {}, {}
    probs, tops = {}, {}
    with ProcessPoolExecutor(max_workers=NWORK, initializer=_init,
                             initargs=(HEADS[tag], random_head)) as ex:
        for n, (seq, got, top) in enumerate(ex.map(_score, jobs, chunksize=4)):
            probs[seq] = got
            if top is not None:
                tops[seq] = top
            if (n + 1) % 1000 == 0:
                log(f"    {tag}{' (random head)' if random_head else ''} "
                    f"{n+1}/{len(jobs)}  {time.time()-t0:.0f}s")
    log(f"    {tag}{' (random head)' if random_head else ''} 完成 {len(probs)},"
        f"用时 {time.time()-t0:.0f}s")
    return probs, tops


# --------------------------------------------------------------- records
def load_records():
    """M26's records, with mut_seq, joined to M50's cached predictor columns by index.

    The join is by position and the assertion below is what makes that legitimate: both caches
    come from the same deterministic build, and every shared field must agree row for row."""
    # The torch env has no ViennaRNA and the ViennaRNA env has no torch, so the record cache is
    # built once under the latter (`--build-cache`) and only read here. Stat the source directly
    # rather than importing M26, which would pull in ViennaRNA on the cache path too.
    st = os.stat(SRC)
    stamp = {"size": st.st_size, "mtime": int(st.st_mtime)}
    if os.path.exists(RCACHE):
        c = json.load(open(RCACHE))
        if c.get("stamp") == stamp:
            log(f"  记录缓存命中({len(c['recs'])} 条)")
            return [{**r, "design": tuple(r["design"])} for r in c["recs"]]
    import m26_model_scored as M26
    log("  重建记录(ViennaRNA 配分函数,数分钟)")
    recs, _ = M26.build()
    c50 = json.load(open(os.path.join(HERE, "m50_recs_cache.json")))["recs"]
    assert len(recs) == len(c50), "记录数与 M50 缓存不一致"
    out = []
    for r, x in zip(recs, c50):
        assert (tuple(r["design"]) == tuple(x["design"]) and r["pt"] == x["pt"]
                and r["depth"] == x["depth"] and abs(r["rf"] - x["rf"]) < 1e-12), "索引对齐失败"
        out.append({"design": tuple(r["design"]), "puzzle": r["puzzle"], "pt": r["pt"],
                    "rf": r["rf"], "depth": r["depth"], "crossed": bool(r["crossed"]),
                    "a": int(r["a"]), "b": int(r["b"]),
                    "wt_seq": r["wt_seq"], "mut_seq": r["mut_seq"],
                    "nmut": x["nmut"], "pmut": x["pmut"], "emut": x["emut"]})
    json.dump({"stamp": stamp, "recs": [{**r, "design": list(r["design"])} for r in out]},
              open(RCACHE, "w"))
    return out


def main():
    from m53_probing_reference import matched, within_sd, wrho, pct, SF
    t0 = time.time()
    log("=== M55:第二个 transformer,让架构轴不再只是一个模型 ===")
    log("  声明已冻结并推送于本文件存在之前。无 API,无外部推理服务,$0。")
    log(f"  主头在声明里定死:{PRIMARY};另外三个作为已声明的敏感性一起跑")
    OUT = {"declaration": os.path.basename(DECL), "primary_head": PRIMARY,
           "cost": "offline after one weights download, $0 -- no API, no external inference"}
    recs = load_records()
    log(f"  干预 {len(recs)}  ({time.time()-t0:.0f}s)")

    # what has to be scored: every pair on its own mutant and its own wild type
    need = collections.defaultdict(set)
    for r in recs:
        need[r["mut_seq"]].add((r["a"], r["b"]))
        need[r["wt_seq"]].add((r["a"], r["b"]))
    wts = sorted({r["wt_seq"] for r in recs})
    jobs = [(s, sorted(need[s]), s in set(wts)) for s in sorted(need)]
    log(f"  需要打分的唯一序列 {len(jobs)},其中野生型 {len(wts)}")

    if os.path.exists(PCACHE):
        C = json.load(open(PCACHE))
        log(f"  RNA-FM 缓存命中({len(C['probs'])} 个头)")
    else:
        C = {"probs": {}, "tops": {}}
        for tag in ([PRIMARY] + [h for h in HEADS if h != PRIMARY]):
            log(f"  RNA-FM 推理,头 = {tag}")
            p, t = run_head(tag, jobs)
            C["probs"][tag] = p
            if tag == PRIMARY:
                C["tops"][tag] = t
        log("  RNA-FM 推理,随机头(V5 对照)")
        pr, tr = run_head(PRIMARY, [j for j in jobs if j[2]], random_head=True)
        C["probs"]["_random"] = pr; C["tops"]["_random"] = tr
        json.dump(C, open(PCACHE, "w"))
    P = C["probs"]; T = C["tops"]

    def key(i, j):
        return f"{i},{j}"

    # ---------------------------------------------------------- V, before any arm
    log("\n=== V 验证(先于任何 arm)===")
    miss = sum(1 for r in recs if key(r["a"], r["b"]) not in P[PRIMARY].get(r["mut_seq"], {}))
    log(f"  K3 未能打分的干预 {miss}/{len(recs)}")
    if miss > 0.05 * len(recs):
        OUT["VOID"] = "K3: 未打分比例超过 5%"; save(OUT); return 1

    vals = [P[PRIMARY][r["mut_seq"]][key(r["a"], r["b"])] for r in recs
            if key(r["a"], r["b"]) in P[PRIMARY].get(r["mut_seq"], {})]
    v1 = bool(min(vals) >= 0.0 and max(vals) <= 1.0)
    log(f"  V1 概率范围 [{min(vals):.5f}, {max(vals):.5f}]  -> {'通过' if v1 else '失败'}")

    # V4 index convention: decode the top L/2 pairs and count canonical ones
    def canon_frac(tops):
        n = c = 0
        for s in wts:
            for i, j, _p in tops.get(s, []):
                n += 1; c += (s[i] + s[j]) in CAN
        return (c / n) if n else 0.0
    cf = canon_frac(T[PRIMARY]); cfr = canon_frac(T["_random"])
    log(f"  V4 载入头的 top-L/2 配对里典型配对占比 {cf:.4f}(阈 0.90);"
        f"随机头 {cfr:.4f}")
    v4 = cf >= 0.90
    v5 = (cf - cfr) > 0.10
    log(f"  V5 权重确实进了模型:载入 {cf:.4f} vs 随机 {cfr:.4f}  "
        f"-> {'通过' if v5 else '失败'}")

    # V2 determinism, recomputed live on one sequence
    _init(HEADS[PRIMARY], False)
    s0 = wts[0]; pr0 = sorted(need[s0])[:3]
    a1 = _score((s0, pr0, False))[1]; a2 = _score((s0, pr0, False))[1]
    d2 = max(abs(a1[k] - a2[k]) for k in a1)
    cached = {k: P[PRIMARY][s0][k] for k in a1}
    d2c = max(abs(a1[k] - cached[k]) for k in a1)
    log(f"  V2 同序列两次 max|Δ| {d2:.2e};与缓存 max|Δ| {d2c:.2e}")
    v2 = d2 < 1e-6 and d2c < 1e-6

    OUT["V"] = {"V1_range": v1, "V2_determinism": bool(v2), "V4_canonical_fraction": bool(v4),
                "V5_weights_loaded": bool(v5), "canonical_frac": round(cf, 4),
                "canonical_frac_random_head": round(cfr, 4), "n_unscored": miss}
    if not (v1 and v2 and v4 and v5):
        OUT["VOID"] = "K2: V1/V2/V4/V5 未通过"; save(OUT); return 1

    # V3 reverse control on the three existing predictors and the depth baseline
    for r in recs:
        r["_s"] = SF(r)
        r["fmut"] = P[PRIMARY][r["mut_seq"]].get(key(r["a"], r["b"]))
        r["fwt"] = P[PRIMARY][r["wt_seq"]].get(key(r["a"], r["b"]))
        for h in HEADS:
            r["f_" + h] = P[h][r["mut_seq"]].get(key(r["a"], r["b"]))
    d3 = {}
    for k in ("rf", "nmut", "pmut", "emut"):
        c, _w, _u = matched(recs, k, SF)
        d3[k] = abs(c - PUB[k])
        log(f"  V3 {k:6s} 我 {c:+.5f}  论文 {PUB[k]:+.5f}  |Δ| {d3[k]:.2e}")
    bycell = collections.defaultdict(list)
    for r in recs:
        bycell[r["design"]].append(r)
    cells = [(k, v) for k, v in bycell.items() if len(v) >= 8]
    rk = {n: wrho(cells, kx) for n, kx in
          (("RibonanzaNet", "nmut"), ("ViennaRNA", "pmut"), ("EternaFold", "emut"))}
    dep = wrho(cells, "depth")
    for n in PUB_RANK:
        log(f"  V3 rank {n:14s} 我 {rk[n]:+.5f}  论文 {PUB_RANK[n]:+.4f}")
    log(f"  V3 rank depth          我 {dep:+.5f}  论文 {PUB_DEPTH:+.4f}")
    v3 = (max(d3.values()) < 1e-4
          and all(abs(rk[n] - PUB_RANK[n]) < 5e-4 for n in PUB_RANK)
          and abs(dep - PUB_DEPTH) < 5e-4)
    OUT["V"]["V3_reverse_control"] = bool(v3)
    if not v3:
        OUT["VOID"] = "K1: V3 未复现已发表的数"; save(OUT); return 1
    log("  V 全部通过 —— 现在才读 RNA-FM 的列")

    # ---------------------------------------------------------- A1 the second transformer's contrast
    log("\n=== A1 RNA-FM 在同一批干预、同一套分层上的身份对比 ===")
    cf_, w_, u_ = matched(recs, "fmut", SF)
    sd_ = within_sd(recs, "fmut", SF)
    log(f"  对比 {cf_:+.5f}  权重 {w_:.0f}  层 {u_}  层内 SD {sd_:.4f}  标准化 {cf_/sd_:+.5f}")
    pzs = sorted({r["puzzle"] for r in recs})
    bypz = collections.defaultdict(list)
    for r in recs:
        bypz[r["puzzle"]].append(r)
    rng = np.random.default_rng(SEED)
    b_f, b_n, b_e, b_ax_f, b_ax_n, b_dax = [], [], [], [], [], []
    for _ in range(NBOOT):
        ix = rng.integers(0, len(pzs), len(pzs))
        srs = [x for q in ix for x in bypz[pzs[q]]]
        f_ = matched(srs, "fmut", SF)[0]; n_ = matched(srs, "nmut", SF)[0]
        e_ = matched(srs, "emut", SF)[0]
        if None in (f_, n_, e_):
            continue
        sf_ = within_sd(srs, "fmut", SF); sn_ = within_sd(srs, "nmut", SF)
        se_ = within_sd(srs, "emut", SF)
        if not (sf_ and sn_ and se_):
            continue
        b_f.append(f_); b_n.append(n_); b_e.append(e_)
        b_ax_f.append(f_ / sf_ - e_ / se_); b_ax_n.append(n_ / sn_ - e_ / se_)
        b_dax.append((f_ / sf_ - e_ / se_) - (n_ / sn_ - e_ / se_))
    ci_f = pct(b_f)
    log(f"  A1 RNA-FM 对比 {cf_:+.5f}  95% {ci_f}  "
        f"{'含 0' if ci_f[0] <= 0 <= ci_f[1] else '不含 0'}")
    nc = matched(recs, "nmut", SF)[0]
    ci_fn = pct([f - n for f, n in zip(b_f, b_n)])
    log(f"  RibonanzaNet 对比 {nc:+.5f};RNA-FM − RibonanzaNet {cf_-nc:+.5f} {ci_fn}")
    OUT["A1_contrast"] = {"contrast": round(cf_, 5), "ci95": ci_f,
                          "ci_contains_zero": bool(ci_f[0] <= 0 <= ci_f[1]),
                          "within_sd": round(sd_, 5), "contrast_in_sd": round(cf_ / sd_, 5),
                          "weight": round(w_, 1), "strata": u_,
                          "vs_ribonanzanet": {"point": round(cf_ - nc, 5), "ci95": ci_fn,
                                              "smaller": bool(abs(cf_) < abs(nc))}}

    # ---------------------------------------------------------- A2 the architecture axis, twice
    log("\n=== A2 架构轴用第二个 transformer 重算(都在自己的单位里)===")
    sdn = within_sd(recs, "nmut", SF); sde = within_sd(recs, "emut", SF)
    ax_f = cf_ / sd_ - matched(recs, "emut", SF)[0] / sde
    ax_n = nc / sdn - matched(recs, "emut", SF)[0] / sde
    ci_axf, ci_axn, ci_d = pct(b_ax_f), pct(b_ax_n), pct(b_dax)
    log(f"  RibonanzaNet − EternaFold  {ax_n:+.5f}  {ci_axn}")
    log(f"  RNA-FM       − EternaFold  {ax_f:+.5f}  {ci_axf}  "
        f"{'含 0' if ci_axf[0] <= 0 <= ci_axf[1] else '不含 0'}")
    log(f"  两条轴之差                 {ax_f-ax_n:+.5f}  {ci_d}  "
        f"{'含 0' if ci_d[0] <= 0 <= ci_d[1] else '不含 0'}")
    OUT["A2_architecture_axis"] = {
        "ribonanzanet_minus_eterna": {"point": round(ax_n, 5), "ci95": ci_axn},
        "rnafm_minus_eterna": {"point": round(ax_f, 5), "ci95": ci_axf,
                               "ci_contains_zero": bool(ci_axf[0] <= 0 <= ci_axf[1]),
                               "positive": bool(ax_f > 0)},
        "difference_between_axes": {"point": round(ax_f - ax_n, 5), "ci95": ci_d,
                                    "ci_contains_zero": bool(ci_d[0] <= 0 <= ci_d[1])}}

    # ---------------------------------------------------------- A3 the level arm vs free depth
    log("\n=== A3 RNA-FM 的 level 臂对免费深度基线 ===")
    rk_f = wrho(cells, "fmut")
    log(f"  RNA-FM 格内秩一致 {rk_f:+.5f};depth {dep:+.5f};"
        f"三个已发表 {', '.join(f'{n} {rk[n]:+.4f}' for n in PUB_RANK)}")
    bp = collections.defaultdict(list)
    for k, v in cells:
        bp[v[0]["puzzle"]].append((k, v))
    pz3 = sorted(bp); rng3 = np.random.default_rng(SEED + 2)
    dfd = []
    for _ in range(NBOOT):
        ix = rng3.integers(0, len(pz3), len(pz3))
        cs = [c for q in ix for c in bp[pz3[q]]]
        a_, b2_ = wrho(cs, "fmut"), wrho(cs, "depth")
        if a_ is not None and b2_ is not None:
            dfd.append(a_ - b2_)
    ci_fd = pct(dfd)
    log(f"  RNA-FM − depth {rk_f-dep:+.5f}  {ci_fd}  "
        f"{'含 0' if ci_fd[0] <= 0 <= ci_fd[1] else '不含 0'}")
    OUT["A3_level_vs_depth"] = {"rnafm": round(rk_f, 5), "depth": round(dep, 5),
                                "difference": round(rk_f - dep, 5), "ci95": ci_fd,
                                "beats_depth": bool(ci_fd[0] > 0)}

    # ---------------------------------------------------------- A5 the other three heads
    log("\n=== A5 另外三个已发布的头(已声明的敏感性,主头不改)===")
    a5 = {}
    for h in HEADS:
        c2, _w2, _u2 = matched(recs, "f_" + h, SF)
        s2 = within_sd(recs, "f_" + h, SF)
        a5[h] = {"contrast": round(c2, 5), "within_sd": round(s2, 5),
                 "contrast_in_sd": round(c2 / s2, 5) if s2 else None,
                 "primary": h == PRIMARY}
        log(f"  {h:12s} 对比 {c2:+.5f}  层内 SD {s2:.4f}  标准化 {c2/s2:+.5f}"
            f"{'   <- 主头' if h == PRIMARY else ''}")
    signs = {np.sign(a5[h]["contrast"]) for h in HEADS}
    k4 = len(signs) > 1
    log(f"  四个头的符号{'不一致 —— K4 触发,正文必须写明符号依赖于头' if k4 else '一致'}")
    OUT["A5_head_sensitivity"] = {"heads": a5, "signs_disagree": bool(k4)}

    # ---------------------------------------------------------- prereg scoring
    S1 = bool(not (ci_f[0] <= 0 <= ci_f[1]))
    S2 = bool(ax_f > 0)
    S3 = bool(not (ci_fd[0] > 0))
    S4 = bool(abs(cf_) < abs(nc))
    log("\n=== 预注册记分 ===")
    det = [f"RNA-FM 对比 {cf_:+.5f} {ci_f}",
           f"RNA-FM − EternaFold {ax_f:+.5f} {ci_axf}",
           f"RNA-FM − depth {rk_f-dep:+.5f} {ci_fd}",
           f"|{cf_:+.5f}| vs RibonanzaNet |{nc:+.5f}|"]
    for i, (sid, okk) in enumerate((("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4))):
        log(f"  {sid}  {'成立' if okk else '**失败**'}   {det[i]}")
    OUT["staked"] = {f"S{i+1}": {"text": D["staked_predictions"][i]["text"],
                                 "pass": p, "detail": det[i]}
                     for i, p in enumerate([S1, S2, S3, S4])}
    OUT["verdict"] = ("architecture_axis_is_a_two_member_class" if (S1 and S2)
                      else "architecture_axis_is_a_property_of_ribonanzanet")
    log(f"\n  裁定:{OUT['verdict']}   {sum([S1,S2,S3,S4])}/4 成立   用时 {time.time()-t0:.0f}s")
    save(OUT)
    return 0


if __name__ == "__main__":
    if "--build-cache" in sys.argv:
        # run under the ViennaRNA env; writes the record cache and exits
        n = len(load_records())
        print(f"cache built: {n} records -> {RCACHE}")
        sys.exit(0)
    sys.exit(main())
