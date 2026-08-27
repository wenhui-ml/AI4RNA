#!/usr/bin/env python3
"""M27 survey: does any corpus other than OK7a contain minable interventions?

This is a SURVEY, not an experiment -- it establishes design-side facts before a declaration is
frozen, the same way the OK7a orientation-flip counts were established before M26 was declared.
No estimate of any effect is computed here and none may be.

The question it answers is narrow and decisive. Our instrument needs compensatory DOUBLE mutants:
two positions changed, together, at a pair. Standard mutate-and-map libraries are SINGLE-mutant
scans, so most M2 deposits cannot supply the intervention at all. Rather than guess which deposits
qualify, this walks RMDB and counts, per deposit, how many mutations each construct carries.

RMDB deposits encode that in the construct NAME -- `<family>` for a wild type, `<family>_<pos><from>-<to>`
for a mutant -- so the count is exact. The sequence route is kept as a fallback but it is the one
that misleads: constructs carry barcodes and padding outside the designed region, so whole-sequence
Hamming distance reports nothing on a library that is entirely made of mutants. We produced that
false negative first and it is the reason the name route exists.

That count IS the portability of the mining step. If it is zero everywhere outside OK7a, `mine`
does not transfer and the paper must say so.
"""
from __future__ import annotations

import argparse, collections, json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = "https://rmdb.stanford.edu/manifest.json"
DETAIL = "https://rmdb.stanford.edu/detail/{}/"
CACHE = os.path.join(HERE, "rmdb_cache")
CANON = {"AU", "CG", "GU"}

# deposits whose title/name suggests mutants; the spectrum below is what actually decides
MUT_RE = re.compile(r"mutate|mutant|M2|MMR|rescue|compensat|two-dimensional|2D", re.I)
# natural vs designed, for the limitation the second corpus is supposed to remove
NAT_RE = re.compile(r"riboswitch|ribozyme|tRNA|rRNA|ribosom|viral|HIV|IRES|intron|RNase P|SRP"
                    r"|glycine|adenine|guanine|TPP|SAM|flavin|P4-P6|tetrahymena", re.I)
DES_RE = re.compile(r"eterna|openknot|player|design|puzzle", re.I)


def fetch(url, dest=None, timeout=180):
    if dest and os.path.exists(dest) and os.path.getsize(dest) > 0:
        return open(dest, "rb").read()
    req = urllib.request.Request(url, headers={"User-Agent": "rna-harness/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
    if dest:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(b)
    return b


MUT = re.compile(r"_(\d+)([ACGUT])-([ACGUT])")


def by_name(text):
    """RMDB deposits encode the design cell and the mutations in the construct NAME.

    A wild type is `<family>`; a single mutant is `<family>_<pos><from>-<to>_...`; a compensatory
    double would carry two such tags. Counting them is exact and needs no window inference, which
    is what the sequence route needs and what we got wrong first: constructs carry barcodes and
    padding outside the designed region, so whole-sequence Hamming distance reports nothing on a
    library that is full of mutants.

    Returns (n_families, mutation-count histogram, n_double, n_orientation_flip) or None when the
    deposit does not name its constructs this way.
    """
    names = re.findall(r"name:([^\t;]+)", text)
    if len(names) < 2:
        return None
    seqs = [m.group(1).replace("T", "U") for l in text.splitlines()
            if l.startswith(("ANNOTATION_DATA", "DATA_ANNOTATION"))
            for m in [re.search(r"sequence:([ACGUTN]+)", l)] if m]
    hist, fams, wt_of, doubles = collections.Counter(), collections.Counter(), {}, []
    for k, n in enumerate(names):
        ms = MUT.findall(n)
        fam = MUT.split(n)[0]
        hist[min(len(ms), 4)] += 1
        fams[fam] += 1
        if not ms and k < len(seqs):
            wt_of.setdefault(fam, seqs[k])
        elif len(ms) == 2:
            doubles.append((fam, ms, seqs[k] if k < len(seqs) else None))
    flips = 0
    for _fam, ms, _sq in doubles:
        (_p1, f1, t1), (_p2, f2, t2) = ms
        # an orientation flip exchanges the two bases: each position's "to" is the other's "from"
        pt = "".join(sorted({f1.replace("T", "U"), f2.replace("T", "U")}))
        if f1 == t2 and f2 == t1 and pt in CANON:
            flips += 1
    return {"n_families": len(fams), "mutation_hist": dict(sorted(hist.items())),
            "n_double": len(doubles), "n_orientation_flip": flips,
            "median_family_size": sorted(fams.values())[len(fams) // 2] if fams else 0}


def constructs(text):
    """every per-row sequence a deposit declares, from its own annotations."""
    # RMDB writes per-row sequences under either key depending on the deposit's vintage;
    # keying on only one of them silently reports an empty corpus, which is not the same
    # thing as a corpus with no interventions.
    out = []
    for l in text.splitlines():
        if l.startswith("ANNOTATION_DATA") or l.startswith("DATA_ANNOTATION"):
            m = re.search(r"sequence:([ACGUTN]+)", l)
            if m:
                out.append(m.group(1).replace("T", "U"))
    if not out:
        m = re.search(r"^SEQUENCE\t(.+)$", text, re.M)
        if m:
            out = [m.group(1).strip().replace("T", "U")]
    return out


def spectrum(seqs):
    """Hamming spectrum against the deposit's own modal sequence, plus the flip count."""
    if len(seqs) < 2:
        return None
    L = collections.Counter(len(s) for s in seqs).most_common(1)[0][0]
    same = [s for s in seqs if len(s) == L]
    if len(same) < 2:
        return None
    wt = collections.Counter(same).most_common(1)[0][0]
    spec, doubles, flips = collections.Counter(), [], 0
    for s in same:
        d = [i for i in range(L) if s[i] != wt[i]]
        spec[min(len(d), 6)] += 1
        if len(d) == 2:
            i, j = d
            wpair = "".join(sorted({wt[i], wt[j]}))
            mpair = "".join(sorted({s[i], s[j]}))
            doubles.append((i, j))
            # an orientation flip: both positions swap, pair type preserved and canonical
            if wpair in CANON and wpair == mpair and s[i] == wt[j] and s[j] == wt[i]:
                flips += 1
    return {"n_constructs": len(same), "length": L, "spectrum": dict(sorted(spec.items())),
            "n_double": len(doubles), "n_orientation_flip": flips}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--all", action="store_true", help="scan every deposit, not just titled ones")
    ap.add_argument("--out", default=os.path.join(HERE, "m27_corpus_survey.json"))
    a = ap.parse_args()
    t0 = time.time()
    print("=== M27 调查:OK7a 之外还有没有可挖的干预 ===", flush=True)
    print("  这是调查,不是实验。不算任何效应量。\n", flush=True)

    man = json.loads(fetch(MANIFEST, os.path.join(CACHE, "manifest.json")).decode())
    print(f"RMDB manifest 共 {len(man)} 条沉积", flush=True)
    cand = man if a.all else [r for r in man if MUT_RE.search(
        " ".join(str(r.get(k, "")) for k in ("name", "pub_title", "rmdb_id")))]
    print(f"  扫描 {len(cand)} 条{'(全库,不按标题过滤)' if a.all else '(标题含突变线索)'}"
          f" —— **决定的是构建体命名里的突变数,不是标题**\n", flush=True)

    rows, nf = [], 0
    for n, r in enumerate(sorted(cand, key=lambda r: -(r.get("data_points") or 0))[:a.max], 1):
        rid = r["rmdb_id"]
        blurb = " ".join(str(r.get(k, "")) for k in ("name", "pub_title"))
        kind = ("natural" if NAT_RE.search(blurb) else
                "designed" if DES_RE.search(blurb) else "unclear")
        rec = {"rmdb_id": rid, "name": str(r.get("name"))[:70], "kind": kind}
        try:
            html = fetch(DETAIL.format(rid), os.path.join(CACHE, f"{rid}.html")).decode("utf8", "ignore")
            m = re.search(r'href="(https://[^"]*?%s\.rdat)"' % re.escape(rid), html)
            if not m:
                rec["skipped"] = "详情页没有 rdat 链接"
            else:
                txt = fetch(m.group(1), os.path.join(CACHE, f"{rid}.rdat")).decode("utf8", "ignore")
                sp = by_name(txt) or spectrum(constructs(txt))
                if sp is None:
                    rec["skipped"] = "既无构建体命名也无逐行序列注释"
                else:
                    rec.update(sp)
                    nf += sp["n_orientation_flip"]
        except Exception as e:                                                 # noqa: BLE001
            rec["skipped"] = f"抓取/解析出错 {repr(e)[:70]}"
        rows.append(rec)
        if "skipped" in rec:
            tail = "skip:" + rec["skipped"][:34]
        else:
            tail = (f"族 {rec.get('n_families', rec.get('n_constructs', 0)):5d}  "
                    f"突变数谱 {rec.get('mutation_hist', {})}  双突变 {rec['n_double']:5d}  "
                    f"取向翻转 {rec['n_orientation_flip']:5d}")
        if n % 5 == 0 or rec.get("n_double"):
            print(f"  {n:3d}/{min(len(cand), a.max)}  {rid:22s} {kind:9s} {tail}", flush=True)

    got = [r for r in rows if r.get("n_orientation_flip", 0) > 0]
    nat = [r for r in got if r["kind"] == "natural"]
    print(f"\n=== 结果({time.time()-t0:.0f}s)===", flush=True)
    print(f"  查了 {len(rows)} 条沉积,累计取向翻转干预 {nf} 次", flush=True)
    print(f"  有可挖干预的沉积 {len(got)} 条,其中**天然 RNA {len(nat)} 条**", flush=True)
    for r in sorted(got, key=lambda x: -x["n_orientation_flip"])[:25]:
        print(f"    {r['rmdb_id']:22s} {r['kind']:9s} 翻转 {r['n_orientation_flip']:5d}  "
              f"族 {r.get('n_families', r.get('n_constructs', 0)):5d}  "
              f"双突变 {r['n_double']:5d}  {r['name'][:40]}", flush=True)
    if not nat:
        print("\n  **天然语料里一次可挖的干预都没有 ⇒ `mine` 不可迁移到天然 RNA,"
              "论文必须照这么写,不许含糊。**", flush=True)
    json.dump({"n_deposits_scanned": len(rows), "total_flips": nf,
               "n_with_interventions": len(got), "n_natural_with_interventions": len(nat),
               "rows": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n写出 {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
