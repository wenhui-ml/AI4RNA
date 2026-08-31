#!/usr/bin/env python3
"""Fetch the four RMDB deposits the M27 census found and this host does not yet hold.

Public data, no credentials, $0. M3SARS and M2SL5 are already cached; these are the rest of the
census hit list, i.e. every remaining library in the public record that contains an orientation
flip at all.
"""
import os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "rmdb_cache")
DETAIL = "https://rmdb.stanford.edu/detail/{}/"
WANT = ["15KREP_2A3_0001", "15KREP_DMS_0001", "15KLIB_2A3_0000", "15KLIB_DMS_0000",
        "15KREP_2A3_0000", "15KREP_DMS_0000", "TTYH2_DMS_0001"]


def fetch(url, dest, timeout=600):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return open(dest, "rb").read()
    req = urllib.request.Request(url, headers={"User-Agent": "rna-harness/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
    open(dest, "wb").write(b)
    return b


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    for rid in WANT:
        rdat = os.path.join(CACHE, f"{rid}.rdat")
        if os.path.exists(rdat) and os.path.getsize(rdat) > 0:
            print(f"{rid:20s} cached {os.path.getsize(rdat)/1e6:.1f} MB", flush=True); continue
        try:
            html = fetch(DETAIL.format(rid), os.path.join(CACHE, f"{rid}.html")).decode("utf8", "ignore")
            m = re.search(r'href="(https://[^"]*?%s\.rdat)"' % re.escape(rid), html)
            if not m:
                print(f"{rid:20s} NO RDAT LINK on detail page", flush=True); continue
            b = fetch(m.group(1), rdat)
            print(f"{rid:20s} got {len(b)/1e6:.1f} MB", flush=True)
        except Exception as e:
            print(f"{rid:20s} FAILED {type(e).__name__}: {e}", flush=True)
    print("done", flush=True)
