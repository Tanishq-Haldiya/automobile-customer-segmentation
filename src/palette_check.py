"""Python twin of the data-viz palette validator.

Computes -- never eyeballs -- the measurable checks on a categorical palette:
  2.  OKLCH lightness band
  3.  OKLCH chroma floor
  4.  CVD separation: OKLab Delta E x100 under Machado-Oliveira-Fernandes (2009)
      protan/deutan simulation at severity 1.0 (tritan reported only)
  4b. Normal-vision floor: worst unsimulated Delta E on the active pair list
  5.  WCAG contrast of each mark against the chart surface

Run:  python -m src.palette_check "#0072B2,#E69F00" --mode light --pairs all
"""
from __future__ import annotations

import itertools
import math
import re
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}

# Keep in lockstep with the JS twin: ASCII whitespace plus the Unicode space
# separators both engines strip.
_WS = "[ \t\n\v\f\r   -     　]+"
_HEX = re.compile(r"^#?[0-9a-fA-F]{6}$")


def _strip(v: str) -> str:
    return re.sub("^%s|%s$" % (_WS, _WS), "", v)


def split_colors(raw: str) -> list:
    return [c for c in (_strip(x) for x in (raw or "").split(",")) if c]


def is_hex(v: str) -> bool:
    return bool(_HEX.match(v))


def _hex2srgb(h: str):
    h = _strip(h).lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin(h):
    return [_s2lin(c) for c in _hex2srgb(h)]


def _rel_lum(h):
    r, g, b = _lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_rel_lum(a), _rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]


def oklch(h):
    L, a, b = _oklab_from_lin(_lin(h))
    return L, math.hypot(a, b)


def _simulate(h, kind):
    r, g, b = _lin(h)
    M = MACHADO[kind]
    clamp = lambda c: max(0.0, min(1.0, c))
    return [clamp(M[i][0] * r + M[i][1] * g + M[i][2] * b) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = _oklab_from_lin(_simulate(h1, kind) if kind else _lin(h1))
    b = _oklab_from_lin(_simulate(h2, kind) if kind else _lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or DEFAULT_SURFACE[mode]
    for c in list(palette) + [surface]:
        if not is_hex(c):
            raise ValueError("not a 6-digit hex colour: %r" % c)

    lo, hi = BAND[mode]
    rows, failures, warnings = [], [], []

    for i, c in enumerate(palette):
        L, C = oklch(c)
        cr = contrast(c, surface)
        band_ok = lo <= L <= hi
        chroma_ok = C >= CHROMA_FLOOR
        contrast_ok = cr >= CONTRAST_MIN
        rows.append({
            "slot": i, "hex": c, "L": round(L, 3), "C": round(C, 3),
            "contrast": round(cr, 2),
            "band": "PASS" if band_ok else "FAIL",
            "chroma": "PASS" if chroma_ok else "FAIL",
            "contrast_check": "PASS" if contrast_ok else "WARN",
        })
        if not band_ok:
            failures.append("slot %d %s: L=%.3f outside %s band %s" % (i, c, L, mode, (lo, hi)))
        if not chroma_ok:
            failures.append("slot %d %s: chroma %.3f below floor %.2f" % (i, c, C, CHROMA_FLOOR))
        if not contrast_ok:
            warnings.append("slot %d %s: contrast %.2f:1 below %.1f -- needs labels or a table view"
                            % (i, c, cr, CONTRAST_MIN))

    idx = list(range(len(palette)))
    pairlist = (list(itertools.combinations(idx, 2)) if pairs == "all"
                else [(i, i + 1) for i in idx[:-1]])

    pair_rows = []
    for i, j in pairlist:
        a, b = palette[i], palette[j]
        p, d, t = (delta_e(a, b, "protan"), delta_e(a, b, "deutan"), delta_e(a, b, "tritan"))
        n = delta_e(a, b)
        worst_cvd = min(p, d)
        if worst_cvd >= CVD_TARGET:
            status = "PASS"
        elif worst_cvd >= CVD_FLOOR:
            status = "WARN"
            warnings.append("pair %d-%d (%s/%s): CVD dE %.1f in the %.0f-%.0f floor band -- "
                            "legal only with secondary encoding"
                            % (i, j, a, b, worst_cvd, CVD_FLOOR, CVD_TARGET))
        else:
            status = "FAIL"
            failures.append("pair %d-%d (%s/%s): CVD dE %.1f below floor %.0f"
                            % (i, j, a, b, worst_cvd, CVD_FLOOR))
        if n < NORMAL_FLOOR:
            status = "FAIL"
            failures.append("pair %d-%d (%s/%s): normal-vision dE %.1f below hard floor %.0f"
                            % (i, j, a, b, n, NORMAL_FLOOR))
        pair_rows.append({
            "pair": "%d-%d" % (i, j), "protan": round(p, 1), "deutan": round(d, 1),
            "tritan": round(t, 1), "normal": round(n, 1), "status": status,
        })

    return {
        "mode": mode, "surface": surface, "pairs": pairs,
        "slots": rows, "pairs_report": pair_rows,
        "failures": failures, "warnings": warnings,
        "ok": not failures,
    }


def report(palette, mode="light", surface=None, pairs="adjacent") -> bool:
    r = validate(palette, mode=mode, surface=surface, pairs=pairs)
    print("mode=%s surface=%s pairs=%s" % (r["mode"], r["surface"], r["pairs"]))
    print("%-5s %-9s %6s %6s %9s %6s %7s %9s"
          % ("slot", "hex", "L", "C", "contrast", "band", "chroma", "contrast"))
    for s in r["slots"]:
        print("%-5d %-9s %6.3f %6.3f %9.2f %6s %7s %9s"
              % (s["slot"], s["hex"], s["L"], s["C"], s["contrast"],
                 s["band"], s["chroma"], s["contrast_check"]))
    print()
    print("%-7s %8s %8s %8s %8s %7s" % ("pair", "protan", "deutan", "tritan", "normal", "status"))
    for p in r["pairs_report"]:
        print("%-7s %8.1f %8.1f %8.1f %8.1f %7s"
              % (p["pair"], p["protan"], p["deutan"], p["tritan"], p["normal"], p["status"]))
    print()
    for w in r["warnings"]:
        print("WARN:", w)
    for f in r["failures"]:
        print("FAIL:", f)
    print("RESULT:", "PASS" if r["ok"] else "FAIL")
    return r["ok"]


if __name__ == "__main__":
    args = sys.argv[1:]
    pal = split_colors(args[0]) if args else []
    mode = args[args.index("--mode") + 1] if "--mode" in args else "light"
    surf = args[args.index("--surface") + 1] if "--surface" in args else None
    prs = args[args.index("--pairs") + 1] if "--pairs" in args else "adjacent"
    sys.exit(0 if report(pal, mode=mode, surface=surf, pairs=prs) else 1)
