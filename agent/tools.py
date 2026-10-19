import inspect
import math
import re

from scipy import stats


# ---------- helpers ----------
def _num(x):
    if isinstance(x, (int, float)):
        v = float(x)
    else:
        s = str(x).strip().replace(",", "").replace("%", "").replace("\u2212", "-")
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if not m:
            raise ValueError(f"not a number: {x!r}")
        v = float(m.group())
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"invalid number: {x!r}")
    return v


def _nums(xs, name="values"):
    if not isinstance(xs, (list, tuple)) or len(xs) == 0:
        raise ValueError(f"{name} must be a non-empty list of numbers (data missing?)")
    return [_num(x) for x in xs]


def _decimals(reported, decimals):
    if decimals is not None:
        return int(decimals)
    s = str(reported).strip().replace("%", "")
    return len(s.split(".")[1]) if "." in s else 0


def _result(method, computed, reported, decimals, summary, details, slack=0.0, match=None):
    tol = 0.5 * 10 ** (-decimals) + slack + 1e-9
    if reported is None:
        diff = None
    else:
        diff = round(computed - reported, 6)
    if match is None:
        match = reported is not None and abs(computed - reported) <= tol
    return {
        "method": method,
        "computed": round(computed, 6),
        "reported": reported,
        "difference": diff,
        "tolerance": round(tol, 6),
        "match": bool(match),
        "summary": summary,
        "details": details,
    }


# ---------- verification methods ----------
def recompute_mean(values, reported, decimals=None):
    vals, rep = _nums(values), _num(reported)
    d = _decimals(reported, decimals)
    mean = sum(vals) / len(vals)
    return _result(
        "recompute_mean", mean, rep, d,
        f"Mean of {len(vals)} values = {mean:.4f}; paper reports {rep}.",
        {"values_used": vals, "n": len(vals)}, slack=0.5 * 10 ** (-d),
    )


def recompute_sum(values, reported, decimals=None):
    vals, rep = _nums(values), _num(reported)
    d = _decimals(reported, decimals)
    total = sum(vals)
    return _result(
        "recompute_sum", total, rep, d,
        f"Sum of {len(vals)} values = {total:.4f}; paper reports {rep}.",
        {"values_used": vals},
    )


def weighted_mean(values, weights, reported, decimals=None):
    vals, w, rep = _nums(values), _nums(weights, "weights"), _num(reported)
    if len(vals) != len(w) or sum(w) == 0:
        raise ValueError("values and weights must have equal length and non-zero total weight")
    d = _decimals(reported, decimals)
    wm = sum(v * x for v, x in zip(vals, w)) / sum(w)
    return _result(
        "weighted_mean", wm, rep, d,
        f"Weighted mean = {wm:.4f}; paper reports {rep}.",
        {"values_used": vals, "weights_used": w}, slack=0.5 * 10 ** (-d),
    )


def improvement(new, baseline, reported, kind="absolute", decimals=None):
    n, b, rep = _num(new), _num(baseline), _num(reported)
    d = _decimals(reported, decimals)
    if kind == "relative_percent":
        if b == 0:
            raise ValueError("baseline is zero; relative improvement undefined")
        comp = (n - b) / abs(b) * 100.0
        text = f"Relative improvement = ({n} - {b}) / {b} * 100 = {comp:.4f}%"
    else:
        comp = n - b
        text = f"Absolute improvement = {n} - {b} = {comp:.4f}"
    return _result("improvement", comp, rep, d, f"{text}; paper reports {rep}.",
                   {"new": n, "baseline": b, "kind": kind}, slack=0.5 * 10 ** (-d))


def f1_from_pr(precision, recall, reported_f1, decimals=None):
    p, r, rep = _num(precision), _num(recall), _num(reported_f1)
    if p + r == 0:
        raise ValueError("precision + recall is zero")
    d = _decimals(reported_f1, decimals)
    f1 = 2 * p * r / (p + r)
    return _result("f1_from_pr", f1, rep, d,
                   f"F1 = 2PR/(P+R) = {f1:.4f}; paper reports {rep}.",
                   {"precision": p, "recall": r})


def ratio_percent(numerator, denominator, reported_percent, decimals=None):
    a, b, rep = _num(numerator), _num(denominator), _num(reported_percent)
    if b == 0:
        raise ValueError("denominator is zero")
    d = _decimals(reported_percent, decimals)
    pct = a / b * 100.0
    return _result("ratio_percent", pct, rep, d,
                   f"{a} / {b} = {pct:.4f}%; paper reports {rep}%.",
                   {"numerator": a, "denominator": b})


def welch_ttest(mean1, std1, n1, mean2, std2, n2, alpha=0.05,
                claimed_significant=True, reported_p=None, decimals=3):
    m1, s1, k1 = _num(mean1), _num(std1), _num(n1)
    m2, s2, k2 = _num(mean2), _num(std2), _num(n2)
    if k1 < 2 or k2 < 2:
        raise ValueError("each group needs n >= 2")
    t, p = stats.ttest_ind_from_stats(m1, s1, int(k1), m2, s2, int(k2), equal_var=False)
    significant = p < float(alpha)
    rep_p = None if reported_p is None else _num(reported_p)
    ok = (significant == bool(claimed_significant))
    if rep_p is not None:
        ok = ok and abs(p - rep_p) <= 0.5 * 10 ** (-int(decimals)) + 1e-9
    verdict = "significant" if significant else "NOT significant"
    return _result(
        "welch_ttest", float(p), rep_p, int(decimals),
        f"Welch t = {t:.3f}, two-sided p = {p:.4f} -> {verdict} at alpha = {alpha}. "
        f"Paper claims significant = {claimed_significant}.",
        {"t": round(float(t), 4), "alpha": float(alpha), "significant": bool(significant),
         "claimed_significant": bool(claimed_significant)},
        match=ok,
    )


def confidence_interval(mean, std, n, reported_low, reported_high, level=0.95, decimals=None):
    m, s, k = _num(mean), _num(std), int(_num(n))
    lo_r, hi_r = _num(reported_low), _num(reported_high)
    if k < 2:
        raise ValueError("n must be >= 2")
    d = _decimals(reported_low, decimals)
    tcrit = float(stats.t.ppf(0.5 + float(level) / 2, k - 1))
    margin = tcrit * s / math.sqrt(k)
    lo, hi = m - margin, m + margin
    tol = 0.5 * 10 ** (-d) + 1e-9
    ok = abs(lo - lo_r) <= tol and abs(hi - hi_r) <= tol
    return _result(
        "confidence_interval", lo, lo_r, d,
        f"{int(float(level)*100)}% CI = [{lo:.3f}, {hi:.3f}]; paper reports [{lo_r}, {hi_r}].",
        {"computed_low": round(lo, 4), "computed_high": round(hi, 4),
         "reported_high": hi_r, "t_critical": round(float(tcrit), 4)},
        match=ok,
    )


def p_from_t(t, df, reported_p, two_sided=True, decimals=3):
    tv, dfv, rep = _num(t), _num(df), _num(reported_p)
    p = stats.t.sf(abs(tv), dfv) * (2 if two_sided else 1)
    return _result("p_from_t", float(p), rep, int(decimals),
                   f"p from t={tv}, df={dfv} is {p:.4f}; paper reports {rep}.",
                   {"t": tv, "df": dfv, "two_sided": bool(two_sided)})


REGISTRY = {
    "recompute_mean": recompute_mean,
    "recompute_sum": recompute_sum,
    "weighted_mean": weighted_mean,
    "improvement": improvement,
    "f1_from_pr": f1_from_pr,
    "ratio_percent": ratio_percent,
    "welch_ttest": welch_ttest,
    "confidence_interval": confidence_interval,
    "p_from_t": p_from_t,
}

CATALOG = """
recompute_mean(values: list[number], reported: number, decimals: int|null)
    - Recompute an average from table values and compare with the reported average.
recompute_sum(values: list[number], reported: number, decimals: int|null)
weighted_mean(values: list, weights: list, reported: number, decimals: int|null)
improvement(new: number, baseline: number, reported: number,
            kind: "absolute"|"relative_percent", decimals: int|null)
    - Check 'X improves over Y by Z' (points or percent).
f1_from_pr(precision: number, recall: number, reported_f1: number, decimals: int|null)
ratio_percent(numerator: number, denominator: number, reported_percent: number,
              decimals: int|null)
    - e.g. accuracy = correct / total * 100.
welch_ttest(mean1, std1, n1, mean2, std2, n2, alpha: number,
            claimed_significant: bool, reported_p: number|null, decimals: int)
    - Test a significance claim from summary statistics (std = sample standard deviation).
confidence_interval(mean, std, n, reported_low, reported_high, level: number, decimals: int|null)
p_from_t(t, df, reported_p, two_sided: bool, decimals: int)
"""


def run_method(method, params):
    if method not in REGISTRY:
        raise ValueError(f"unknown method {method!r}")
    fn = REGISTRY[method]
    sig = inspect.signature(fn)
    clean = {k: v for k, v in (params or {}).items() if k in sig.parameters}
    missing = [n for n, p in sig.parameters.items()
               if p.default is inspect.Parameter.empty and n not in clean]
    if missing:
        raise ValueError(f"missing parameters for {method}: {missing}")
    return fn(**clean)
