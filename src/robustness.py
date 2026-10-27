"""Re-run the whole experiment on many freshly generated ledgers.

One ledger has only 29 planted cases, so a single case moves case recall by
about 3.4 points. Differences between methods smaller than that are noise.
Generating the books again with different random seeds (same shop, same kinds
of problems, different amounts, dates and picks) shows how much of a result is
the method and how much is luck.

Run:  python src/robustness.py            (30 seeds, about a minute)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect as D  # noqa: E402
import generate_ledger as G  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "robustness.csv"


def one_run(seed: int, k: int = 50) -> list[dict]:
    coa, users, lines, labels = G.generate(seed)
    e = D.build_entries(lines, users)
    r = D.risk_table(e)
    methods = {
        "Combined": r.risk,
        "Rules only": r.rule_score + r.if_pct * 1e-6,
        "Isolation Forest only": r.if_score,
    }
    rows = []
    for name, score in methods.items():
        ev = D.evaluate(score, labels, ks=(k,)).iloc[0]
        rows.append({"seed": seed, "method": name, "precision": ev.precision,
                     "case_recall": ev.case_recall})
    return rows


def run(seeds=range(1, 31), k: int = 50) -> pd.DataFrame:
    return pd.DataFrame([row for s in seeds for row in one_run(s, k)])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("method")[["precision", "case_recall"]]
    out = pd.concat({"mean": g.mean(), "worst": g.min(), "best": g.max()}, axis=1)
    return out.swaplevel(axis=1).sort_index(axis=1, level=0)


if __name__ == "__main__":
    df = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(summarize(df).round(3))
    wide = df.pivot(index="seed", columns="method", values="case_recall")
    print("Combined beats rules only on",
          int((wide["Combined"] > wide["Rules only"]).sum()), "of", len(wide), "seeds;",
          "ties on", int((wide["Combined"] == wide["Rules only"]).sum()))
