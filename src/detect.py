"""Journal-entry anomaly detection for the Two Gulls Creamery ledger.

Three layers, each usable on its own:

1. Rule-based journal-entry tests, the kind auditors run under AS 2401 / ISA 240
   (weekend postings, round amounts, entries just under an approval limit, ...).
   Each rule gives a reason a reviewer can read.
2. A Benford first-two-digits test on human-entered amounts, used to find
   amount ranges that occur far more often than they should.
3. An Isolation Forest over engineered entry features, which ranks entries by
   how unusual they are without being told what "unusual" means.

`risk_table()` combines them into one ranked review list.

Weights and thresholds were set from audit reasoning before the results were
looked at. They were not tuned against the answer key, and they should not be.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import IsolationForest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
APPROVAL_LIMIT = 5_000.00
CLOSE_DAYS = 10            # books for a month are closed 10 days after month-end
RARE_PAIR_MAX = 5          # a user/account pair seen this often or less is "rare"
NEW_VENDOR_WINDOW = 120    # days after a vendor's first appearance
BURN_IN_DAYS = 60          # vendors first seen in the first 60 days are not "new"
REVENUE_ACCOUNTS = {4000, 4100}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(data_dir: Path = DATA_DIR):
    lines = pd.read_csv(data_dir / "journal_entries.csv",
                        parse_dates=["posting_date", "entry_ts"],
                        dtype={"approved_by": str, "vendor": str, "invoice_no": str})
    for col in ("approved_by", "vendor", "invoice_no"):
        lines[col] = lines[col].fillna("")
    users = pd.read_csv(data_dir / "users.csv")
    coa = pd.read_csv(data_dir / "chart_of_accounts.csv")
    labels_path = data_dir / "anomaly_labels.csv"
    labels = pd.read_csv(labels_path) if labels_path.exists() else None
    return lines, users, coa, labels


def build_entries(lines: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """One row per journal entry. Opening balances are excluded from testing."""
    lines = lines[lines.source != "OPEN"]
    g = lines.groupby("je_id", sort=False)
    e = g.agg(
        posting_date=("posting_date", "first"),
        entry_ts=("entry_ts", "first"),
        source=("source", "first"),
        doc_type=("doc_type", "first"),
        created_by=("created_by", "first"),
        approved_by=("approved_by", "first"),
        vendor=("vendor", "first"),
        invoice_no=("invoice_no", "first"),
        description=("description", "first"),
        amount=("debit", "sum"),
        n_lines=("line_no", "count"),
    )
    e["amount"] = e["amount"].round(2)
    e["accounts"] = g["account"].apply(lambda s: tuple(sorted(set(s))))
    system_users = set(users.loc[users.is_system, "user_id"])
    e["is_human"] = ~e.created_by.isin(system_users)
    return e


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
_HOLIDAYS = None


def _holidays(e):
    global _HOLIDAYS
    if _HOLIDAYS is None:
        cal = USFederalHolidayCalendar()
        _HOLIDAYS = set(cal.holidays(e.entry_ts.min(), e.entry_ts.max()).date)
    return _HOLIDAYS


def rule_weekend_or_holiday(e):
    """Entered by a person on a weekend or federal holiday. The shop is open on
    weekends, so automated postings then are normal; people keying entries are not."""
    hol = _holidays(e)
    off = (e.entry_ts.dt.weekday >= 5) | e.entry_ts.dt.date.isin(hol)
    return e.is_human & off


def rule_after_hours(e):
    """Entered by a person before 7:00 or after 20:00."""
    h = e.entry_ts.dt.hour
    return e.is_human & ((h < 7) | (h >= 20))


def rule_round_amount(e):
    """$1,000 or more and an exact multiple of $500. Real invoices rarely land there."""
    c = (e.amount * 100).round().astype("int64")
    return (e.amount >= 1000) & (c % 50_000 == 0)


def rule_just_below_limit(e):
    """Entered by a person at 90-100% of the approval limit: the classic sign of
    splitting a payment to avoid sign-off."""
    return e.is_human & (e.amount >= 0.9 * APPROVAL_LIMIT) & (e.amount < APPROVAL_LIMIT)


def _invoice_key(vendor, invoice_no):
    m = re.search(r"\d+", invoice_no or "")
    return f"{vendor}|{int(m.group())}" if m else None


def rule_duplicate_invoice(e):
    """A vendor invoice whose number (ignoring formatting) was already booked, or
    that repeats the same vendor and amount within 14 days, plus its payment."""
    inv = e[(e.source == "AP") & (e.doc_type == "invoice")].sort_values("entry_ts")
    keys = inv.apply(lambda r: _invoice_key(r.vendor, r.invoice_no), axis=1)
    same_number = keys.duplicated(keep="first") & keys.notna()
    # same vendor, same amount, within 14 days, whatever the invoice number says
    prev = inv.groupby(["vendor", "amount"]).posting_date.shift()
    same_amount = (inv.posting_date - prev).dt.days <= 14
    dup_inv = inv.index[same_number | same_amount.fillna(False)]
    flagged = set(dup_inv)
    dup_refs = set(zip(e.loc[dup_inv, "vendor"], e.loc[dup_inv, "invoice_no"]))
    pays = e[(e.doc_type == "payment")]
    flagged |= set(pays.index[[(v, i) in dup_refs for v, i in zip(pays.vendor, pays.invoice_no)]])
    return e.index.to_series().isin(flagged)


def rule_approval_gap(e):
    """A payment or manual entry by a person, at or above the approval limit, with
    no approver or approved by the same person who entered it."""
    big = e.amount >= APPROVAL_LIMIT
    gap = (e.approved_by == "") | (e.approved_by == e.created_by)
    # The shop's control applies when money moves or a manual entry is made,
    # not when a vendor invoice is first recorded in AP.
    in_scope = ~((e.source == "AP") & (e.doc_type == "invoice"))
    return e.is_human & big & gap & in_scope


def rule_rare_user_account(e):
    """A person posting to an account they have touched in 5 or fewer entries in two
    years. Catches people working outside their job."""
    pairs = e[e.is_human][["created_by", "accounts"]].explode("accounts")
    counts = pairs.groupby(["created_by", "accounts"]).size()
    rare = counts[counts <= RARE_PAIR_MAX]
    rare_set = set(rare.index)
    hit = pairs.apply(lambda r: (r.created_by, r.accounts) in rare_set, axis=1)
    hit_ids = set(pairs.index[hit])
    return e.index.to_series().isin(hit_ids)


def rule_post_close(e):
    """Entered after the books for its posting month were closed."""
    close = (e.posting_date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=CLOSE_DAYS + 1))
    return e.entry_ts >= close


def rule_manual_revenue(e):
    """A manual journal entry that moves revenue. Revenue should come from the
    POS and invoicing, not from hand-keyed entries."""
    touches = e.accounts.apply(lambda a: bool(REVENUE_ACCOUNTS & set(a)))
    return (e.source == "GJ") & touches


def rule_new_vendor_spend(e):
    """A vendor that first appeared after the history window and was paid the
    approval limit or more within its first 120 days."""
    v = e[e.vendor != ""]
    first = v.groupby("vendor").posting_date.min()
    start = e.posting_date.min()
    new = first[first > start + pd.Timedelta(days=BURN_IN_DAYS)]
    flagged = set()
    for vendor, d0 in new.items():
        window = v[(v.vendor == vendor) & (v.posting_date <= d0 + pd.Timedelta(days=NEW_VENDOR_WINDOW))]
        spend = window[window.doc_type == "invoice"].amount.sum()
        if spend >= APPROVAL_LIMIT:
            flagged |= set(window.index)
    return e.index.to_series().isin(flagged)


# --------------------------------------------------------------------------- #
# Benford
# --------------------------------------------------------------------------- #
def benford_first_two(amounts: pd.Series) -> pd.DataFrame:
    """First-two-digits test (Nigrini). Returns one row per digit pair 10-99
    with observed and expected proportions and a z statistic."""
    a = amounts[amounts >= 10]
    ftd = a.apply(lambda x: int(f"{x:.10e}".replace(".", "")[:2]))
    n = len(ftd)
    d = np.arange(10, 100)
    expected = np.log10(1 + 1 / d)
    observed = ftd.value_counts().reindex(d, fill_value=0).to_numpy() / n
    # z with continuity correction (Nigrini 2012)
    num = np.abs(observed - expected) - 1 / (2 * n)
    z = np.clip(num, 0, None) / np.sqrt(expected * (1 - expected) / n)
    out = pd.DataFrame({"digits": d, "observed": observed, "expected": expected,
                        "count": (observed * n).round().astype(int), "z": z})
    out.attrs["n"] = n
    out.attrs["mad"] = float(np.mean(np.abs(observed - expected)))
    return out


def benford_first_digit(amounts: pd.Series) -> pd.DataFrame:
    a = amounts[amounts >= 1]
    fd = a.apply(lambda x: int(f"{x:.10e}"[0]))
    n = len(fd)
    d = np.arange(1, 10)
    expected = np.log10(1 + 1 / d)
    observed = fd.value_counts().reindex(d, fill_value=0).to_numpy() / n
    out = pd.DataFrame({"digit": d, "observed": observed, "expected": expected})
    out.attrs["n"] = n
    out.attrs["mad"] = float(np.mean(np.abs(observed - expected)))
    return out


MIRROR_DOC_TYPES = {"payment", "receipt", "tax_remittance", "petty_replenish", "accrual_reversal"}


def benford_population(e: pd.DataFrame) -> pd.DataFrame:
    """The amounts Benford's law can reasonably be expected to describe.

    Two standard exclusions (Nigrini): entries that only mirror another entry
    (a payment repeats its invoice amount and a reversal repeats its accrual,
    so counting both double-counts it),
    and fixed recurring amounts such as rent or a subscription, which are set
    by a contract rather than arising naturally. Left in, those two alone
    produce the biggest "spikes" in this ledger and drown out everything else.
    """
    p = e[e.is_human & ~e.doc_type.isin(MIRROR_DOC_TYPES)]
    key = p.vendor.where(p.vendor != "", p.description)
    repeats = p.groupby([key, p.amount]).amount.transform("size")
    return p[repeats < 3]


def rule_benford_spike(e, z_crit: float = 2.576):
    """Entered by a person, with an amount whose first two digits occur
    significantly more often than Benford's law predicts (p < 0.01)."""
    table = benford_first_two(benford_population(e).amount)
    spikes = set(table[(table.z > z_crit) & (table.observed > table.expected)].digits)
    ftd = e.amount.apply(lambda x: int(f"{x:.10e}".replace(".", "")[:2]) if x >= 10 else -1)
    return e.is_human & ftd.isin(spikes)


# name -> (function, weight). Weights reflect how specific each signal is:
# 3 = rarely innocent, 2 = worth a look, 1 = weak on its own.
RULES = {
    "duplicate_invoice": (rule_duplicate_invoice, 3),
    "post_close": (rule_post_close, 3),
    "approval_gap": (rule_approval_gap, 3),
    "rare_user_account": (rule_rare_user_account, 2),
    "manual_revenue": (rule_manual_revenue, 2),
    "new_vendor_spend": (rule_new_vendor_spend, 2),
    "just_below_limit": (rule_just_below_limit, 2),
    "weekend_or_holiday": (rule_weekend_or_holiday, 1),
    "after_hours": (rule_after_hours, 1),
    "round_amount": (rule_round_amount, 1),
    "benford_spike": (rule_benford_spike, 1),
}


def run_rules(e: pd.DataFrame) -> pd.DataFrame:
    hits = pd.DataFrame({name: fn(e).astype(bool) for name, (fn, _) in RULES.items()},
                        index=e.index)
    weights = pd.Series({name: w for name, (_, w) in RULES.items()})
    hits["rule_score"] = hits[list(RULES)].mul(weights).sum(axis=1)
    hits["reasons"] = hits[list(RULES)].apply(
        lambda r: ", ".join(n for n in RULES if r[n]), axis=1)
    return hits


# --------------------------------------------------------------------------- #
# Isolation Forest
# --------------------------------------------------------------------------- #
def features(e: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=e.index)
    f["log_amount"] = np.log10(e.amount.clip(lower=1))
    f["entry_hour"] = e.entry_ts.dt.hour + e.entry_ts.dt.minute / 60
    f["entry_weekday"] = e.entry_ts.dt.weekday
    f["lag_days"] = (e.entry_ts.dt.normalize() - e.posting_date).dt.days
    f["days_to_month_end"] = ((e.posting_date + pd.offsets.MonthEnd(0)) - e.posting_date).dt.days
    f["is_human"] = e.is_human.astype(int)
    f["n_lines"] = e.n_lines
    cents = (e.amount * 100).round().astype("int64")
    f["round_100"] = (cents % 10_000 == 0).astype(int)
    f["round_1000"] = (cents % 100_000 == 0).astype(int)
    # how often this user posts to these accounts (rarest account wins)
    pairs = e[["created_by", "accounts"]].explode("accounts")
    pc = pairs.groupby(["created_by", "accounts"]).size().rename("n")
    pairs = pairs.join(pc, on=["created_by", "accounts"])
    f["log_user_account_count"] = np.log10(pairs.groupby(level=0).n.min())
    # how common this exact combination of accounts is
    sig = e.accounts.astype(str)
    f["log_pattern_count"] = np.log10(sig.map(sig.value_counts()))
    # how long the vendor had existed when this entry was posted
    first = e[e.vendor != ""].groupby("vendor").posting_date.min()
    age = (e.posting_date - e.vendor.map(first)).dt.days
    f["vendor_age_days"] = age.fillna(365).clip(upper=365)
    for s in ["POS", "AP", "BANK", "GJ", "PAYROLL", "AR"]:
        f[f"src_{s}"] = (e.source == s).astype(int)
    return f


def isolation_scores(e: pd.DataFrame, seed: int = 42) -> pd.Series:
    X = features(e)
    model = IsolationForest(n_estimators=500, contamination="auto", random_state=seed)
    model.fit(X)
    return pd.Series(-model.score_samples(X), index=e.index, name="if_score")


# --------------------------------------------------------------------------- #
# Combined ranking and evaluation
# --------------------------------------------------------------------------- #
IF_WEIGHT = 3  # the forest's percentile rank (0-1) counts as up to one strong rule


def risk_table(e: pd.DataFrame) -> pd.DataFrame:
    hits = run_rules(e)
    iso = isolation_scores(e)
    out = e[["posting_date", "entry_ts", "source", "created_by", "approved_by",
             "vendor", "description", "amount"]].join(hits)
    out["if_score"] = iso
    out["if_pct"] = iso.rank(pct=True)
    out["risk"] = out.rule_score + IF_WEIGHT * out.if_pct
    return out.sort_values("risk", ascending=False)


def evaluate(scores: pd.Series, labels: pd.DataFrame, ks=(25, 50, 100)) -> pd.DataFrame:
    """Precision and recall when a reviewer works down the list to depth k.
    A case counts as found if any of its entries is in the top k."""
    order = scores.sort_values(ascending=False).index
    bad = set(labels.je_id)
    cases = labels.groupby("case_id").je_id.apply(set)
    rows = []
    for k in ks:
        top = set(order[:k])
        tp = len(top & bad)
        rows.append({"k": k,
                     "precision": tp / k,
                     "entry_recall": tp / len(bad),
                     "case_recall": float(np.mean([bool(s & top) for s in cases]))})
    return pd.DataFrame(rows).set_index("k")


def cases_found(scores: pd.Series, labels: pd.DataFrame, k: int) -> pd.DataFrame:
    top = set(scores.sort_values(ascending=False).index[:k])
    c = labels.groupby(["anomaly_type", "case_id"]).je_id.apply(lambda s: bool(set(s) & top))
    return c.groupby(level=0).agg(found="sum", cases="count")


if __name__ == "__main__":
    lines, users, coa, labels = load()
    e = build_entries(lines, users)
    r = risk_table(e)
    print(r.head(15)[["posting_date", "created_by", "amount", "risk", "reasons"]])
    if labels is not None:
        print(evaluate(r.risk, labels))
