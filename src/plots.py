"""Charts for the README and notebook. One quiet style, three series colors."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#8f8e89"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.major.size": 0, "ytick.major.size": 0,
    "font.size": 10, "legend.frameon": False, "legend.labelcolor": TEXT_2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def _subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=TEXT_2, fontsize=9.5, va="bottom")


def monthly_sales(lines: pd.DataFrame, path=None):
    s = lines[lines.account.isin([4000, 4100])].copy()
    s["month"] = pd.to_datetime(s.posting_date).dt.to_period("M")
    m = (s.credit - s.debit).groupby(s.month).sum()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(m))
    ax.bar(x, m.values / 1000, width=0.62, color=BLUE, zorder=2)
    ax.set_xticks(x[::3], [p.strftime("%b %Y") for p in m.index[::3]])
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:.0f}K")
    ax.set_title("Monthly revenue, Two Gulls Creamery", pad=26)
    _subtitle(ax, "Summer carries the year. And one winter month makes no sense at all.")
    ax.axhline(0, color=TEXT_2, linewidth=0.8, zorder=3)
    low = int(np.argmin(m.values))
    ax.annotate(f"{m.index[low].strftime('%b %Y')}:\nnegative revenue",
                (x[low], 0), xytext=(x[low], 50), textcoords="data",
                color=TEXT, fontsize=9, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    return fig


def benford_two_digits(table: pd.DataFrame, spikes, path=None):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(table.digits, table.observed * 100, width=0.7, color=BLUE, zorder=2,
           label="Observed share")
    ax.plot(table.digits, table.expected * 100, color=TEXT_2, zorder=3,
            label="Benford's law")
    for d in spikes:
        row = table[table.digits == d].iloc[0]
        ax.annotate(str(d), (d, row.observed * 100), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=TEXT, fontsize=8.5)
    ax.set_xticks(range(10, 100, 10))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("First two digits of the amount")
    ax.set_title("First-two-digits test on hand-entered amounts", pad=26)
    _subtitle(ax, f"{table.attrs['n']} amounts. Labelled bars occur more often than "
                  "Benford predicts (p < 0.01).")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    return fig


def review_curve(curves: dict, baseline: pd.Series, max_k: int = 200, path=None):
    """curves: name -> Series indexed by k (entries reviewed) of share of cases found.
    baseline: the expected share found when entries are reviewed in random order."""
    colors = {"Combined": BLUE, "Rules only": ORANGE, "Isolation Forest only": AQUA}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    k = np.arange(0, max_k + 1)
    ax.plot(k, baseline.reindex(k).to_numpy() * 100, color=GRAY, linewidth=1.2,
            label="Random order")
    ends = []
    for name, s in curves.items():
        s = s.reindex(k).ffill().fillna(0)
        ax.plot(k, s * 100, color=colors[name], label=name, zorder=3)
        ends.append((name, s.iloc[-1] * 100))
    # direct labels at the right edge; lines that end together share one label
    grouped = {}
    for name, y in ends:
        grouped.setdefault(round(y, 1), []).append(name)
    for y, names in grouped.items():
        ax.annotate(" and ".join(names), (max_k, y), xytext=(max_k + 4, y),
                    textcoords="data", va="center", color=TEXT, fontsize=9)
    ax.axvline(50, color=GRID, linewidth=1, zorder=1)
    ax.text(51, 3, "top 50", color=TEXT_2, fontsize=9)
    ax.set_xlim(0, max_k)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Journal entries reviewed, highest risk first")
    ax.set_title("Share of planted cases found", pad=26)
    _subtitle(ax, "A case counts as found when any of its entries appears in the list.")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.subplots_adjust(right=0.80)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    if path:
        fig.savefig(path, dpi=160)
    return fig


def case_recall_curve(scores: pd.Series, labels: pd.DataFrame, max_k: int = 200) -> pd.Series:
    order = list(scores.sort_values(ascending=False).index[:max_k])
    case_of = labels.groupby("je_id").case_id.apply(set)
    n_cases = labels.case_id.nunique()
    found, out = set(), [0.0]
    for je in order:
        found |= case_of.get(je, set())
        out.append(len(found) / n_cases)
    return pd.Series(out, index=range(0, len(out)))


def random_case_recall(n_entries: int, labels: pd.DataFrame, max_k: int = 200) -> pd.Series:
    """Expected share of cases found after reviewing k random entries:
    a case with s entries is missed with probability C(N-s, k) / C(N, k)."""
    from math import lgamma, exp

    def log_comb(n, r):
        return lgamma(n + 1) - lgamma(r + 1) - lgamma(n - r + 1)

    sizes = labels.groupby("case_id").je_id.nunique().to_numpy()
    out = []
    for k in range(0, max_k + 1):
        miss = [exp(log_comb(n_entries - s, k) - log_comb(n_entries, k)) for s in sizes]
        out.append(1 - float(np.mean(miss)))
    return pd.Series(out, index=range(0, max_k + 1))
