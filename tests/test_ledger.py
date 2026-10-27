import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import detect as D  # noqa: E402


@pytest.fixture(scope="module")
def data():
    lines, users, coa, labels = D.load()
    return lines, users, coa, labels, D.build_entries(lines, users)


def test_every_entry_balances(data):
    lines = data[0]
    sums = lines.groupby("je_id")[["debit", "credit"]].sum().round(2)
    assert (sums.debit == sums.credit).all()


def test_trial_balance_nets_to_zero(data):
    lines = data[0]
    assert round(lines.debit.sum() - lines.credit.sum(), 2) == 0


def test_every_line_uses_a_real_account(data):
    lines, _, coa = data[:3]
    assert set(lines.account) <= set(coa.account)


def test_labels_point_at_real_entries(data):
    lines, labels = data[0], data[3]
    assert set(labels.je_id) <= set(lines.je_id)


def _toy(**cols):
    base = dict(
        posting_date=pd.Timestamp("2025-03-31"),
        entry_ts=pd.Timestamp("2025-04-02 10:00"),
        source="GJ", doc_type="adjustment", created_by="mchen", approved_by="",
        vendor="", invoice_no="", description="x", amount=100.0, n_lines=2,
        accounts=(1000, 6900), is_human=True,
    )
    base.update(cols)
    return pd.DataFrame([base], index=["JE1"])


def test_post_close_boundary():
    on_time = _toy(entry_ts=pd.Timestamp("2025-04-10 23:59"))
    late = _toy(entry_ts=pd.Timestamp("2025-04-11 00:00"))
    assert not D.rule_post_close(on_time).iloc[0]
    assert D.rule_post_close(late).iloc[0]


@pytest.mark.parametrize("amount,expected", [(1000, True), (7500, True), (4200, False),
                                             (500, False), (1000.01, False)])
def test_round_amount(amount, expected):
    assert D.rule_round_amount(_toy(amount=amount)).iloc[0] == expected


@pytest.mark.parametrize("amount,expected", [(4499.99, False), (4500, True),
                                             (4999.99, True), (5000, False)])
def test_just_below_limit(amount, expected):
    assert D.rule_just_below_limit(_toy(amount=amount)).iloc[0] == expected


def test_system_entries_are_not_people():
    assert not D.rule_after_hours(_toy(is_human=False,
                                       entry_ts=pd.Timestamp("2025-04-02 02:15"))).iloc[0]


@pytest.mark.parametrize("a,b", [("CBD-1234", "CBD-1234A"), ("CBD-1234", "CBD-01234"),
                                 ("CBD-1234", "CBD1234"), ("CBD-1234", "CBD-1234 ")])
def test_invoice_numbers_match_despite_formatting(a, b):
    assert D._invoice_key("Casco Bay Dairy Co.", a) == D._invoice_key("Casco Bay Dairy Co.", b)


def test_benford_expected_shares_sum_to_one():
    t = D.benford_first_two(pd.Series([12.0, 345.0, 99.0, 1000.0]))
    assert abs(t.expected.sum() - 1) < 1e-9


def test_ranking_is_reproducible(data):
    e = data[4]
    a = D.risk_table(e).risk
    b = D.risk_table(e).risk
    pd.testing.assert_series_equal(a, b)
