"""Generate a synthetic general ledger for Two Gulls Creamery, a fictional
seasonal ice cream shop in Portland, Maine, with labelled anomalies planted in it.

Everything here is invented: the shop, the people, the vendors and every number.
The point is to have realistic, balanced books with a known answer key, so that
anomaly-detection methods can be measured instead of just demonstrated.

Outputs (in ./data):
    chart_of_accounts.csv  - the accounts the shop uses
    users.csv              - who can post entries and what their job is
    journal_entries.csv    - every journal line (one row per debit or credit)
    anomaly_labels.csv     - the answer key: which entries were planted, and why

Run:  python src/generate_ledger.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

SEED = 42
START = date(2024, 1, 1)
END = date(2025, 12, 31)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SALES_TAX_RATE = 0.08      # Maine tax on prepared food
CARD_SHARE = 0.85          # share of sales paid by card
CARD_FEE_RATE = 0.026      # processor fee on card sales
APPROVAL_LIMIT = 5_000.00  # payments and manual entries at or above this need approval

rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------- #
# Master data
# --------------------------------------------------------------------------- #
ACCOUNTS = {
    1000: ("Cash - Operating", "Asset"),
    1010: ("Petty Cash", "Asset"),
    1200: ("Accounts Receivable", "Asset"),
    1500: ("Equipment", "Asset"),
    1590: ("Accumulated Depreciation", "Asset"),
    2000: ("Accounts Payable", "Liability"),
    2100: ("Sales Tax Payable", "Liability"),
    2200: ("Payroll Liabilities", "Liability"),
    2300: ("Accrued Expenses", "Liability"),
    3000: ("Owner's Equity", "Equity"),
    3100: ("Owner Draws", "Equity"),
    4000: ("Sales - Scoop Shop", "Revenue"),
    4100: ("Sales - Catering", "Revenue"),
    5000: ("COGS - Dairy & Ingredients", "Expense"),
    5100: ("COGS - Cones & Packaging", "Expense"),
    6000: ("Wages", "Expense"),
    6010: ("Payroll Taxes", "Expense"),
    6100: ("Rent", "Expense"),
    6200: ("Utilities", "Expense"),
    6300: ("Repairs & Maintenance", "Expense"),
    6400: ("Marketing", "Expense"),
    6500: ("Card Processing Fees", "Expense"),
    6600: ("Insurance", "Expense"),
    6700: ("Software & Subscriptions", "Expense"),
    6800: ("Depreciation Expense", "Expense"),
    6900: ("Office & Misc", "Expense"),
}

USERS = [
    ("pos_system", "Point-of-sale nightly export", "System", True),
    ("payroll_svc", "Payroll provider feed", "System", True),
    ("jlopez", "Jamie Lopez", "Bookkeeper (AP, AR, bank)", False),
    ("mchen", "Morgan Chen", "Staff accountant (month-end close)", False),
    ("dsmith", "Dana Smith", "Owner", False),
    ("tpark", "Taylor Park", "Shift manager (petty cash only)", False),
]

# Vendors: name -> (expense account, payment terms in days)
VENDORS = {
    "Casco Bay Dairy Co.": (5000, 14),
    "Pine State Cone Supply": (5100, 30),
    "Harborview Properties LLC": (6100, 0),
    "Northern Light Power": (6200, 20),
    "Seacoast Refrigeration Service": (6300, 30),
    "Gullwing Print & Signs": (6400, 30),
    "Mainstay Mutual Insurance": (6600, 15),
    "ScoopPOS Software": (6700, 0),
}

# Average daily net sales (before tax) by month, year one.
MONTH_SALES = {1: 360, 2: 385, 3: 560, 4: 1040, 5: 1920, 6: 3040,
               7: 3840, 8: 3680, 9: 2080, 10: 1200, 11: 560, 12: 640}
# Staff on the payroll by month.
MONTH_STAFF = {1: 4, 2: 4, 3: 5, 4: 8, 5: 13, 6: 18,
               7: 18, 8: 18, 9: 12, 10: 8, 11: 5, 12: 5}

HOLIDAYS = set(
    USFederalHolidayCalendar().holidays(START, END).date
)


def cents(x: float) -> int:
    return int(round(x * 100))


def is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in HOLIDAYS


def next_business_day(d: date) -> date:
    while not is_business_day(d):
        d += timedelta(days=1)
    return d


def month_end(d: date) -> date:
    return (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).date()


def office_ts(d: date, start_h: int = 8, end_h: int = 17) -> datetime:
    """A timestamp during office hours on a business day on or after d."""
    d = next_business_day(d)
    minutes = int(rng.integers(start_h * 60, end_h * 60))
    return datetime.combine(d, time(minutes // 60, minutes % 60, int(rng.integers(0, 60))))


# --------------------------------------------------------------------------- #
# Ledger builder
# --------------------------------------------------------------------------- #
@dataclass
class Ledger:
    lines: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    next_id: int = 100001
    invoice_seq: dict = field(default_factory=dict)
    # running balances needed for remittances
    sales_tax_by_month: dict = field(default_factory=dict)
    payroll_liab_by_month: dict = field(default_factory=dict)

    def post(self, posting_date, entry_ts, source, doc_type, created_by, description,
             legs, approved_by="", vendor="", invoice_no=""):
        """legs: list of (account, debit_cents, credit_cents). Must balance."""
        dr = sum(l[1] for l in legs)
        cr = sum(l[2] for l in legs)
        if dr != cr:
            raise ValueError(f"Unbalanced entry {description}: {dr} vs {cr}")
        if dr <= 0:
            raise ValueError(f"Empty entry {description}")
        je_id = f"JE{self.next_id}"
        self.next_id += 1
        for i, (acct, d_c, c_c) in enumerate(legs, start=1):
            self.lines.append({
                "je_id": je_id,
                "line_no": i,
                "posting_date": posting_date,
                "entry_ts": entry_ts,
                "source": source,
                "doc_type": doc_type,
                "created_by": created_by,
                "approved_by": approved_by,
                "vendor": vendor,
                "invoice_no": invoice_no,
                "description": description,
                "account": acct,
                "account_name": ACCOUNTS[acct][0],
                "debit": d_c / 100,
                "credit": c_c / 100,
            })
        return je_id

    def label(self, case_id, anomaly_type, je_ids, explanation):
        for je in je_ids:
            self.labels.append({"je_id": je, "case_id": case_id,
                                "anomaly_type": anomaly_type,
                                "explanation": explanation})

    def new_invoice_no(self, vendor):
        prefix = {"Casco Bay Dairy Co.": "CBD", "Pine State Cone Supply": "PSC",
                  "Harborview Properties LLC": "HVP", "Northern Light Power": "NLP",
                  "Seacoast Refrigeration Service": "SRS", "Gullwing Print & Signs": "GPS",
                  "Mainstay Mutual Insurance": "MMI", "ScoopPOS Software": "SPS",
                  "Coastal Supply Partners LLC": "CSP", "Back Cove Marina": "BCM"}[vendor]
        n = self.invoice_seq.get(vendor, int(rng.integers(1000, 9000)))
        n += int(rng.integers(1, 40))
        self.invoice_seq[vendor] = n
        return f"{prefix}-{n}"

    # ---- composite transactions -------------------------------------------
    def ap_invoice_and_payment(self, vendor, inv_date, amount, account=None,
                               description=None, invoice_no=None, pay_date=None,
                               created_by="jlopez", inv_ts=None, pay_ts=None):
        """Book a vendor invoice to AP, then pay it. Returns (invoice_je, payment_je),
        or just (invoice_je,) when the payment would fall after the end of the data."""
        account = account or VENDORS[vendor][0]
        terms = VENDORS.get(vendor, (None, 30))[1]
        invoice_no = invoice_no or self.new_invoice_no(vendor)
        inv_date = next_business_day(inv_date)
        description = description or f"{vendor} - {ACCOUNTS[account][0]}"
        a = cents(amount)
        inv_ts = inv_ts or office_ts(inv_date + timedelta(days=int(rng.integers(0, 3))))
        inv = self.post(inv_date, inv_ts, "AP", "invoice", created_by,
                        f"{description} ({invoice_no})",
                        [(account, a, 0), (2000, 0, a)],
                        vendor=vendor, invoice_no=invoice_no)
        pay_date = pay_date or next_business_day(inv_date + timedelta(days=terms))
        if pay_date > END:          # still unpaid at year-end: stays in AP
            return (inv,)
        approver = "dsmith" if amount >= APPROVAL_LIMIT else ""
        pay_ts = pay_ts or office_ts(pay_date)
        pay = self.post(pay_date, pay_ts, "BANK", "payment", created_by,
                        f"Payment {vendor} ({invoice_no})",
                        [(2000, a, 0), (1000, 0, a)],
                        approved_by=approver, vendor=vendor, invoice_no=invoice_no)
        return inv, pay


L = Ledger()


# --------------------------------------------------------------------------- #
# Normal business
# --------------------------------------------------------------------------- #
def opening_balances():
    L.post(START, datetime(2024, 1, 2, 10, 5), "OPEN", "opening", "mchen",
           "Opening balances carried forward from FY2023",
           [(1000, cents(85_000), 0), (1010, cents(300), 0), (1500, cents(120_000), 0),
            (1590, 0, cents(30_000)), (3000, 0, cents(175_300))])


def shop_is_open(d: date) -> bool:
    if (d.month, d.day) in {(12, 25), (1, 1)}:
        return False
    if d in HOLIDAYS and d.month == 11 and d.weekday() == 3:   # Thanksgiving
        return False
    if d.month in (11, 12, 1, 2, 3) and d.weekday() in (0, 1):  # winter: closed Mon/Tue
        return False
    return True


def daily_sales():
    d = START
    while d <= END:
        if shop_is_open(d):
            base = MONTH_SALES[d.month] * (1.06 if d.year == 2025 else 1.0)
            if d.weekday() >= 5:
                base *= 1.35
            if (d.month, d.day) == (7, 4):
                base *= 1.8
            net = base * rng.lognormal(0, 0.25)
            tax = net * SALES_TAX_RATE
            fees = (net + tax) * CARD_SHARE * CARD_FEE_RATE
            n, t, f = cents(net), cents(tax), cents(fees)
            ts = datetime.combine(d + timedelta(days=1), time(2, 15, int(rng.integers(0, 60))))
            L.post(d, ts, "POS", "daily_sales", "pos_system",
                   f"POS daily sales {d.isoformat()}",
                   [(1000, n + t - f, 0), (6500, f, 0), (4000, 0, n), (2100, 0, t)])
            L.sales_tax_by_month[(d.year, d.month)] = L.sales_tax_by_month.get((d.year, d.month), 0) + t
        d += timedelta(days=1)


def payroll():
    d = date(2024, 1, 5)  # first Friday
    while d <= END:
        staff = MONTH_STAFF[d.month]
        # summer crews are mostly part-time students; the winter crew works longer shifts
        hours = 80 * rng.uniform(0.6, 0.75) if staff > 8 else 80 * rng.uniform(0.75, 0.95)
        gross = staff * hours * 17.50 * rng.uniform(0.97, 1.03)
        withheld = gross * 0.22
        employer_tax = gross * 0.1065
        g, w, e = cents(gross), cents(withheld), cents(employer_tax)
        L.post(d, datetime.combine(d, time(6, 0, int(rng.integers(0, 60)))), "PAYROLL",
               "payroll", "payroll_svc", f"Payroll run, period ending {d.isoformat()}",
               [(6000, g, 0), (6010, e, 0), (1000, 0, g - w), (2200, 0, w + e)],
               approved_by="dsmith")
        key = (d.year, d.month)
        L.payroll_liab_by_month[key] = L.payroll_liab_by_month.get(key, 0) + w + e
        d += timedelta(days=14)


def monthly_remittances():
    for (y, m), amount in sorted(L.sales_tax_by_month.items()):
        first_next = (pd.Timestamp(y, m, 1) + pd.offsets.MonthBegin(1)).date()
        pay = next_business_day(first_next + timedelta(days=14))
        if pay > END:
            continue
        L.post(pay, office_ts(pay), "BANK", "tax_remittance", "jlopez",
               f"Maine sales tax remittance {y}-{m:02d}",
               [(2100, amount, 0), (1000, 0, amount)],
               approved_by="dsmith" if amount >= cents(APPROVAL_LIMIT) else "")
    for (y, m), amount in sorted(L.payroll_liab_by_month.items()):
        pay = (pd.Timestamp(y, m, 1) + pd.offsets.MonthBegin(1)).date() + timedelta(days=9)
        pay = next_business_day(pay)
        if pay > END:
            continue
        L.post(pay, datetime.combine(pay, time(6, 30, 0)), "PAYROLL", "tax_remittance",
               "payroll_svc", f"Payroll tax and withholding remittance {y}-{m:02d}",
               [(2200, amount, 0), (1000, 0, amount)], approved_by="dsmith")


def vendors():
    # Dairy: weekly, scales with sales
    d = date(2024, 1, 3)
    while d <= END - timedelta(days=14):
        weekly_sales = MONTH_SALES[d.month] * 7 * (1.06 if d.year == 2025 else 1.0)
        L.ap_invoice_and_payment("Casco Bay Dairy Co.", d,
                                 round(weekly_sales * 0.21 * rng.lognormal(0, 0.12), 2))
        d += timedelta(days=7)
    # Cones and packaging: every two weeks
    d = date(2024, 1, 10)
    while d <= END - timedelta(days=30):
        biweekly_sales = MONTH_SALES[d.month] * 14
        L.ap_invoice_and_payment("Pine State Cone Supply", d,
                                 round(biweekly_sales * 0.045 * rng.lognormal(0, 0.15) + 60, 2))
        d += timedelta(days=14)
    for y in (2024, 2025):
        for m in range(1, 13):
            first = date(y, m, 1)
            L.ap_invoice_and_payment("Harborview Properties LLC", first, 4200.00,
                                     description="Monthly rent - 14 Commercial St")
            summer = m in (6, 7, 8)
            power = rng.normal(1650 if summer else 780, 90)
            L.ap_invoice_and_payment("Northern Light Power", date(y, m, 18), round(power, 2))
            L.ap_invoice_and_payment("Mainstay Mutual Insurance", date(y, m, 5), 612.40)
            L.ap_invoice_and_payment("ScoopPOS Software", date(y, m, 2), 189.00)
            if rng.random() < (0.9 if 4 <= m <= 9 else 0.35):
                L.ap_invoice_and_payment("Seacoast Refrigeration Service",
                                         date(y, m, int(rng.integers(3, 26))),
                                         round(rng.lognormal(np.log(650), 0.6), 2))
            if m in (3, 4, 5, 6) or rng.random() < 0.15:
                L.ap_invoice_and_payment("Gullwing Print & Signs",
                                         date(y, m, int(rng.integers(3, 26))),
                                         round(rng.lognormal(np.log(420), 0.55), 2))
    # One legitimate big purchase: a new dipping cabinet.
    L.ap_invoice_and_payment("Seacoast Refrigeration Service", date(2025, 3, 12), 14_800.00,
                             account=1500, description="New 12-flavor dipping cabinet")


def catering():
    for y in (2024, 2025):
        for m in (5, 6, 7, 8, 9):
            for _ in range(int(rng.integers(1, 4))):
                d = next_business_day(date(y, m, int(rng.integers(2, 26))))
                net = round(rng.uniform(600, 2500), 2)
                n, t = cents(net), cents(net * SALES_TAX_RATE)
                L.post(d, office_ts(d), "AR", "invoice", "jlopez",
                       "Catering - private event",
                       [(1200, n + t, 0), (4100, 0, n), (2100, 0, t)])
                L.sales_tax_by_month[(y, m)] = L.sales_tax_by_month.get((y, m), 0) + t
                paid = next_business_day(d + timedelta(days=int(rng.integers(12, 35))))
                if paid > END:
                    continue
                L.post(paid, office_ts(paid), "BANK", "receipt", "jlopez",
                       "Catering payment received",
                       [(1000, n + t, 0), (1200, 0, n + t)])


def petty_cash():
    for y in (2024, 2025):
        for m in range(1, 13):
            spent = 0
            for _ in range(int(rng.integers(2, 5))):
                d = date(y, m, int(rng.integers(1, 28)))
                minutes = int(rng.integers(11 * 60, 21 * 60 + 30))
                ts = datetime.combine(d, time(minutes // 60, minutes % 60))
                amt = cents(rng.uniform(12, 95))
                spent += amt
                L.post(d, ts, "GJ", "petty_cash", "tpark",
                       rng.choice(["Petty cash - cleaning supplies", "Petty cash - napkins and spoons",
                                   "Petty cash - ice", "Petty cash - batteries and tape"]),
                       [(6900, amt, 0), (1010, 0, amt)])
            d = next_business_day(month_end(date(y, m, 1)) - timedelta(days=2))
            L.post(d, office_ts(d), "BANK", "petty_replenish", "jlopez",
                   "Replenish petty cash", [(1010, spent, 0), (1000, 0, spent)])


def month_end_close():
    for y in (2024, 2025):
        for m in range(1, 13):
            me = month_end(date(y, m, 1))
            first_next = me + timedelta(days=1)
            close_ts = office_ts(first_next + timedelta(days=int(rng.integers(1, 6))), 9, 18)
            L.post(me, close_ts, "GJ", "depreciation", "mchen",
                   f"Monthly depreciation {y}-{m:02d}",
                   [(6800, cents(1150 if (y, m) < (2025, 3) else 1397.67), 0),
                    (1590, 0, cents(1150 if (y, m) < (2025, 3) else 1397.67))])
            # Utilities accrual (estimate for usage not yet billed), reversed next month.
            est = cents(rng.normal(1100 if m in (6, 7, 8) else 520, 60))
            L.post(me, close_ts + timedelta(minutes=7), "GJ", "accrual", "mchen",
                   f"Accrue unbilled utilities {y}-{m:02d}",
                   [(6200, est, 0), (2300, 0, est)])
            if first_next <= END:
                L.post(first_next, close_ts + timedelta(minutes=9), "GJ", "accrual_reversal",
                       "mchen", f"Reverse utilities accrual {y}-{m:02d}",
                       [(2300, est, 0), (6200, 0, est)])


def owner_draws():
    for d, amt in [(date(2024, 9, 16), 10_000), (date(2024, 12, 16), 5_000),
                   (date(2025, 9, 15), 12_000), (date(2025, 12, 15), 6_000)]:
        L.post(d, office_ts(d), "GJ", "owner_draw", "dsmith", "Owner draw",
               [(3100, cents(amt), 0), (1000, 0, cents(amt))])


# --------------------------------------------------------------------------- #
# Planted anomalies (the answer key)
# --------------------------------------------------------------------------- #
def plant_anomalies(seed: int = SEED):
    lines = pd.DataFrame(L.lines)

    # 1. Duplicate payments: the same invoice entered and paid twice, with the
    #    invoice number changed slightly so a simple exact-match check misses it.
    dairy = lines[(lines.doc_type == "invoice") & (lines.source == "AP")
                  & (lines.vendor.isin(["Casco Bay Dairy Co.", "Pine State Cone Supply"]))
                  & (lines.line_no == 1)]
    picks = dairy.sample(5, random_state=seed)
    for k, (_, row) in enumerate(picks.iterrows(), start=1):
        inv_no = row.invoice_no
        variant = [inv_no + "A", inv_no.replace("-", "-0"), inv_no + "-1",
                   inv_no.replace("-", ""), inv_no + " "][k - 1]
        d = row.posting_date + timedelta(days=int(rng.integers(2, 9)))
        je = L.ap_invoice_and_payment(row.vendor, d, row.debit, invoice_no=variant)
        L.label(f"DUP-{k}", "duplicate_payment", je,
                f"Invoice {inv_no} entered a second time as '{variant}' and paid again.")

    # 2. Split purchases: one repair job billed as two invoices just under the
    #    $5,000 approval limit so neither payment needs the owner's sign-off.
    for k, d in enumerate([date(2024, 7, 9), date(2024, 8, 20),
                           date(2025, 6, 17), date(2025, 7, 29)], start=1):
        ids = []
        base_no = L.new_invoice_no("Seacoast Refrigeration Service")
        n = int(base_no.split("-")[1])
        for part, amt in enumerate([rng.uniform(4_880, 4_995), rng.uniform(4_700, 4_960)]):
            ids += L.ap_invoice_and_payment(
                "Seacoast Refrigeration Service", d, round(amt, 2),
                description="Walk-in freezer compressor service",
                invoice_no=f"SRS-{n + part}")
        L.label(f"SPLIT-{k}", "split_below_threshold", ids,
                "One repair job billed as two invoices just below the $5,000 approval limit.")

    # 3. Ghost vendor: a new 'supplier' that appears in late summer, invoices
    #    in round-ish amounts just under the limit and is paid fast.
    ids = []
    for d in [date(2025, 8, 5), date(2025, 8, 19), date(2025, 9, 2),
              date(2025, 9, 23), date(2025, 10, 14), date(2025, 11, 4)]:
        amt = float(rng.integers(420, 499)) * 10
        ids += L.ap_invoice_and_payment(
            "Coastal Supply Partners LLC", d, amt, account=5100,
            description="Coastal Supply Partners LLC - packaging",
            pay_date=next_business_day(d + timedelta(days=3)))
    L.label("GHOST-1", "ghost_vendor", ids,
            "Vendor with no history, invoices just under the approval limit, paid within days.")

    # 4. Off-hours manual entries: cash taken out of the bank late on a Sunday night.
    for k, d in enumerate([date(2024, 5, 19), date(2024, 10, 27),
                           date(2025, 4, 13), date(2025, 11, 23)], start=1):
        ts = datetime.combine(d, time(23, int(rng.integers(5, 55))))
        amt = cents(rng.uniform(1_800, 3_200))
        je = L.post(d, ts, "GJ", "adjustment", "jlopez", "Misc adjustment",
                    [(6300, amt, 0), (1000, 0, amt)])
        L.label(f"OFFHRS-{k}", "off_hours_manual", [je],
                "Manual cash entry posted by the bookkeeper close to midnight on a Sunday.")

    # 5. Round-dollar reclasses with no support, self-approved.
    for k, (d, amt, acct) in enumerate([(date(2024, 3, 28), 7_500, 6400),
                                         (date(2024, 11, 26), 10_000, 6300),
                                         (date(2025, 2, 25), 6_000, 6900),
                                         (date(2025, 10, 28), 8_500, 6400)], start=1):
        je = L.post(d, office_ts(d, 9, 18), "GJ", "adjustment", "mchen",
                    "Reclass per discussion",
                    [(acct, cents(amt), 0), (1000, 0, cents(amt))],
                    approved_by="mchen")
        L.label(f"ROUND-{k}", "round_dollar_reclass", [je],
                "Large round-dollar cash entry, vague description, approved by its own author.")

    # 6. Window dressing: fake catering revenue booked on the last day of a
    #    quarter, reversed quietly a few days into the next one.
    for k, (d, amt) in enumerate([(date(2024, 6, 30), 18_000), (date(2024, 12, 31), 15_000),
                                   (date(2025, 9, 30), 22_500)], start=1):
        book = L.post(d, office_ts(d + timedelta(days=2), 9, 18), "GJ", "adjustment", "mchen",
                      "Catering contract - event series",
                      [(1200, cents(amt), 0), (4100, 0, cents(amt))], approved_by="dsmith")
        rev_d = next_business_day(d + timedelta(days=4))
        rev = L.post(rev_d, office_ts(rev_d, 9, 18), "GJ", "adjustment", "mchen",
                     "Correct catering billing",
                     [(4100, cents(amt), 0), (1200, 0, cents(amt))], approved_by="dsmith")
        L.label(f"REVCUT-{k}", "period_end_revenue", [book, rev],
                "Revenue booked on the last day of a quarter and reversed days later.")

    # 7. Unusual user: the shift manager, who only ever touches petty cash,
    #    books 'refunds' straight out of the operating account.
    for k, d in enumerate([date(2024, 8, 3), date(2025, 7, 12), date(2025, 8, 9)], start=1):
        ts = datetime.combine(d, time(19, int(rng.integers(0, 59))))
        amt = cents(rng.uniform(850, 1_600))
        je = L.post(d, ts, "GJ", "adjustment", "tpark", "Customer refund adjustment",
                    [(4000, amt, 0), (1000, 0, amt)])
        L.label(f"USER-{k}", "unusual_user_account", [je],
                "Shift manager with petty-cash-only duties reverses sales against the bank account.")

    # 8. Post-close entries: booked into a month weeks after it was closed.
    for k, (period_end, lag) in enumerate([(date(2024, 4, 30), 33), (date(2024, 9, 30), 41),
                                            (date(2025, 5, 31), 27), (date(2025, 8, 31), 48)],
                                           start=1):
        amt = cents(rng.uniform(2_400, 4_400))
        je = L.post(period_end, office_ts(period_end + timedelta(days=lag), 9, 18), "GJ",
                    "adjustment", "mchen", "Accrual true-up",
                    [(5000, amt, 0), (2300, 0, amt)])
        L.label(f"LATE-{k}", "post_close_entry", [je],
                f"Entry dated into a closed month, created {lag} days after month-end.")

    # 9. Personal expense coded as repairs: the owner's boat slip.
    ids = []
    for d in [date(2025, 6, 4), date(2025, 7, 7), date(2025, 8, 5)]:
        ids += L.ap_invoice_and_payment("Back Cove Marina", d, 1_850.00, account=6300,
                                        description="Back Cove Marina - dock services")
    L.label("PERS-1", "personal_expense", ids,
            "A marina slip fee booked as shop repairs: a personal cost on the business.")


def generate(seed: int = SEED):
    """Build the whole ledger for one random seed. Returns
    (chart_of_accounts, users, journal_entries, anomaly_labels) as DataFrames."""
    global rng, L
    rng = np.random.default_rng(seed)
    L = Ledger()
    opening_balances()
    daily_sales()
    payroll()
    catering()
    vendors()
    petty_cash()
    month_end_close()
    owner_draws()
    monthly_remittances()
    plant_anomalies(seed)

    je = pd.DataFrame(L.lines)
    je["posting_date"] = pd.to_datetime(je["posting_date"])
    je["entry_ts"] = pd.to_datetime(je["entry_ts"])
    je = je.sort_values(["posting_date", "entry_ts", "je_id", "line_no"]).reset_index(drop=True)

    # sanity: every entry balances, and so does the whole ledger
    sums = je.groupby("je_id")[["debit", "credit"]].sum().round(2)
    assert (sums.debit == sums.credit).all(), "an entry does not balance"
    assert round(je.debit.sum() - je.credit.sum(), 2) == 0

    coa = pd.DataFrame([{"account": k, "account_name": v[0], "account_type": v[1]}
                        for k, v in ACCOUNTS.items()])
    users = pd.DataFrame(USERS, columns=["user_id", "name", "role", "is_system"])
    return coa, users, je, pd.DataFrame(L.labels)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    coa, users, je, labels = generate(SEED)
    out = je.assign(posting_date=je.posting_date.dt.date)
    coa.to_csv(DATA_DIR / "chart_of_accounts.csv", index=False)
    users.to_csv(DATA_DIR / "users.csv", index=False)
    out.to_csv(DATA_DIR / "journal_entries.csv", index=False)
    labels.to_csv(DATA_DIR / "anomaly_labels.csv", index=False)

    n_je = je.je_id.nunique()
    print(f"{n_je:,} journal entries, {len(je):,} lines, "
          f"{labels.case_id.nunique()} planted cases covering {labels.je_id.nunique()} entries "
          f"({labels.je_id.nunique() / n_je:.1%}).")


if __name__ == "__main__":
    main()
