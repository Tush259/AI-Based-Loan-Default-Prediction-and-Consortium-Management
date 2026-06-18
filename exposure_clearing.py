import pandas as pd
import numpy as np
def load_banks(filepath="bank_dataset.csv"):
    return pd.read_csv(filepath)


def allocate_consortium(loan_amount, filepath="bank_dataset.csv"):

    banks_df = load_banks(filepath)
    eligible = []

    for _, bank in banks_df.iterrows():
        projected = bank['Current_Exposure'] + bank['Loan_Share']
        if projected <= bank['Max_Exposure_Limit']:
            eligible.append(bank)

    if not eligible:
        return pd.DataFrame(), "❌ No eligible banks — all exposure limits exceeded!"

    n              = len(eligible)
    share_per_bank = round(loan_amount / n, 2)

    rows = []
    for bank in eligible:
        rows.append({
            "Bank":                 bank['Bank_ID'],
            "Loan Share (₹)":       share_per_bank,
            "Liquidity Ratio":      bank['Liquidity_Ratio'],
            "Capital Adequacy":     bank['Capital_Adequacy'],
            "Current Exposure":     bank['Current_Exposure'],
            "Max Exposure Limit":   bank['Max_Exposure_Limit'],
            "Past Default Rate":    bank['Past_Default_Rate'],
            "Operational Capacity": bank['Operational_Capacity'],
            "Priority Label":       int(bank['Priority_Label']),
        })

    allocation_df = pd.DataFrame(rows)
    message       = f"✅ ₹{loan_amount:,.0f} distributed equally among {n} eligible banks"
    return allocation_df, message


def liquidity_priority(allocation_df):

    if allocation_df.empty:
        return allocation_df

    sorted_df = allocation_df.sort_values(
        "Liquidity Ratio", ascending=True
    ).reset_index(drop=True)

    sorted_df["Repayment Priority Rank"] = sorted_df.index + 1
    return sorted_df


def simulate_clearing(allocation_df, repayment_amount):

    if allocation_df.empty:
        return pd.DataFrame()

    priority_df = liquidity_priority(allocation_df)
    remaining   = float(repayment_amount)
    log         = []

    for _, bank in priority_df.iterrows():
        share = float(bank["Loan Share (₹)"])

        if remaining <= 0:
            log.append({
                "Bank":                     bank["Bank"],
                "Repayment Priority Rank":  int(bank["Repayment Priority Rank"]),
                "Loan Share (₹)":           share,
                "Amount Received (₹)":      0.0,
                "Remaining Pool After (₹)": 0.0,
                "Status":                   "⏳ Pending"
            })
        elif remaining >= share:
            remaining -= share
            log.append({
                "Bank":                     bank["Bank"],
                "Repayment Priority Rank":  int(bank["Repayment Priority Rank"]),
                "Loan Share (₹)":           share,
                "Amount Received (₹)":      share,
                "Remaining Pool After (₹)": round(remaining, 2),
                "Status":                   "✅ Fully Cleared"
            })
        else:
            log.append({
                "Bank":                     bank["Bank"],
                "Repayment Priority Rank":  int(bank["Repayment Priority Rank"]),
                "Loan Share (₹)":           share,
                "Amount Received (₹)":      round(remaining, 2),
                "Remaining Pool After (₹)": 0.0,
                "Status":                   "⚠️ Partially Cleared"
            })
            remaining = 0.0

    return pd.DataFrame(log)


def get_bank_summary(filepath="bank_dataset.csv"):
    return pd.read_csv(filepath)


def get_bank_eligibility_table(filepath="bank_dataset.csv"):
    df = pd.read_csv(filepath)
    df["Projected Exposure"] = df["Current_Exposure"] + df["Loan_Share"]
    df["Available Headroom"] = df["Max_Exposure_Limit"] - df["Current_Exposure"]
    df["Eligible"] = df.apply(
        lambda r: "✅ Yes" if r["Projected Exposure"] <= r["Max_Exposure_Limit"] else "❌ No",
        axis=1
    )
    return df[[
        "Bank_ID", "Current_Exposure", "Loan_Share",
        "Projected Exposure", "Max_Exposure_Limit",
        "Available Headroom", "Eligible"
    ]]