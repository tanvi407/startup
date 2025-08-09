# indian_startup_funding_analysis.py
# Full script: load -> clean -> analyze -> visualize -> recommend

import os
import re
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser

# Optional: nicer missing value visualization
try:
    import missingno as msno
    MISSINGNO_AVAILABLE = True
except Exception:
    MISSINGNO_AVAILABLE = False

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# --------------------------
# Config
# --------------------------
DATA_PATH = "startup_funding.csv"   # change if your file name differs
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Exchange rate used to convert USD -> INR (adjust as needed)
USD_TO_INR = 82.0

# Some helper constants
CRORE_TO_INR = 1e7   # 1 crore = 10,000,000 INR
LAKH_TO_INR = 1e5    # 1 lakh = 100,000 INR
MILLION_TO_INR = 1e6

# --------------------------
# Utility functions
# --------------------------
def safe_parse_date(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x)
    try:
        # dateutil parser handles many formats
        return pd.to_datetime(parser.parse(str(x), dayfirst=True, fuzzy=True))
    except Exception:
        # try common numeric formats
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return pd.to_datetime(x, format=fmt)
            except Exception:
                continue
    return pd.NaT

def parse_amount_to_inr(amount_str, usd_to_inr=USD_TO_INR):
    """
    Convert messy funding amount string to numeric INR value.
    Heuristics for:
     - 'Undisclosed', 'Unknown', NaN -> np.nan
     - Contains 'USD' or '$' -> convert using usd_to_inr
     - 'cr', 'crore' -> * CRORE_TO_INR
     - 'lakh', 'lac' -> * LAKH_TO_INR
     - 'mn', 'million' -> * MILLION_TO_INR
     - plain numbers with commas
    Returns float INR (or np.nan).
    """
    if pd.isna(amount_str):
        return np.nan
    s = str(amount_str).strip().lower()
    if s in ("", "nan", "undisclosed", "unknown", "na", "0"):
        return np.nan

    # remove parentheses
    s = re.sub(r"[()]", "", s)

    # Handle ranges like '1-2' => take average
    range_match = re.search(r"(\d+(?:[\.,]\d+)?)\s*[-–—]\s*(\d+(?:[\.,]\d+)?)", s)
    if range_match:
        a = float(range_match.group(1).replace(",", "").replace(" ", "").replace("—", "."))
        b = float(range_match.group(2).replace(",", "").replace(" ", "").replace("—", "."))
        s = str((a + b) / 2.0)

    # find numeric part
    num_match = re.search(r"([0-9]+(?:[\,\.][0-9]+)*)", s)
    if not num_match:
        # nothing numeric
        return np.nan
    num = num_match.group(1).replace(",", "")
    try:
        v = float(num)
    except Exception:
        return np.nan

    # detect unit
    if "usd" in s or "$" in s or "us$" in s:
        # assume v is in USD millions if 'mn' or plain USD -> heuristics:
        if "mn" in s or "m" in s or "million" in s:
            return v * MILLION_TO_INR * usd_to_inr
        else:
            # assume USD thousands or units - treat as USD -> convert directly
            return v * usd_to_inr

    if "cr" in s or "crore" in s:
        return v * CRORE_TO_INR

    if "lakh" in s or "lac" in s:
        return v * LAKH_TO_INR

    if "mn" in s or "m" in s or "million" in s:
        return v * MILLION_TO_INR

    # if value looks too small (<100) and 'k' present -> thousands
    if "k" in s:
        return v * 1e3

    # fallback: if number > 1e4 treat as INR already (e.g. 1000000)
    if v > 1e4:
        return v

    # otherwise assume millions
    return v * MILLION_TO_INR

def clean_text_field(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    # unify separators
    s = re.sub(r"\s+", " ", s)
    # common replacements
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return s

# --------------------------
# Load data
# --------------------------
def load_data(path=DATA_PATH):
    # Try with default encodings and a few variants
    read_attempts = [
        {"encoding": "utf-8"},
        {"encoding": "latin-1"},
    ]
    for params in read_attempts:
        try:
            df = pd.read_csv(path, **params, low_memory=False)
            print(f"Loaded data with shape {df.shape} using {params}")
            return df
        except Exception as e:
            # try next
            last_exc = e
    raise RuntimeError(f"Failed to read CSV: {last_exc}")

df = load_data(DATA_PATH)

# Look at columns and try to guess relevant ones
print("Columns found:", list(df.columns)[:40])

# Common Kaggle variations use columns like: 'date', 'City', 'InvestmentType', 'Amount', 'StartupName', 'Investors', 'Sector'
# Normalize column names to lowercased no-space forms
df.columns = [c.strip() for c in df.columns]
col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
df.rename(columns=col_map, inplace=True)

# Provide alias guesses
aliases = {
    'date': None,
    'city': None,
    'investment_type': None,
    'amount': None,
    'startup_name': None,
    'investors': None,
    'sector': None,
    'round': None,
    'amount_in_usd': None,
}
for c in df.columns:
    lc = c.lower()
    if 'date' in lc and aliases['date'] is None:
        aliases['date'] = c
    if ('city' in lc or 'location' in lc) and aliases['city'] is None:
        aliases['city'] = c
    if ('investment' in lc and 'type' in lc) and aliases['investment_type'] is None:
        aliases['investment_type'] = c
    if ('amount' in lc or 'raised' in lc) and aliases['amount'] is None:
        aliases['amount'] = c
    if ('startup' in lc or 'company' in lc or 'name' in lc) and aliases['startup_name'] is None:
        aliases['startup_name'] = c
    if ('investor' in lc or 'investors' in lc) and aliases['investors'] is None:
        aliases['investors'] = c
    if ('sector' in lc or 'vertical' in lc) and aliases['sector'] is None:
        aliases['sector'] = c
    if ('round' in lc or 'investment_type' in lc or 'stage' in lc) and aliases['round'] is None:
        aliases['round'] = c

print("Auto-detected columns mapping (may be None):")
for k, v in aliases.items():
    print(f"  {k}: {v}")

# If key columns are missing, try to infer common names
# Ensure we have at least amount & date & startup_name
required = ['date', 'amount', 'startup_name']
for r in required:
    if aliases.get(r) is None:
        # try any reasonable fallback
        for cand in ['funded_on', 'funding_round', 'announced_on', 'name', 'company', 'amount_usd']:
            if cand in df.columns:
                aliases[r] = cand
                break

# Rename columns to canonical short names in dataframe
canonical = {}
for key, col in aliases.items():
    if col is not None:
        canonical[col] = key
df = df.rename(columns=canonical)

# Make copies of raw columns
if 'amount' in df.columns:
    df['raw_amount'] = df['amount'].astype(str)
else:
    df['raw_amount'] = np.nan

if 'date' in df.columns:
    df['raw_date'] = df['date']
else:
    df['raw_date'] = np.nan

# --------------------------
# Clean / Preprocess
# --------------------------
# Clean text columns
text_cols = [c for c in df.columns if df[c].dtype == object]
for c in text_cols:
    df[c] = df[c].apply(clean_text_field)

# Parse dates
if 'date' in df.columns:
    df['date_parsed'] = df['date'].apply(safe_parse_date)
else:
    df['date_parsed'] = pd.NaT

# Fill missing dates from other columns heuristics
# (If a column 'announced_on' exists use it, already handled earlier via alias mapping)

# Extract year, month
df['year'] = df['date_parsed'].dt.year
df['month'] = df['date_parsed'].dt.month
df['year_month'] = df['date_parsed'].dt.to_period('M')

# Parse amounts -> INR numeric
df['amount_in_inr'] = df['raw_amount'].apply(parse_amount_to_inr)

# Some Kaggle datasets include a column 'amount_in_usd' or 'amount_usd' - if present, use that to fill NaNs
for c in df.columns:
    if 'amount' in c and 'usd' in c:
        try:
            df['amount_in_inr'] = df['amount_in_inr'].fillna(df[c].astype(float) * USD_TO_INR)
        except Exception:
            # if values are strings, try parsing numbers
            df['amount_in_inr'] = df['amount_in_inr'].fillna(df[c].apply(lambda x: float(str(x).replace(",", "")) if pd.notna(x) and str(x).strip() != "" else np.nan) * USD_TO_INR)

# Drop rows with no date and no amount
df_clean = df.copy()
print("Before dropping rows:", df_clean.shape)
df_clean = df_clean[~(df_clean['date_parsed'].isna() & df_clean['amount_in_inr'].isna())]
print("After dropping rows without date & amount:", df_clean.shape)

# Normalize city names (lowercase and strip)
if 'city' in df_clean.columns:
    df_clean['city'] = df_clean['city'].str.title().str.replace(r"\,.*", "", regex=True).str.strip()

# Normalize sector / vertical
if 'sector' in df_clean.columns:
    df_clean['sector'] = df_clean['sector'].str.title().str.replace(r"[^A-Za-z0-9 &/,-]", "", regex=True)

# Normalize investor list (split on commas / semicolons and keep trimmed)
def split_investors(inv):
    if pd.isna(inv):
        return []
    s = re.split(r",|;| and |/| & |\\|", str(inv))
    s = [x.strip() for x in s if x and x.strip().lower() not in ("undisclosed", "unknown")]
    return s

if 'investors' in df_clean.columns:
    df_clean['investors_list'] = df_clean['investors'].apply(split_investors)
else:
    df_clean['investors_list'] = [[] for _ in range(len(df_clean))]

# --------------------------
# Analysis
# --------------------------

# 1) Funding trends over time (yearly + monthly)
funding_by_year = df_clean.groupby('year', dropna=True)['amount_in_inr'].sum().sort_index()
funding_by_month = df_clean.groupby('year_month')['amount_in_inr'].sum().sort_index()

# 2) Top sectors
top_sectors = None
if 'sector' in df_clean.columns:
    top_sectors = (
        df_clean.dropna(subset=['sector'])
        .groupby('sector')['amount_in_inr']
        .agg(['sum', 'count'])
        .sort_values('sum', ascending=False)
    )

# 3) Top cities
top_cities = None
if 'city' in df_clean.columns:
    top_cities = (
        df_clean.dropna(subset=['city'])
        .groupby('city')['amount_in_inr']
        .agg(['sum', 'count'])
        .sort_values('sum', ascending=False)
    )

# 4) Top startups by funding
top_startups = None
if 'startup_name' in df_clean.columns:
    top_startups = (
        df_clean.dropna(subset=['startup_name'])
        .groupby('startup_name')['amount_in_inr']
        .agg(['sum', 'count'])
        .sort_values('sum', ascending=False)
    )

# 5) Active investors (by number of deals & total invested)
investor_expanded = []
for idx, row in df_clean.iterrows():
    for inv in row['investors_list']:
        investor_expanded.append({
            'investor': inv,
            'amount_in_inr': row['amount_in_inr'],
            'date_parsed': row['date_parsed'],
            'startup_name': row.get('startup_name', None)
        })
investor_df = pd.DataFrame(investor_expanded)
if not investor_df.empty:
    top_investors_by_deals = investor_df.groupby('investor').size().sort_values(ascending=False).head(30)
    top_investors_by_amount = investor_df.groupby('investor')['amount_in_inr'].sum().sort_values(ascending=False).head(30)
else:
    top_investors_by_deals = pd.Series(dtype=int)
    top_investors_by_amount = pd.Series(dtype=float)

# 6) Investment type distributions
if 'investment_type' in df_clean.columns:
    invtype_dist = df_clean['investment_type'].value_counts(dropna=True)
else:
    invtype_dist = pd.Series(dtype=int)

# --------------------------
# Visualizations (saved to PLOT_DIR)
# --------------------------
def save_fig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("Saved:", path)

# 1. Funding by Year (bar)
if not funding_by_year.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    funding_by_year_div = funding_by_year / 1e9  # convert to billion INR for readability
    sns.barplot(x=funding_by_year_div.index.astype(int), y=funding_by_year_div.values, ax=ax)
    ax.set_title("Total Funding by Year (Billion INR)")
    ax.set_ylabel("Funding (Billion INR)")
    ax.set_xlabel("Year")
    save_fig(fig, "funding_by_year.png")
    plt.close(fig)

# 2. Funding by Month (time series)
if not funding_by_month.empty:
    fig, ax = plt.subplots(figsize=(12, 5))
    funding_by_month_float = funding_by_month.astype(float) / 1e9
    funding_by_month_float.plot(ax=ax, marker='o')
    ax.set_title("Funding over Time (Monthly) - Billion INR")
    ax.set_ylabel("Funding (Billion INR)")
    ax.set_xlabel("Year-Month")
    save_fig(fig, "funding_monthly_ts.png")
    plt.close(fig)

# 3. Top sectors (top 15)
if top_sectors is not None and not top_sectors.empty:
    top15 = top_sectors.head(15).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(y=top15.index, x=top15['sum'] / 1e9, ax=ax)
    ax.set_xlabel("Total Funding (Billion INR)")
    ax.set_ylabel("Sector")
    ax.set_title("Top 15 Sectors by Funding")
    save_fig(fig, "top_sectors.png")
    plt.close(fig)

# 4. Top cities
if top_cities is not None and not top_cities.empty:
    top10 = top_cities.head(10).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top10['sum'].values / 1e9, y=top10.index, ax=ax)
    ax.set_xlabel("Total Funding (Billion INR)")
    ax.set_title("Top 10 Cities by Funding")
    save_fig(fig, "top_cities.png")
    plt.close(fig)

# 5. Top startups
if top_startups is not None and not top_startups.empty:
    ts10 = top_startups.head(10).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=ts10['sum'].values / 1e9, y=ts10.index, ax=ax)
    ax.set_xlabel("Total Funding (Billion INR)")
    ax.set_title("Top 10 Funded Startups")
    save_fig(fig, "top_startups.png")
    plt.close(fig)

# 6. Top investors by amount
if not top_investors_by_amount.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    iv = top_investors_by_amount.head(15) / 1e9
    sns.barplot(x=iv.values, y=iv.index, ax=ax)
    ax.set_xlabel("Total Invested (Billion INR)")
    ax.set_title("Top 15 Investors by Invested Amount")
    save_fig(fig, "top_investors_amount.png")
    plt.close(fig)

# 7. Investment type distribution (pie or bar)
if not invtype_dist.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    invtype_dist.plot(kind='bar', ax=ax)
    ax.set_title("Investment Type Distribution (Counts)")
    ax.set_ylabel("Number of Deals")
    save_fig(fig, "investment_type_distribution.png")
    plt.close(fig)

# 8. Missing value visualization (optional)
if MISSINGNO_AVAILABLE:
    fig = msno.matrix(df_clean, figsize=(12, 4))
    plt.savefig(os.path.join(PLOT_DIR, "missing_matrix.png"))
    plt.close()

# --------------------------
# Print summarised tables (top rows)
# --------------------------
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("\n=== Funding by Year (INR Sum) ===")
print(funding_by_year.fillna(0).apply(lambda x: f"{x:,.0f}").to_string())

if top_sectors is not None and not top_sectors.empty:
    print("\n=== Top Sectors (by funding) ===")
    print(top_sectors.head(10).to_string())

if top_cities is not None and not top_cities.empty:
    print("\n=== Top Cities (by funding) ===")
    print(top_cities.head(10).to_string())

if top_startups is not None and not top_startups.empty:
    print("\n=== Top Startups (by funding) ===")
    print(top_startups.head(10).to_string())

if not investor_df.empty:
    print("\n=== Top Investors by Number of Deals ===")
    print(top_investors_by_deals.head(10).to_string())
    print("\n=== Top Investors by Total Amount (INR) ===")
    print(top_investors_by_amount.head(10).apply(lambda x: f"{x:,.0f}").to_string())

# --------------------------
# Automated recommendations (template + computed cues)
# --------------------------
def generate_recommendations(df_clean, funding_by_year, top_sectors, top_cities, top_investors_by_deals):
    recs = []
    # Basic signals
    recent_year = int(funding_by_year.dropna().index.max()) if not funding_by_year.dropna().empty else None
    if recent_year:
        recs.append(f"Observed data up to year {recent_year}. Investors should review macro trend for {recent_year} vs previous years to identify momentum shifts.")
    # Sector advice
    if top_sectors is not None and not top_sectors.empty:
        top_sector = top_sectors.index[0]
        recs.append(f"Sector Focus: {top_sector} is the top-funded sector — consider early-stage investments in adjacent verticals (e.g., B2B tools, fintech infra) where valuations may be more attractive.")
    # City advice
    if top_cities is not None and not top_cities.empty:
        top_city = top_cities.index[0]
        recs.append(f"Geographic focus: {top_city} remains a funding hub. Founders outside major hub cities should evaluate remote-first traction strategies to tap investor pools.")
    # Investor advice
    if not top_investors_by_deals.empty:
        active_inv = top_investors_by_deals.head(3).index.tolist()
        recs.append(f"Active investors: {', '.join(active_inv)} are among the most active. Founders should target investor syndicates including these names for better chances in follow-on rounds.")
    # Amount distribution cue
    amt_median = df_clean['amount_in_inr'].dropna().median()
    if not np.isnan(amt_median):
        if amt_median < 1e7:  # less than 1 crore
            recs.append("Deal size insight: Median deal size is relatively small (< 1 Crore INR). Seed-stage investors can find many early-stage opportunities, while growth-stage investors should hunt for outliers.")
        else:
            recs.append("Deal size insight: Median deal sizes are sizable; consider selecting for scalability and unit economics early.")
    # Exit and timing
    recs.append("General: Investors should diversify across stages (seed + Series A) and track sector-specific KPIs (GMV for marketplaces, ARR for SaaS, CAC/LTV for consumer-facing startups).")
    recs.append("Founders: Focus on clear unit economics, define measurable traction metrics, and prepare to showcase path-to-profitability as valuations tighten.")
    return recs

recommendations = generate_recommendations(df_clean, funding_by_year, top_sectors, top_cities, top_investors_by_deals)
print("\n=== Actionable Recommendations ===")
for i, r in enumerate(recommendations, 1):
    print(f"{i}. {r}")

# Save a short text recommendations file
with open(os.path.join(PLOT_DIR, "recommendations.txt"), "w", encoding="utf-8") as f:
    f.write("Indian Startup Funding Analysis - Recommendations\n\n")
    for i, r in enumerate(recommendations, 1):
        f.write(f"{i}. {r}\n")

print(f"\nAll plots + recommendations saved in '{PLOT_DIR}/'. Script complete.")
