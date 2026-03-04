"""
Build dashboard data for static GitHub Pages deployment.
Run from repo root: python scripts/build_data.py [--out-dir data]
Outputs: data/snapshot.json, data/events.json, data/meta.json, data/charts/*.png
"""
from __future__ import print_function
import argparse
import json
import os
import re
import time

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from io import BytesIO
from scipy.stats import rankdata

try:
    import investpy
except ImportError:
    investpy = None

KEY_EVENTS = [
    "Fed", "Federal Reserve", "Interest Rate", "FOMC",
    "ISM Manufacturing", "ISM Non-Manufacturing", "ISM Services", "ISM",
    "CPI", "Consumer Price Index", "Nonfarm Payrolls", "NFP", "Employment",
    "PPI", "Producer Price Index", "PCE", "Core PCE", "Personal Consumption",
    "Retail Sales", "GDP", "Gross Domestic Product", "Unemployment", "Jobless Claims", "Initial Claims",
    "Housing Starts", "Building Permits", "Durable Goods", "Factory Orders",
    "Consumer Confidence", "Michigan Consumer", "Trade Balance", "Trade Deficit",
    "Beige Book", "Fed Minutes", "JOLTS", "Job Openings"
]

# Only Indices + Industries
STOCK_GROUPS = {
    "Indices": ["QQQE", "MGK", "QQQ", "IBIT", "RSP", "MDY", "IWM", "TLT", "SPY", "ETHA", "DIA"],
    "Industries": [
        "TAN", "KCE", "IBUY", "QQQE", "JETS", "IBB", "SMH", "CIBR", "UTES", "ROBO", "IGV", "WCLD", "ITA", "PAVE", "BLOK", "AIQ", "IYZ", "PEJ", "FDN", "KBE",
        "UNG", "BOAT", "KWEB", "KRE", "IBIT", "XRT", "IHI", "DRIV", "MSOS", "SOCL", "XLU", "ARKF", "SLX", "ARKK", "XTN", "XME", "KIE", "GLD", "GXC", "SCHH",
        "GDX", "IPAY", "IWM", "XOP", "VNQ", "EATZ", "FXI", "DBA", "ICLN", "SILJ", "REZ", "LIT", "SLV", "XHB", "XHE", "PBJ", "USO", "DBC", "FCG", "XBI",
        "ARKG", "CPER", "XES", "OIH", "PPH", "FNGS", "URA", "WGMI", "REMX"
    ],
}

LEVERAGED_ETFS = {
    "QQQ": {"long": ["TQQQ"], "short": ["SQQQ"]},
    "MDY": {"long": ["MIDU"], "short": []},
    "IWM": {"long": ["TNA"], "short": ["TZA"]},
    "TLT": {"long": ["TMF"], "short": ["TMV"]},
    "SPY": {"long": ["SPXL", "UPRO"], "short": ["SPXS", "SH"]},
    "ETHA": {"long": ["ETHU"], "short": []},
    "XLK": {"long": ["TECL"], "short": ["TECS"]},
    "XLI": {"long": ["DUSL"], "short": []},
    "XLC": {"long": ["LTL"], "short": []},
    "XLF": {"long": ["FAS"], "short": ["FAZ"]},
    "XLU": {"long": ["UTSL"], "short": []},
    "XLY": {"long": ["WANT"], "short": ["SCC"]},
    "XLRE": {"long": ["DRN"], "short": ["DRV"]},
    "XLP": {"long": ["UGE"], "short": ["SZK"]},
    "XLB": {"long": ["UYM"], "short": ["SMN"]},
    "XLE": {"long": ["ERX"], "short": ["ERY"]},
    "XLV": {"long": ["CURE"], "short": []},
    "SMH": {"long": ["SOXL"], "short": ["SOXS"]},
    "ARKK": {"long": ["TARK"], "short": ["SARK"]},
    "XTN": {"long": ["TPOR"], "short": []},
    "KWEB": {"long": ["CWEB"], "short": []},
    "XRT": {"long": ["RETL"], "short": []},
    "KRE": {"long": ["DPST"], "short": []},
    "DRIV": {"long": ["EVAV"], "short": []},
    "XBI": {"long": ["LABU"], "short": ["LABD"]},
    "ROBO": {"long": ["UBOT"], "short": []},
    "XHB": {"long": ["NAIL"], "short": []},
    "FNGS": {"long": ["FNGB"], "short": ["FNGD"]},
    "WCLD": {"long": ["CLDL"], "short": []},
    "XOP": {"long": ["GUSH"], "short": ["DRIP"]},
    "FDN": {"long": ["WEBL"], "short": ["WEBS"]},
    "FXI": {"long": ["YINN"], "short": ["YANG"]},
    "PEJ": {"long": ["OOTO"], "short": []},
    "USO": {"long": ["UCO"], "short": ["SCO"]},
    "PPH": {"long": ["PILL"], "short": []},
    "ITA": {"long": ["DFEN"], "short": []},
    "SLV": {"long": ["AGQ"], "short": ["ZSL"]},
    "GLD": {"long": ["UGL"], "short": ["GLL"]},
    "UNG": {"long": ["BOIL"], "short": ["KOLD"]},
    "GDX": {"long": ["NUGT", "GDXU"], "short": ["JDST", "GDXD"]},
    "IBIT": {"long": ["BITX", "BITU"], "short": ["SBIT", "BITI"]},
    "MSOS": {"long": ["MSOX"], "short": []},
    "REMX": {"long": [], "short": []},
    "EWY": {"long": ["KORU"], "short": []},
    "IEV": {"long": ["EURL"], "short": []},
    "EWJ": {"long": ["EZJ"], "short": []},
    "EWW": {"long": ["MEXX"], "short": []},
    "ASHR": {"long": ["CHAU"], "short": []},
    "INDA": {"long": ["INDL"], "short": []},
    "EEM": {"long": ["EDC"], "short": ["EDZ"]},
    "EWZ": {"long": ["BRZU"], "short": []}
}

SECTOR_COLORS = {
    "Information Technology": "#3f51b5", "Industrials": "#333", "Emerging Markets": "#00bcd4",
    "Consumer Discretionary": "#4caf50", "Health Care": "#e91e63", "Financials": "#ff5722",
    "Energy": "#795548", "Communication Services": "#9c27b0", "Real Estate": "#673ab7",
    "Commodities": "#8b6914", "Materials": "#ff9800", "Utilities": "#009688",
    "Consumer Staples": "#8bc34a", "Broad Market": "#9e9e9e",
}

Industries_COLORS = {
    "SMH": "#3f51b5", "ARKK": "#3f51b5", "XTN": "#333", "KWEB": "#00bcd4", "XRT": "#4caf50", "KRE": "#ff5722",
    "ARKF": "#3f51b5", "ARKG": "#e91e63", "BOAT": "#333", "DRIV": "#4caf50", "KBE": "#ff5722", "XES": "#795548",
    "XBI": "#e91e63", "OIH": "#795548", "SOCL": "#9c27b0", "ROBO": "#333", "AIQ": "#3f51b5", "XHB": "#4caf50",
    "FNGS": "#9e9e9e", "BLOK": "#3f51b5", "LIT": "#ff9800", "WCLD": "#3f51b5", "XOP": "#795548", "FDN": "#4caf50",
    "TAN": "#795548", "IBB": "#e91e63", "PAVE": "#333", "PEJ": "#4caf50", "KCE": "#ff5722", "XHE": "#e91e63",
    "IBUY": "#4caf50", "MSOS": "#4caf50", "FCG": "#795548", "JETS": "#4caf50", "IPAY": "#ff5722", "SLX": "#ff9800",
    "IGV": "#3f51b5", "CIBR": "#3f51b5", "EATZ": "#4caf50", "PPH": "#e91e63", "IHI": "#e91e63", "UTES": "#009688",
    "ICLN": "#795548", "XME": "#ff9800", "IYZ": "#9c27b0", "URA": "#795548", "ITA": "#333", "VNQ": "#673ab7",
    "SCHH": "#673ab7", "KIE": "#ff5722", "REZ": "#673ab7", "CPER": "#8b6914", "PBJ": "#8bc34a", "SLV": "#8b6914",
    "GLD": "#8b6914", "SILJ": "#ff9800", "GDX": "#ff9800", "FXI": "#00bcd4", "GXC": "#00bcd4", "USO": "#8b6914",
    "DBA": "#8b6914", "UNG": "#8b6914", "DBC": "#8b6914", "WGMI": "#3f51b5", "REMX": "#ff9800",
}


def get_ticker_to_sector_mapping():
    color_to_sector = {c: s for s, c in SECTOR_COLORS.items()}
    return {t: color_to_sector.get(c, "Broad Market") for t, c in Industries_COLORS.items()}


TICKER_TO_SECTOR = get_ticker_to_sector_mapping()


def get_leveraged_etfs(ticker):
    if ticker in LEVERAGED_ETFS:
        return LEVERAGED_ETFS[ticker].get("long", []), LEVERAGED_ETFS[ticker].get("short", [])
    return [], []


def get_upcoming_key_events(days_ahead=7):
    if investpy is None:
        return []
    today = datetime.today()
    end_date = today + timedelta(days=days_ahead)
    from_date = today.strftime('%d/%m/%Y')
    to_date = end_date.strftime('%d/%m/%Y')
    try:
        calendar = investpy.news.economic_calendar(
            time_zone=None, time_filter='time_only', countries=['united states'],
            importances=['high'], categories=None, from_date=from_date, to_date=to_date
        )
        if calendar.empty:
            return []
        pattern = '|'.join(KEY_EVENTS)
        filtered = calendar[
            (calendar['event'].str.contains(pattern, case=False, na=False)) &
            (calendar['importance'].str.lower() == 'high')
        ]
        if filtered.empty:
            return []
        filtered = filtered.sort_values(['date', 'time'])
        return filtered[['date', 'time', 'event']].to_dict('records')
    except Exception as e:
        print("Economic calendar error:", e)
        return []


def calculate_atr(hist_data, period=14):
    try:
        hl = hist_data['High'] - hist_data['Low']
        hc = (hist_data['High'] - hist_data['Close'].shift()).abs()
        lc = (hist_data['Low'] - hist_data['Close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    except Exception:
        return None


def calculate_rrs(stock_data, spy_data, atr_length=14, length_rolling=50, length_sma=20, atr_multiplier=1.0):
    try:
        merged = pd.merge(
            stock_data[['High', 'Low', 'Close']], spy_data[['High', 'Low', 'Close']],
            left_index=True, right_index=True, suffixes=('_stock', '_spy'), how='inner'
        )
        if len(merged) < atr_length + 1:
            return None
        for prefix in ['stock', 'spy']:
            h, l, c = merged[f'High_{prefix}'], merged[f'Low_{prefix}'], merged[f'Close_{prefix}']
            tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
            merged[f'atr_{prefix}'] = tr.ewm(alpha=1/atr_length, adjust=False).mean()
        sc = merged['Close_stock'] - merged['Close_stock'].shift(1)
        spy_c = merged['Close_spy'] - merged['Close_spy'].shift(1)
        spy_pi = spy_c / merged['atr_spy']
        expected = spy_pi * merged['atr_stock'] * atr_multiplier
        rrs = (sc - expected) / merged['atr_stock']
        rolling_rrs = rrs.rolling(window=length_rolling, min_periods=1).mean()
        rrs_sma = rolling_rrs.rolling(window=length_sma, min_periods=1).mean()
        return pd.DataFrame({'RRS': rrs, 'rollingRRS': rolling_rrs, 'RRS_SMA': rrs_sma}, index=merged.index)
    except Exception:
        return None


def calculate_sma(hist_data, period=50):
    try:
        return hist_data['Close'].rolling(window=period).mean().iloc[-1]
    except Exception:
        return None


def calculate_ema(hist_data, period=10):
    try:
        return hist_data['Close'].ewm(span=period, adjust=False).mean().iloc[-1]
    except Exception:
        return None


def calculate_abc_rating(hist_data):
    try:
        ema10 = calculate_ema(hist_data, 10)
        ema20 = calculate_ema(hist_data, 20)
        sma50 = calculate_sma(hist_data, 50)
        if ema10 is None or ema20 is None or sma50 is None:
            return None
        if ema10 > ema20 and ema20 > sma50:
            return "A"
        if (ema10 > ema20 and ema20 < sma50) or (ema10 < ema20 and ema20 > sma50):
            return "B"
        if ema10 < ema20 and ema20 < sma50:
            return "C"
    except Exception:
        pass
    return None


def create_rs_chart_png(rrs_data, ticker, charts_dir):
    try:
        recent = rrs_data.tail(20)
        if len(recent) == 0:
            return None
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 2))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        rolling_rrs = recent['rollingRRS'].values
        rrs_sma = recent['RRS_SMA'].values
        max_idx = rolling_rrs.argmax()
        bar_colors = ['#4ade80' if i == max_idx else '#b0b0b0' for i in range(len(rolling_rrs))]
        ax.bar(range(len(rolling_rrs)), rolling_rrs, color=bar_colors, width=0.8, edgecolor='none')
        ax.plot(range(len(rrs_sma)), rrs_sma, color='yellow', lw=2)
        ax.axhline(y=0, color='#808080', linestyle='--', linewidth=1)
        mn = min(rolling_rrs.min(), rrs_sma.min() if len(rrs_sma) else 0)
        mx = max(rolling_rrs.max(), rrs_sma.max() if len(rrs_sma) else 0)
        pad = 0.1 if mn == mx else (mx - mn) * 0.2
        ax.set_ylim(mn - pad, mx + pad)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        fig.tight_layout(pad=0)
        safe = re.sub(r'[^a-zA-Z0-9]', '_', ticker)
        path = os.path.join(charts_dir, f"{safe}.png")
        fig.savefig(path, format='png', dpi=80, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        return f"data/charts/{safe}.png"
    except Exception as e:
        print("Chart error", ticker, e)
        return None


def get_stock_data(ticker_symbol, charts_dir):
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="21d")
        daily = stock.history(period="60d")
        if len(hist) < 2 or len(daily) < 50:
            return None

        daily_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
        intraday_change = (hist['Close'].iloc[-1] / hist['Open'].iloc[-1] - 1) * 100
        five_day_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[-6] - 1) * 100 if len(hist) >= 6 else None
        twenty_day_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[-21] - 1) * 100 if len(hist) >= 21 else None

        sma50 = calculate_sma(daily)
        atr = calculate_atr(daily)
        current_close = daily['Close'].iloc[-1]
        atr_pct = (atr / current_close) * 100 if atr and current_close else None
        dist_sma50_atr = (100 * (current_close / sma50 - 1) / atr_pct) if (sma50 and atr_pct and atr_pct != 0) else None
        abc_rating = calculate_abc_rating(daily)

        rs_sts = None
        rrs_data = None
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)
        try:
            stock_history = stock.history(start=start_date, end=end_date)
            spy_history = yf.Ticker("SPY").history(start=start_date, end=end_date)
            if stock_history is not None and spy_history is not None:
                rrs_data = calculate_rrs(stock_history, spy_history, atr_length=14, length_rolling=50, length_sma=20, atr_multiplier=1.0)
                if rrs_data is not None and len(rrs_data) >= 21:
                    recent_21 = rrs_data['rollingRRS'].iloc[-21:]
                    ranks = rankdata(recent_21, method='average')
                    rs_sts = ((ranks[-1] - 1) / (len(recent_21) - 1)) * 100
        except Exception as e:
            print("RRS error", ticker_symbol, e)

        rs_chart_path = create_rs_chart_png(rrs_data, ticker_symbol, charts_dir) if rrs_data is not None and len(rrs_data) > 0 else None
        long_etfs, short_etfs = get_leveraged_etfs(ticker_symbol)

        return {
            "ticker": ticker_symbol,
            "daily": round(daily_change, 2) if daily_change is not None else None,
            "intra": round(intraday_change, 2) if intraday_change is not None else None,
            "5d": round(five_day_change, 2) if five_day_change is not None else None,
            "20d": round(twenty_day_change, 2) if twenty_day_change is not None else None,
            "atr_pct": round(atr_pct, 1) if atr_pct is not None else None,
            "dist_sma50_atr": round(dist_sma50_atr, 2) if dist_sma50_atr is not None else None,
            "rs": round(rs_sts, 0) if rs_sts is not None else None,
            "rs_chart": rs_chart_path,
            "long": long_etfs,
            "short": short_etfs,
            "abc": abc_rating
        }
    except Exception as e:
        print("Error", ticker_symbol, e)
        return None


# ── Screener functions ────────────────────────────────────────────────────────

def get_index_components(etf_symbol):
    """Get holdings of SPY or QQQ via Wikipedia"""
    try:
        if etf_symbol == 'SPY':
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return table['Symbol'].str.replace('.', '-', regex=False).tolist()
        elif etf_symbol == 'QQQ':
            tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            for t in tables:
                if 'Ticker' in t.columns:
                    return t['Ticker'].tolist()
        return []
    except Exception as e:
        print(f"  Could not fetch {etf_symbol} components: {e}")
        return []


def get_screener_universe():
    """Build universe from SPY + QQQ components"""
    print("  Fetching index components for screener universe...")
    spy = get_index_components('SPY')
    qqq = get_index_components('QQQ')
    universe = list(set(spy + qqq))
    print(f"  Universe size: {len(universe)} tickers (SPY+QQQ)")
    return universe


def run_screener(universe, screener_type='trend', top_n=10):
    """
    screener_type='trend':
        Price > $20, Price > SMA50, Price > SMA200, Avg Volume > 500K

    screener_type='momentum':
        Price > $20, Avg Volume > 500K, Relative Volume > 3x, Today > +10%
    """
    print(f"  Running {screener_type} screener on {len(universe)} tickers...")
    results = []
    spy_cache = {}  # reuse SPY data

    for i, ticker in enumerate(universe):
        try:
            stock = yf.Ticker(ticker)

            if screener_type == 'trend':
                hist = stock.history(period='1y')
                if len(hist) < 200:
                    continue
                price = hist['Close'].iloc[-1]
                if price < 20:
                    continue
                avg_vol = hist['Volume'].iloc[-21:-1].mean()
                if avg_vol < 500000:
                    continue
                sma50 = hist['Close'].rolling(50).mean().iloc[-1]
                sma200 = hist['Close'].rolling(200).mean().iloc[-1]
                if not (price > sma50 and price > sma200):
                    continue
                score = price / sma50
                results.append((ticker, score))

            elif screener_type == 'momentum':
                hist = stock.history(period='60d')
                if len(hist) < 22:
                    continue
                price = hist['Close'].iloc[-1]
                if price < 20:
                    continue
                avg_vol = hist['Volume'].iloc[-21:-1].mean()
                if avg_vol < 500000:
                    continue
                today_open = hist['Open'].iloc[-1]
                today_close = hist['Close'].iloc[-1]
                today_pct = (today_close / today_open - 1) * 100
                if today_pct < 10:
                    continue
                today_vol = hist['Volume'].iloc[-1]
                rel_vol = today_vol / avg_vol if avg_vol > 0 else 0
                if rel_vol < 3:
                    continue
                results.append((ticker, today_pct))

            if i % 50 == 0:
                print(f"    [{screener_type}] Screened {i}/{len(universe)}...")
            time.sleep(0.08)
        except Exception:
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    top = [t for t, _ in results[:top_n]]
    print(f"  {screener_type} screener: {len(results)} matches, top {len(top)}: {top}")
    return top


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data", help="Output directory (default: data)")
    args = parser.parse_args()
    out_dir = args.out_dir
    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("Fetching economic events...")
    events = get_upcoming_key_events()

    # ── Load manual watchlist ─────────────────────────────────────────────
    manual_path = os.path.join(repo_root, "watchlist_manual.txt")
    manual_tickers = []
    if os.path.exists(manual_path):
        with open(manual_path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip().upper()
                if t and not t.startswith("#"):
                    manual_tickers.append(t)
        print(f"Loaded {len(manual_tickers)} manual tickers from watchlist_manual.txt")
    else:
        print("watchlist_manual.txt not found — create it in repo root to add manual tickers.")

    # ── Run screeners ─────────────────────────────────────────────────────
    print("Running screeners...")
    universe = get_screener_universe()
    trend_tickers    = run_screener(universe, screener_type='trend',    top_n=10)
    momentum_tickers = run_screener(universe, screener_type='momentum', top_n=10)
    screener_tickers = list(dict.fromkeys(trend_tickers + momentum_tickers))  # dedup, keep order

    # Save screener results so it's visible in repo
    screener_out = os.path.join(repo_root, "watchlist_screener.txt")
    with open(screener_out, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by screener — do not edit manually\n")
        f.write("# Trend Screener (Price>SMA50>SMA200, AvgVol>500K) — Top 10\n")
        for t in trend_tickers:
            f.write(t + "\n")
        f.write("# Momentum Screener (RelVol>3x, Today>+10%, AvgVol>500K) — Top 10\n")
        for t in momentum_tickers:
            f.write(t + "\n")

    # ── Build all groups ──────────────────────────────────────────────────
    all_groups = {}
    if manual_tickers:
        all_groups["Watchlist"] = manual_tickers
    if screener_tickers:
        all_groups["Screener"] = screener_tickers
    all_groups.update(STOCK_GROUPS)

    print("Fetching stock data...")
    groups_data = {}
    all_ticker_data = {}
    for group_name, tickers in all_groups.items():
        rows = []
        for i, ticker in enumerate(tickers):
            print(f"  [{group_name}] {i+1}/{len(tickers)} {ticker}")
            if ticker in all_ticker_data:
                rows.append(all_ticker_data[ticker])
                continue
            row = get_stock_data(ticker, charts_dir)
            if row:
                rows.append(row)
                all_ticker_data[ticker] = row
            time.sleep(0.15)
        groups_data[group_name] = rows

    print("Computing column ranges...")
    column_ranges = {}
    for group_name, rows in groups_data.items():
        daily_v  = [r["daily"] for r in rows if r.get("daily")  is not None]
        intra_v  = [r["intra"] for r in rows if r.get("intra")  is not None]
        five_v   = [r["5d"]    for r in rows if r.get("5d")     is not None]
        twenty_v = [r["20d"]   for r in rows if r.get("20d")    is not None]
        column_ranges[group_name] = {
            "daily": (min(daily_v)  if daily_v  else -10, max(daily_v)  if daily_v  else 10),
            "intra": (min(intra_v)  if intra_v  else -10, max(intra_v)  if intra_v  else 10),
            "5d":    (min(five_v)   if five_v   else -20, max(five_v)   if five_v   else 20),
            "20d":   (min(twenty_v) if twenty_v else -30, max(twenty_v) if twenty_v else 30),
        }

    snapshot = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "groups": groups_data,
        "column_ranges": column_ranges,
    }
    meta = {
        "SECTOR_COLORS": SECTOR_COLORS,
        "TICKER_TO_SECTOR": TICKER_TO_SECTOR,
        "Industries_COLORS": Industries_COLORS,
        "SECTOR_ORDER": list(SECTOR_COLORS.keys()),
        "default_symbol": STOCK_GROUPS["Indices"][0] if STOCK_GROUPS["Indices"] else "SPY",
    }

    snapshot_path = os.path.join(out_dir, "snapshot.json")
    events_path   = os.path.join(out_dir, "events.json")
    meta_path     = os.path.join(out_dir, "meta.json")

    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Wrote", snapshot_path, events_path, meta_path, "and charts in", charts_dir)


if __name__ == "__main__":
    main()
