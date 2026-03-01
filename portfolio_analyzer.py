"""
Stock Portfolio Analysis Script with Technical & Fundamental Analysis
Version: 4.0 — SIP Entry Intelligence Edition

KEY IMPROVEMENT OVER v3:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OLD PROBLEM: Script recommended BUY when stock was already bullish
(price already up). You bought at top, stock dipped, no money left,
next month stock shows BEARISH and gets skipped entirely. You always
missed buying at the dip and lost your SIP discipline.

NEW SOLUTION: Two separate signals per stock:
  1. TRADING RECOMMENDATION (is this a good stock?) — unchanged
  2. SIP ENTRY SCORE (is NOW a good PRICE to buy?) — NEW

The SIP Entry Score is HIGH when price is LOW (dip = opportunity).
It is LOW when price is HIGH (rally = wait). This means:
  - Even if stock is "BUY", if price is stretched → INVEST LESS
  - Even if stock is "HOLD", if price dipped → ACCUMULATE NOW

Allocation is now weighted by SIP Entry Score, not just stock quality.
Stocks at dips get MORE allocation. Stocks at highs get LESS.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import warnings
import time
import sys
import os
warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────
MONTHLY_INVESTMENT = 48000  # INR

PORTFOLIO_STOCKS = [
    "BANKBEES.NS", "HDFCBANK.NS", "ITBEES.NS", "TECHM.NS",
    "TCS.NS", "INFY.NS", "RELIANCE.NS", "TMPV.NS",
    "NIFTYBEES.NS", "M&M.NS", "GOLDBEES.NS", "SILVERBEES.NS"
]

SECTOR_MAPPING = {
    "Banking & Finance":       ["BANKBEES.NS", "HDFCBANK.NS"],
    "Information Technology":  ["ITBEES.NS", "TECHM.NS", "TCS.NS", "INFY.NS"],
    "Index & Diversified":     ["NIFTYBEES.NS", "RELIANCE.NS", "TMPV.NS"],
    "Automobile":              ["M&M.NS"],
    "Commodities":             ["GOLDBEES.NS", "SILVERBEES.NS"],
}

GITHUB_USERNAME = os.environ.get('GITHUB_REPOSITORY', '').split('/')[0] \
    if '/' in os.environ.get('GITHUB_REPOSITORY', '') else "your-username"
GITHUB_REPO = os.environ.get('GITHUB_REPOSITORY', '').split('/')[1] \
    if '/' in os.environ.get('GITHUB_REPOSITORY', '') else "your-repo"

EMAIL_CONFIG = {
    'enabled':          False,
    'recipient_email':  "your.email@gmail.com",
    'sender_email':     "sender.email@gmail.com",
    'sender_password':  "your-app-specific-password",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SIP ENTRY INTELLIGENCE ENGINE
#  Question answered: "Is this a smart PRICE to invest this month?"
#  This is SEPARATE from whether the stock is fundamentally/technically good.
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_sip_entry_score(data, current_price, sma_50, rsi, week52_high, week52_low):
    """
    SIP Entry Score (0–100):
    HIGH score = price is at a good entry level (buy the dip)
    LOW score  = price is stretched / near highs (wait for pullback)

    4 components:
      1. RSI Zone          (30 pts) — lower RSI = less overbought = better entry
      2. Price vs SMA-50   (25 pts) — below SMA-50 = value zone
      3. 52-Week Position  (25 pts) — closer to 52W low = better entry
      4. Recent Drawdown   (20 pts) — recent pullback = buying opportunity
    """
    score = 0

    # 1. RSI Zone
    if rsi and rsi > 0:
        if   rsi < 30:  score += 30   # Oversold — best possible entry
        elif rsi < 40:  score += 27   # Very attractive
        elif rsi < 50:  score += 22   # Good, below midpoint
        elif rsi < 55:  score += 15   # Neutral / acceptable
        elif rsi < 65:  score += 8    # Slightly overbought
        elif rsi < 70:  score += 3    # Overbought
        else:           score += 0    # Very overbought — avoid this month

    # 2. Price vs SMA-50 (buying below SMA-50 = value zone)
    if current_price and sma_50 and sma_50 > 0:
        pct = ((current_price - sma_50) / sma_50) * 100
        if   pct < -5:  score += 25   # Well below SMA-50 — deep value
        elif pct < 0:   score += 20   # Slightly below — good entry
        elif pct < 3:   score += 12   # Just above — fair
        elif pct < 8:   score += 5    # Moderately above — pricey
        else:           score += 0    # Far above SMA-50 — stretched

    # 3. 52-Week Position (where in the annual range is the price?)
    if week52_high and week52_low and week52_high > week52_low and current_price:
        pos = (current_price - week52_low) / (week52_high - week52_low) * 100
        if   pos < 25:  score += 25   # Bottom 25% of range — excellent accumulation
        elif pos < 40:  score += 20   # Lower half — good value
        elif pos < 55:  score += 14   # Mid range — neutral
        elif pos < 70:  score += 7    # Upper half — slight premium
        elif pos < 85:  score += 2    # Near highs — cautious
        else:           score += 0    # Near 52W high — wait

    # 4. Recent drawdown from 20-day high (recent dip = opportunity)
    if data is not None and len(data) >= 20:
        recent_high = data['Close'].rolling(20).max().iloc[-1]
        if recent_high and recent_high > 0 and current_price:
            dd = ((current_price - recent_high) / recent_high) * 100
            if   dd < -10: score += 20   # Significant pullback — strong buy zone
            elif dd < -5:  score += 16   # Healthy pullback
            elif dd < -2:  score += 11   # Minor dip
            elif dd < 0:   score += 7    # Slight retreat
            else:          score += 2    # At/near recent high — no margin of safety

    return round(min(score, 100), 1)


def get_sip_verdict(sip_score, combined_score):
    """
    Combines stock quality (combined_score) with entry timing (sip_score)
    into a single actionable SIP verdict for this month.
    """
    if combined_score < 25:
        return "SKIP",            "#ef4444", "⛔"   # Fundamentally weak — skip regardless of price

    if   sip_score >= 70:  return "ACCUMULATE NOW",  "#00c853", "🟢"  # Price in value zone — invest full/more
    elif sip_score >= 55:  return "GOOD ENTRY",       "#43a047", "✅"  # Decent price — invest as planned
    elif sip_score >= 40:  return "INVEST NORMALLY",  "#fb8c00", "🟡"  # Fair price — stick to SIP
    elif sip_score >= 25:  return "INVEST LESS",       "#ff7043", "🟠"  # Slightly stretched — invest 50-70%
    else:                  return "WAIT THIS MONTH",  "#e53935", "🔴"  # Near highs — save for next dip


def get_sip_advice(sip_score, combined_score, current_price, sma_50, rsi, week52_low, week52_high):
    """Plain English advice for the investor."""
    verdict, _, _ = get_sip_verdict(sip_score, combined_score)

    pos_str = ""
    if week52_high and week52_low and week52_high > week52_low and current_price:
        pos = (current_price - week52_low) / (week52_high - week52_low) * 100
        pos_str = f" Price is at {pos:.0f}% of its 52-week range."

    sma_str = ""
    if current_price and sma_50 and sma_50 > 0:
        diff = ((current_price - sma_50) / sma_50) * 100
        sma_str = f" {abs(diff):.1f}% {'above' if diff > 0 else 'below'} SMA-50."

    rsi_str = f" RSI: {rsi:.0f}." if rsi else ""

    if verdict == "ACCUMULATE NOW":
        return f"Price is in the value zone.{pos_str}{sma_str}{rsi_str} Strong buying opportunity — consider investing your full SIP amount or slightly more this month."
    elif verdict == "GOOD ENTRY":
        return f"Decent entry point.{pos_str}{sma_str}{rsi_str} Invest your planned amount."
    elif verdict == "INVEST NORMALLY":
        return f"Fair price.{pos_str}{sma_str}{rsi_str} Stick to your regular SIP. Discipline matters more than timing here."
    elif verdict == "INVEST LESS":
        return f"Price is slightly stretched.{pos_str}{sma_str}{rsi_str} Consider investing 50–70% of your planned amount and saving the rest for when the price dips."
    elif verdict == "WAIT THIS MONTH":
        return f"Price is near recent highs.{pos_str}{sma_str}{rsi_str} Save this month's amount in a liquid fund and deploy it when RSI drops below 50 or price pulls back 5%+ from current level."
    else:
        return f"Stock quality is weak (score: {combined_score:.0f}/100). Skip this month and re-evaluate the stock."


# ═══════════════════════════════════════════════════════════════════════════════
#  STOCK ANALYZER (unchanged from v3.1)
# ═══════════════════════════════════════════════════════════════════════════════

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock  = yf.Ticker(ticker)
        self.data   = None
        self.info   = None

    def fetch_data(self, period="6mo", max_retries=3):
        for attempt in range(max_retries):
            try:
                print(f"  Fetching {self.ticker} (attempt {attempt+1}/{max_retries})...")
                self.data = self.stock.history(period=period)
                self.info = self.stock.info
                if self.data is None or len(self.data) == 0:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return False
                print(f"  ✅ {len(self.data)} days fetched")
                return True
            except Exception as e:
                print(f"  ❌ {self.ticker}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return False
        return False

    def calculate_rsi(self, period=14):
        if self.data is None or len(self.data) < period:
            return 0
        delta  = self.data['Close'].diff()
        gain   = delta.where(delta > 0, 0).rolling(period).mean()
        loss   = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rsi    = 100 - (100 / (1 + gain / loss))
        result = rsi.iloc[-1]
        return 0 if pd.isna(result) else result

    def calculate_macd(self):
        if self.data is None or len(self.data) < 26:
            return 0, 0, 0
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        sig  = macd.ewm(span=9, adjust=False).mean()
        hist = macd - sig
        def clean(v): return 0 if pd.isna(v) else v
        return clean(macd.iloc[-1]), clean(sig.iloc[-1]), clean(hist.iloc[-1])

    def calculate_moving_averages(self):
        if self.data is None or len(self.data) == 0:
            return 0, 0, 0, 0
        def ma(n): 
            v = self.data['Close'].rolling(n).mean().iloc[-1] if len(self.data) >= n else 0
            return 0 if pd.isna(v) else v
        cp = self.data['Close'].iloc[-1]
        return (0 if pd.isna(cp) else cp, ma(20), ma(50), ma(200))

    def calculate_bollinger_bands(self, period=20):
        if self.data is None or len(self.data) < period:
            return 0, 0, 0
        sma = self.data['Close'].rolling(period).mean()
        std = self.data['Close'].rolling(period).std()
        def clean(v): return 0 if pd.isna(v) else v
        return clean((sma+std*2).iloc[-1]), clean(sma.iloc[-1]), clean((sma-std*2).iloc[-1])

    def get_fundamental_data(self):
        if self.info is None:
            return {}
        i = self.info
        def pct(key): return (i.get(key, 0) or 0) * 100
        return {
            'market_cap':     i.get('marketCap', 0),
            'pe_ratio':       i.get('trailingPE', 0),
            'forward_pe':     i.get('forwardPE', 0),
            'peg_ratio':      i.get('pegRatio', 0),
            'dividend_yield': pct('dividendYield'),
            'beta':           i.get('beta', 0),
            'profit_margin':  pct('profitMargins'),
            'roe':            pct('returnOnEquity'),
            'debt_to_equity': i.get('debtToEquity', 0),
            'revenue_growth': pct('revenueGrowth'),
            'current_price':  i.get('currentPrice', 0),
            '52w_high':       i.get('fiftyTwoWeekHigh', 0),
            '52w_low':        i.get('fiftyTwoWeekLow', 0),
        }

    def calculate_technical_score(self):
        score = 0
        rsi   = self.calculate_rsi()
        if rsi > 0:
            if 30 <= rsi <= 70:   score += 20
            elif 70 < rsi <= 80:  score += 10
            elif 20 <= rsi < 30:  score += 15

        _, _, hist = self.calculate_macd()
        if hist > 0:    score += 20
        elif hist > -0.5: score += 10

        cp, s20, s50, s200 = self.calculate_moving_averages()
        if cp and s20  and cp > s20:   score += 10
        if cp and s50  and cp > s50:   score += 10
        if cp and s200 and cp > s200:  score += 10

        upper, _, lower = self.calculate_bollinger_bands()
        if cp and lower and upper and upper > lower:
            bp = (cp - lower) / (upper - lower)
            if 0.2 <= bp <= 0.5:  score += 15
            elif 0.5 < bp <= 0.8: score += 10

        if self.data is not None and len(self.data) >= 30:
            chg = (self.data['Close'].iloc[-1] - self.data['Close'].iloc[-30]) \
                  / self.data['Close'].iloc[-30] * 100
            if not pd.isna(chg):
                if chg > 5:    score += 15
                elif chg > 0:  score += 10
                elif chg > -5: score += 5
        return round(score, 2)

    def calculate_fundamental_score(self):
        score = 0
        f = self.get_fundamental_data()

        pe = f.get('pe_ratio', 0)
        if 0 < pe < 15:       score += 20
        elif 15 <= pe < 25:   score += 15
        elif 25 <= pe < 35:   score += 10

        dy = f.get('dividend_yield', 0)
        if dy > 3:     score += 15
        elif dy > 1.5: score += 10
        elif dy > 0.5: score += 5

        roe = f.get('roe', 0)
        if roe > 20:   score += 20
        elif roe > 15: score += 15
        elif roe > 10: score += 10

        pm = f.get('profit_margin', 0)
        if pm > 20:   score += 15
        elif pm > 10: score += 10
        elif pm > 5:  score += 5

        rg = f.get('revenue_growth', 0)
        if rg > 20:   score += 15
        elif rg > 10: score += 10
        elif rg > 0:  score += 5

        de = f.get('debt_to_equity', 0)
        if 0 <= de < 0.5:   score += 15
        elif 0.5 <= de < 1: score += 10
        elif 1 <= de < 2:   score += 5
        return round(score, 2)

    def get_recommendation(self):
        tech  = self.calculate_technical_score()
        fund  = self.calculate_fundamental_score()
        combo = tech * 0.5 + fund * 0.5
        if combo >= 70: return "STRONG BUY", combo, tech, fund
        if combo >= 55: return "BUY",        combo, tech, fund
        if combo >= 40: return "HOLD",       combo, tech, fund
        if combo >= 25: return "REDUCE",     combo, tech, fund
        return           "SELL",             combo, tech, fund


# ═══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def get_sector_for_stock(ticker):
    for sector, stocks in SECTOR_MAPPING.items():
        if ticker in stocks:
            return sector
    return "Other"


def get_ist_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))


def analyze_portfolio(stocks):
    results    = []
    successful = 0
    failed     = 0

    print(f"\n{'='*70}\nAnalyzing {len(stocks)} stocks...\n{'='*70}\n")

    for idx, ticker in enumerate(stocks, 1):
        print(f"[{idx}/{len(stocks)}] {ticker}...")
        analyzer = StockAnalyzer(ticker)

        if analyzer.fetch_data():
            try:
                rec, combo, tech, fund = analyzer.get_recommendation()
                cp, s20, s50, s200    = analyzer.calculate_moving_averages()
                rsi                   = analyzer.calculate_rsi()
                macd, signal, hist    = analyzer.calculate_macd()
                f                     = analyzer.get_fundamental_data()

                w52h = f.get('52w_high', 0)
                w52l = f.get('52w_low',  0)

                # ── NEW: SIP Entry Intelligence ────────────────────────────
                sip_score              = calculate_sip_entry_score(
                                             analyzer.data, cp, s50, rsi, w52h, w52l)
                sip_verdict, sip_color, sip_icon = get_sip_verdict(sip_score, combo)
                sip_advice             = get_sip_advice(
                                             sip_score, combo, cp, s50, rsi, w52l, w52h)

                results.append({
                    'ticker':            ticker,
                    'sector':            get_sector_for_stock(ticker),
                    'recommendation':    rec,
                    'combined_score':    combo,
                    'technical_score':   tech,
                    'fundamental_score': fund,
                    'current_price':     cp,
                    'rsi':               rsi,
                    'macd':              macd,
                    'signal':            signal,
                    'sma_20':            s20,
                    'sma_50':            s50,
                    'sma_200':           s200,
                    'pe_ratio':          f.get('pe_ratio', 0),
                    'dividend_yield':    f.get('dividend_yield', 0),
                    'market_cap':        f.get('market_cap', 0),
                    'roe':               f.get('roe', 0),
                    'profit_margin':     f.get('profit_margin', 0),
                    'revenue_growth':    f.get('revenue_growth', 0),
                    '52w_high':          w52h,
                    '52w_low':           w52l,
                    # SIP fields
                    'sip_score':         sip_score,
                    'sip_verdict':       sip_verdict,
                    'sip_color':         sip_color,
                    'sip_icon':          sip_icon,
                    'sip_advice':        sip_advice,
                })
                successful += 1
                print(f"  ✅ {rec} ({combo:.1f}) | SIP Entry: {sip_verdict} ({sip_score:.0f}/100)")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append(_err(ticker))
                failed += 1
        else:
            results.append(_err(ticker))
            failed += 1

        if idx < len(stocks):
            time.sleep(1)

    print(f"\n✅ {successful} succeeded | ❌ {failed} failed\n")
    if successful == 0:
        print("ERROR: No stocks analyzed. Exiting.")
        sys.exit(1)
    return results


def _err(ticker):
    return {
        'ticker': ticker, 'sector': get_sector_for_stock(ticker),
        'recommendation': 'DATA ERROR', 'combined_score': 0,
        'technical_score': 0, 'fundamental_score': 0, 'current_price': 0,
        'rsi': 0, 'macd': 0, 'signal': 0,
        'sma_20': 0, 'sma_50': 0, 'sma_200': 0,
        'pe_ratio': 0, 'dividend_yield': 0, 'market_cap': 0,
        'roe': 0, 'profit_margin': 0, 'revenue_growth': 0,
        '52w_high': 0, '52w_low': 0,
        'sip_score': 0, 'sip_verdict': 'DATA ERROR',
        'sip_color': '#9e9e9e', 'sip_icon': '⚠️',
        'sip_advice': 'Data unavailable — skip this month.',
    }


def calculate_allocation(results, monthly_investment):
    """
    NEW v4 allocation logic:
    Allocation weight = combined_score × sip_modifier

    sip_modifier boosts stocks at dips, reduces stocks at highs.
    Stocks with WAIT verdict get 0 allocation this month.
    Their amount is redistributed to stocks with better entry scores.
    """
    SIP_MODIFIERS = {
        "ACCUMULATE NOW":  1.4,   # Price in value zone → give more
        "GOOD ENTRY":      1.15,  # Good price → slight boost
        "INVEST NORMALLY": 1.0,   # Fair price → standard
        "INVEST LESS":     0.55,  # Stretched → reduce
        "WAIT THIS MONTH": 0.0,   # Near highs → skip this month
        "SKIP":            0.0,   # Weak stock → skip
        "DATA ERROR":      0.0,
    }

    eligible = [r for r in results
                if r.get('combined_score', 0) >= 25
                and SIP_MODIFIERS.get(r.get('sip_verdict', ''), 0) > 0]

    # Fallback: if everyone is at highs, equal split among quality stocks
    if not eligible:
        eligible = [r for r in results if r.get('combined_score', 0) >= 25]
        for r in results:
            r['sip_modifier_used'] = 1.0
        if not eligible:
            eligible = results

    total_weight = sum(
        r['combined_score'] * SIP_MODIFIERS.get(r.get('sip_verdict', ''), 1.0)
        for r in eligible
    )

    for r in results:
        mod = SIP_MODIFIERS.get(r.get('sip_verdict', ''), 0.0)
        r['sip_modifier_used'] = mod
        if r in eligible and total_weight > 0:
            w = (r['combined_score'] * mod) / total_weight
            r['allocation_amount']  = monthly_investment * w
            r['allocation_percent'] = w * 100
        else:
            r['allocation_amount']  = 0
            r['allocation_percent'] = 0

    return results


def group_by_sector(results):
    sectors = {}
    for r in results:
        sectors.setdefault(r.get('sector', 'Other'), []).append(r)
    for s in sectors:
        sectors[s].sort(key=lambda x: x['combined_score'], reverse=True)
    return sectors


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_github_pages_html(results, monthly_investment, github_url=""):
    ist        = get_ist_time()
    sectors    = group_by_sector(results)
    total_alloc = sum(r.get('allocation_amount', 0) for r in results)
    avg_score  = sum(r['combined_score'] for r in results) / len(results)
    strong_buy = sum(1 for r in results if r['recommendation'] == 'STRONG BUY')
    buy        = sum(1 for r in results if r['recommendation'] == 'BUY')
    hold       = sum(1 for r in results if r['recommendation'] == 'HOLD')
    reduce     = sum(1 for r in results if r['recommendation'] == 'REDUCE')
    sell       = sum(1 for r in results if r['recommendation'] == 'SELL')

    sip_counts = {
        'ACCUMULATE NOW':  sum(1 for r in results if r.get('sip_verdict') == 'ACCUMULATE NOW'),
        'GOOD ENTRY':      sum(1 for r in results if r.get('sip_verdict') == 'GOOD ENTRY'),
        'INVEST NORMALLY': sum(1 for r in results if r.get('sip_verdict') == 'INVEST NORMALLY'),
        'INVEST LESS':     sum(1 for r in results if r.get('sip_verdict') == 'INVEST LESS'),
        'WAIT THIS MONTH': sum(1 for r in results if r.get('sip_verdict') in ('WAIT THIS MONTH','SKIP','DATA ERROR')),
    }

    SECTOR_COLORS = {
        "Banking & Finance":      "#1e88e5",
        "Information Technology": "#43a047",
        "Index & Diversified":    "#fb8c00",
        "Automobile":             "#8e24aa",
        "Commodities":            "#f9a825",
    }
    SECTOR_ICONS = {
        "Banking & Finance":      "🏦",
        "Information Technology": "💻",
        "Index & Diversified":    "📈",
        "Automobile":             "🚗",
        "Commodities":            "🥇",
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Analysis — {ist.strftime('%d %b %Y')}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:       #0d1117;
  --surf:     #161b22;
  --surf2:    #1c2128;
  --border:   rgba(255,255,255,0.08);
  --text:     #e6edf3;
  --muted:    #7d8590;
  --green:    #3fb950;
  --yellow:   #d29922;
  --orange:   #f0883e;
  --red:      #f85149;
  --blue:     #58a6ff;
  --purple:   #bc8cff;
}}
*{{ margin:0; padding:0; box-sizing:border-box; }}
body{{
  font-family:'DM Sans',sans-serif;
  background:var(--bg);
  color:var(--text);
  min-height:100vh;
  font-size:14px;
  line-height:1.5;
}}

/* ── HEADER ─────────────────────────── */
.hdr{{
  background:var(--surf);
  border-bottom:1px solid var(--border);
  padding:24px 20px 20px;
}}
.hdr-inner{{
  max-width:1300px;
  margin:0 auto;
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  flex-wrap:wrap;
  gap:12px;
}}
.hdr-title{{
  font-family:'Syne',sans-serif;
  font-size:1.7em;
  font-weight:800;
  letter-spacing:-.5px;
}}
.hdr-title span{{color:var(--blue);}}
.hdr-sub{{font-size:.78em;color:var(--muted);margin-top:4px;font-family:'DM Mono',monospace;}}
.time-badge{{
  background:var(--surf2);
  border:1px solid var(--border);
  border-radius:8px;
  padding:8px 14px;
  font-family:'DM Mono',monospace;
  font-size:.75em;
  color:var(--muted);
  white-space:nowrap;
}}

/* ── OVERVIEW ────────────────────────── */
.ovr{{max-width:1300px;margin:24px auto 0;padding:0 16px;}}
.ovr-grid{{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:12px;
  margin-bottom:20px;
}}
.ov{{
  background:var(--surf);
  border:1px solid var(--border);
  border-radius:12px;
  padding:16px;
}}
.ov-lbl{{
  font-size:.65em;
  text-transform:uppercase;
  letter-spacing:1px;
  color:var(--muted);
  font-family:'DM Mono',monospace;
  margin-bottom:8px;
}}
.ov-val{{
  font-family:'Syne',sans-serif;
  font-size:1.8em;
  font-weight:700;
  line-height:1;
}}
.ov-sub{{font-size:.72em;color:var(--muted);margin-top:4px;}}

/* ── REC PILLS ───────────────────────── */
.rec-row{{
  display:flex;
  flex-wrap:wrap;
  gap:10px;
  margin-bottom:20px;
}}
.rec-pill{{
  flex:1;
  min-width:75px;
  background:var(--surf);
  border:1px solid var(--border);
  border-radius:10px;
  padding:12px 8px;
  text-align:center;
}}
.rec-pill .rp-n{{
  font-family:'Syne',sans-serif;
  font-size:1.9em;
  font-weight:800;
  line-height:1;
}}
.rec-pill .rp-l{{font-size:.62em;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:.5px;}}
.rec-pill.sb{{border-top:3px solid var(--green);}}
.rec-pill.b {{border-top:3px solid #7ee787;}}
.rec-pill.h {{border-top:3px solid var(--yellow);}}
.rec-pill.r {{border-top:3px solid var(--orange);}}
.rec-pill.s {{border-top:3px solid var(--red);}}

/* ── SIP INTELLIGENCE BANNER ─────────── */
.sip-banner{{
  background:linear-gradient(135deg,rgba(63,185,80,.06) 0%,rgba(0,0,0,0) 60%,rgba(88,166,255,.04) 100%);
  border:1px solid rgba(63,185,80,.2);
  border-radius:16px;
  padding:20px 22px;
  margin-bottom:20px;
}}
.sip-banner-hd{{
  font-family:'Syne',sans-serif;
  font-size:1em;
  font-weight:700;
  color:var(--green);
  margin-bottom:6px;
}}
.sip-banner-txt{{
  font-size:.78em;
  color:var(--muted);
  max-width:780px;
  line-height:1.6;
  margin-bottom:14px;
}}
.sip-banner-txt strong{{color:var(--text);}}
.sip-pills{{display:flex;flex-wrap:wrap;gap:8px;}}
.sp{{
  padding:5px 13px;
  border-radius:20px;
  font-family:'DM Mono',monospace;
  font-size:.7em;
  font-weight:500;
  display:flex;
  align-items:center;
  gap:5px;
}}
.sp.acc{{background:rgba(63,185,80,.12);color:#3fb950;border:1px solid rgba(63,185,80,.25);}}
.sp.good{{background:rgba(100,200,100,.08);color:#7ee787;border:1px solid rgba(100,200,100,.15);}}
.sp.norm{{background:rgba(210,153,34,.1);color:#d29922;border:1px solid rgba(210,153,34,.2);}}
.sp.less{{background:rgba(240,136,62,.1);color:#f0883e;border:1px solid rgba(240,136,62,.2);}}
.sp.wait{{background:rgba(248,81,73,.1);color:#f85149;border:1px solid rgba(248,81,73,.2);}}

/* ── SIP GUIDE ───────────────────────── */
.guide{{max-width:1300px;margin:0 auto 24px;padding:0 16px;}}
.guide-inner{{
  background:var(--surf);
  border:1px solid var(--border);
  border-radius:16px;
  padding:18px 22px;
}}
.guide-hd{{
  font-family:'Syne',sans-serif;
  font-size:.9em;
  font-weight:700;
  margin-bottom:14px;
}}
.guide-grid{{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:10px;
}}
.gi{{
  padding:12px;
  border-radius:10px;
  border:1px solid var(--border);
}}
.gi-v{{font-family:'DM Mono',monospace;font-size:.68em;font-weight:700;margin-bottom:5px;}}
.gi-w{{font-size:.74em;font-weight:600;margin-bottom:4px;}}
.gi-h{{font-size:.7em;color:var(--muted);line-height:1.45;}}

/* ── SECTORS ─────────────────────────── */
.secs{{max-width:1300px;margin:0 auto;padding:0 16px 40px;}}
.sec{{margin-bottom:28px;}}
.sec-hd{{
  display:flex;
  align-items:center;
  gap:10px;
  padding:12px 16px;
  background:var(--surf);
  border:1px solid var(--border);
  border-radius:10px;
  margin-bottom:12px;
  flex-wrap:wrap;
}}
.sec-name{{
  font-family:'Syne',sans-serif;
  font-weight:700;
  font-size:1em;
  flex:1;
}}
.sec-badge{{
  font-family:'DM Mono',monospace;
  font-size:.68em;
  padding:3px 9px;
  border-radius:6px;
  background:var(--surf2);
  color:var(--muted);
}}
.sec-alloc{{
  font-family:'DM Mono',monospace;
  font-size:.72em;
  padding:3px 10px;
  border-radius:6px;
  color:#fff;
  font-weight:600;
}}
.sgrid{{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
  gap:14px;
}}

/* ── STOCK CARD ───────────────────────── */
.card{{
  background:var(--surf);
  border:1px solid var(--border);
  border-radius:14px;
  overflow:hidden;
  transition:transform .2s,box-shadow .2s;
}}
.card:hover{{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.4);}}

/* card top row */
.ctop{{
  padding:14px 16px 10px;
  border-bottom:1px solid var(--border);
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
}}
.cticker{{
  font-family:'Syne',sans-serif;
  font-size:1.1em;
  font-weight:800;
}}
.cprice{{
  font-family:'DM Mono',monospace;
  font-size:.75em;
  color:var(--muted);
  margin-top:2px;
}}
.rec-b{{
  font-family:'DM Mono',monospace;
  font-size:.6em;
  font-weight:700;
  padding:3px 9px;
  border-radius:5px;
  letter-spacing:.3px;
  white-space:nowrap;
}}
.rb-strong-buy{{background:rgba(63,185,80,.18);color:#3fb950;}}
.rb-buy{{background:rgba(100,200,100,.12);color:#7ee787;}}
.rb-hold{{background:rgba(210,153,34,.18);color:#d29922;}}
.rb-reduce{{background:rgba(240,136,62,.18);color:#f0883e;}}
.rb-sell{{background:rgba(248,81,73,.18);color:#f85149;}}
.rb-data-error{{background:rgba(125,133,144,.18);color:#7d8590;}}

/* SIP verdict stripe */
.sip-stripe{{
  padding:10px 16px;
  display:flex;
  align-items:flex-start;
  gap:10px;
  border-bottom:1px solid var(--border);
}}
.sip-vbadge{{
  font-family:'DM Mono',monospace;
  font-size:.64em;
  font-weight:700;
  padding:4px 10px;
  border-radius:6px;
  white-space:nowrap;
  flex-shrink:0;
  margin-top:1px;
}}
.sip-txt{{font-size:.72em;color:var(--muted);line-height:1.5;}}

/* score bars */
.scores{{padding:10px 16px;border-bottom:1px solid var(--border);}}
.srow{{display:flex;align-items:center;gap:8px;margin-bottom:5px;}}
.srow:last-child{{margin-bottom:0;}}
.slbl{{
  font-size:.62em;
  color:var(--muted);
  font-family:'DM Mono',monospace;
  width:58px;
  flex-shrink:0;
}}
.sbar-w{{flex:1;background:var(--surf2);border-radius:3px;height:5px;overflow:hidden;}}
.sbar{{height:100%;border-radius:3px;}}
.snum{{font-family:'DM Mono',monospace;font-size:.65em;width:26px;text-align:right;}}

/* metrics row */
.mets{{
  display:grid;
  grid-template-columns:repeat(5,1fr);
  border-top:1px solid var(--border);
}}
.met{{padding:7px 4px;text-align:center;border-right:1px solid var(--border);}}
.met:last-child{{border-right:none;}}
.met-l{{font-size:.55em;color:var(--muted);font-family:'DM Mono',monospace;text-transform:uppercase;display:block;margin-bottom:2px;}}
.met-v{{font-family:'DM Mono',monospace;font-size:.78em;font-weight:600;}}

/* allocation footer */
.alloc-ft{{
  padding:10px 16px;
  background:var(--surf2);
  border-top:1px solid var(--border);
  display:flex;
  justify-content:space-between;
  align-items:center;
}}
.alloc-lbl{{font-size:.65em;color:var(--muted);font-family:'DM Mono',monospace;}}
.alloc-amt{{
  font-family:'Syne',sans-serif;
  font-size:1.05em;
  font-weight:700;
  text-align:right;
}}
.alloc-pct{{font-size:.68em;color:var(--muted);text-align:right;}}

/* disclaimer */
.disc{{max-width:1300px;margin:0 auto 24px;padding:0 16px;}}
.disc-inner{{
  background:rgba(210,153,34,.04);
  border:1px solid rgba(210,153,34,.2);
  border-radius:10px;
  padding:14px 18px;
  font-size:.75em;
  color:var(--muted);
  line-height:1.6;
}}
.disc-inner strong{{color:var(--yellow);}}

.footer{{
  border-top:1px solid var(--border);
  padding:16px 20px;
  text-align:center;
  font-size:.7em;
  color:var(--muted);
  font-family:'DM Mono',monospace;
}}

/* responsive */
@media(max-width:600px){{
  .ovr-grid{{grid-template-columns:repeat(2,1fr);}}
  .sgrid{{grid-template-columns:1fr;}}
  .guide-grid{{grid-template-columns:1fr 1fr;}}
  .mets{{grid-template-columns:repeat(3,1fr);}}
  .hdr-inner{{flex-direction:column;}}
}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="hdr">
  <div class="hdr-inner">
    <div>
      <div class="hdr-title">📊 <span>Portfolio</span> Analysis</div>
      <div class="hdr-sub">SIP Entry Intelligence Edition · v4.0</div>
    </div>
    <div class="time-badge">🕐 {ist.strftime('%d %b %Y · %I:%M %p IST')}</div>
  </div>
</div>

<!-- OVERVIEW -->
<div class="ovr">
  <div class="ovr-grid">
    <div class="ov">
      <div class="ov-lbl">Monthly SIP</div>
      <div class="ov-val" style="color:var(--blue);">₹{monthly_investment:,}</div>
      <div class="ov-sub">₹{total_alloc:,.0f} allocated</div>
    </div>
    <div class="ov">
      <div class="ov-lbl">Stocks</div>
      <div class="ov-val">{len(results)}</div>
      <div class="ov-sub">{len(sectors)} sectors</div>
    </div>
    <div class="ov">
      <div class="ov-lbl">Avg Score</div>
      <div class="ov-val" style="color:{'var(--green)' if avg_score>=55 else 'var(--yellow)' if avg_score>=40 else 'var(--red)'};">{avg_score:.1f}</div>
      <div class="ov-sub">Portfolio health</div>
    </div>
    <div class="ov">
      <div class="ov-lbl">Good Entries</div>
      <div class="ov-val" style="color:var(--green);">{sip_counts['ACCUMULATE NOW']+sip_counts['GOOD ENTRY']}</div>
      <div class="ov-sub">Accumulate + Good</div>
    </div>
    <div class="ov">
      <div class="ov-lbl">Wait/Skip</div>
      <div class="ov-val" style="color:var(--red);">{sip_counts['WAIT THIS MONTH']}</div>
      <div class="ov-sub">Deferred this month</div>
    </div>
  </div>

  <!-- SIP INTELLIGENCE BANNER -->
  <div class="sip-banner">
    <div class="sip-banner-hd">🎯 SIP Entry Intelligence — This Month</div>
    <div class="sip-banner-txt">
      Each stock now has <strong>two separate signals</strong>: stock quality (is it worth owning?)
      and <strong>SIP Entry Score</strong> (is NOW a good <em>price</em> to buy?).
      Allocation is automatically higher for stocks at dips and lower for stocks near highs —
      so your monthly SIP discipline is preserved but your money is deployed smarter.
    </div>
    <div class="sip-pills">
      <div class="sp acc">🟢 ACCUMULATE NOW · {sip_counts['ACCUMULATE NOW']}</div>
      <div class="sp good">✅ GOOD ENTRY · {sip_counts['GOOD ENTRY']}</div>
      <div class="sp norm">🟡 INVEST NORMALLY · {sip_counts['INVEST NORMALLY']}</div>
      <div class="sp less">🟠 INVEST LESS · {sip_counts['INVEST LESS']}</div>
      <div class="sp wait">🔴 WAIT THIS MONTH · {sip_counts['WAIT THIS MONTH']}</div>
    </div>
  </div>

  <!-- REC SUMMARY -->
  <div class="rec-row">
    <div class="rec-pill sb"><div class="rp-n" style="color:var(--green);">{strong_buy}</div><div class="rp-l">Strong Buy</div></div>
    <div class="rec-pill b"> <div class="rp-n" style="color:#7ee787;">{buy}</div><div class="rp-l">Buy</div></div>
    <div class="rec-pill h"> <div class="rp-n" style="color:var(--yellow);">{hold}</div><div class="rp-l">Hold</div></div>
    <div class="rec-pill r"> <div class="rp-n" style="color:var(--orange);">{reduce}</div><div class="rp-l">Reduce</div></div>
    <div class="rec-pill s"> <div class="rp-n" style="color:var(--red);">{sell}</div><div class="rp-l">Sell</div></div>
  </div>
</div>

<!-- SIP GUIDE -->
<div class="guide">
  <div class="guide-inner">
    <div class="guide-hd">📖 What each SIP verdict means — and what to do</div>
    <div class="guide-grid">
      <div class="gi" style="background:rgba(63,185,80,.04);border-color:rgba(63,185,80,.18);">
        <div class="gi-v" style="color:#3fb950;">🟢 ACCUMULATE NOW</div>
        <div class="gi-w">Invest full amount or slightly more</div>
        <div class="gi-h">RSI below 45, price near/below SMA-50, lower 40% of 52W range. Ideal time to add.</div>
      </div>
      <div class="gi" style="background:rgba(100,200,100,.03);border-color:rgba(100,200,100,.12);">
        <div class="gi-v" style="color:#7ee787;">✅ GOOD ENTRY</div>
        <div class="gi-w">Invest as planned</div>
        <div class="gi-h">Not a screaming deal but not stretched. Stick to your regular SIP amount.</div>
      </div>
      <div class="gi" style="background:rgba(210,153,34,.04);border-color:rgba(210,153,34,.12);">
        <div class="gi-v" style="color:#d29922;">🟡 INVEST NORMALLY</div>
        <div class="gi-w">Stick to regular SIP — discipline matters</div>
        <div class="gi-h">Fair price, mid-range valuation. Your consistent investing habit matters more than timing here.</div>
      </div>
      <div class="gi" style="background:rgba(240,136,62,.04);border-color:rgba(240,136,62,.12);">
        <div class="gi-v" style="color:#f0883e;">🟠 INVEST LESS</div>
        <div class="gi-w">Invest 50–70% of planned amount</div>
        <div class="gi-h">Price slightly stretched. Save the remaining 30–50% and deploy it when the stock dips next month.</div>
      </div>
      <div class="gi" style="background:rgba(248,81,73,.04);border-color:rgba(248,81,73,.12);">
        <div class="gi-v" style="color:#f85149;">🔴 WAIT THIS MONTH</div>
        <div class="gi-w">Save this month's amount for next dip</div>
        <div class="gi-h">Price near recent highs, RSI elevated. Park in liquid fund. Deploy when RSI &lt; 50 or price falls 5%+.</div>
      </div>
    </div>
  </div>
</div>

<!-- SECTORS -->
<div class="secs">
"""

    for sector, stocks in sectors.items():
        sc = SECTOR_COLORS.get(sector, "#58a6ff")
        si = SECTOR_ICONS.get(sector, "📊")
        sa = sum(s.get('allocation_amount', 0) for s in stocks)

        html += f"""
  <div class="sec">
    <div class="sec-hd" style="border-left:4px solid {sc};">
      <span style="font-size:1.2em;">{si}</span>
      <span class="sec-name" style="color:{sc};">{sector}</span>
      <span class="sec-badge">{len(stocks)} stocks</span>
      <span class="sec-alloc" style="background:{sc};">₹{sa:,.0f}</span>
    </div>
    <div class="sgrid">
"""

        for s in stocks:
            combo     = s.get('combined_score',    0) or 0
            tech      = s.get('technical_score',   0) or 0
            fund      = s.get('fundamental_score', 0) or 0
            sip_s     = s.get('sip_score',         0) or 0
            sip_v     = s.get('sip_verdict',       'N/A')
            sip_col   = s.get('sip_color',         '#7d8590')
            sip_ic    = s.get('sip_icon',          '')
            sip_adv   = s.get('sip_advice',        '')
            cp        = s.get('current_price',     0) or 0
            rsi       = s.get('rsi',               0) or 0
            macd      = s.get('macd',              0) or 0
            pe        = s.get('pe_ratio',          0) or 0
            roe       = s.get('roe',               0) or 0
            dy        = s.get('dividend_yield',    0) or 0
            alloc_amt = s.get('allocation_amount',  0) or 0
            alloc_pct = s.get('allocation_percent', 0) or 0

            rc = 'rb-' + s['recommendation'].lower().replace(' ','-')

            cb_col = '#3fb950' if combo >= 70 else '#d29922' if combo >= 45 else '#f85149'
            tb_col = '#58a6ff' if tech  >= 55 else '#d29922' if tech  >= 35 else '#f85149'
            fb_col = '#bc8cff' if fund  >= 55 else '#d29922' if fund  >= 35 else '#f85149'

            html += f"""
      <div class="card" style="border-top:3px solid {sc};">
        <!-- top -->
        <div class="ctop">
          <div>
            <div class="cticker">{s['ticker']}</div>
            <div class="cprice">₹{cp:.2f}</div>
          </div>
          <span class="rec-b {rc}">{s['recommendation']}</span>
        </div>
        <!-- SIP verdict -->
        <div class="sip-stripe">
          <span class="sip-vbadge" style="background:color-mix(in srgb,{sip_col} 14%,transparent);color:{sip_col};border:1px solid color-mix(in srgb,{sip_col} 28%,transparent);">
            {sip_ic} {sip_v}
          </span>
          <span class="sip-txt">{sip_adv}</span>
        </div>
        <!-- score bars -->
        <div class="scores">
          <div class="srow">
            <span class="slbl">OVERALL</span>
            <div class="sbar-w"><div class="sbar" style="width:{combo}%;background:{cb_col};"></div></div>
            <span class="snum" style="color:{cb_col};">{combo:.0f}</span>
          </div>
          <div class="srow">
            <span class="slbl">TECHNICAL</span>
            <div class="sbar-w"><div class="sbar" style="width:{tech}%;background:{tb_col};"></div></div>
            <span class="snum" style="color:{tb_col};">{tech:.0f}</span>
          </div>
          <div class="srow">
            <span class="slbl">FUNDMNTL</span>
            <div class="sbar-w"><div class="sbar" style="width:{fund}%;background:{fb_col};"></div></div>
            <span class="snum" style="color:{fb_col};">{fund:.0f}</span>
          </div>
          <div class="srow">
            <span class="slbl">SIP ENTRY</span>
            <div class="sbar-w"><div class="sbar" style="width:{sip_s}%;background:{sip_col};"></div></div>
            <span class="snum" style="color:{sip_col};">{sip_s:.0f}</span>
          </div>
        </div>
        <!-- metrics -->
        <div class="mets">
          <div class="met">
            <span class="met-l">RSI</span>
            <span class="met-v" style="color:{'#f85149' if rsi>70 else '#3fb950' if rsi<40 else 'var(--text)'};">{rsi:.0f}</span>
          </div>
          <div class="met">
            <span class="met-l">MACD</span>
            <span class="met-v" style="color:{'#3fb950' if macd>0 else '#f85149'};">{macd:.1f}</span>
          </div>
          <div class="met"><span class="met-l">P/E</span><span class="met-v">{pe:.1f}</span></div>
          <div class="met"><span class="met-l">ROE</span><span class="met-v">{roe:.0f}%</span></div>
          <div class="met"><span class="met-l">DIV</span><span class="met-v">{dy:.1f}%</span></div>
        </div>
        <!-- allocation footer -->
        <div class="alloc-ft">
          <div class="alloc-lbl">THIS MONTH'S SIP</div>
          <div>
            <div class="alloc-amt" style="color:{sc};">{'₹'+f'{alloc_amt:,.0f}' if alloc_amt>0 else '— Deferred'}</div>
            <div class="alloc-pct">{'({:.1f}% of portfolio)'.format(alloc_pct) if alloc_pct>0 else 'Save for next dip'}</div>
          </div>
        </div>
      </div>
"""

        html += "    </div>\n  </div>\n"

    html += f"""
</div>

<!-- DISCLAIMER -->
<div class="disc">
  <div class="disc-inner">
    <strong>⚠️ Disclaimer:</strong> For educational purposes only. Not financial advice.
    SIP verdicts are algorithmic signals based on RSI, SMA-50, 52W range, and recent drawdown.
    They do not guarantee future returns. A "WAIT" verdict means defer this stock's SIP — not sell.
    Always consult a SEBI-registered advisor before investing.
  </div>
</div>

<div class="footer">
  Stock Quality: Technical (50%) — RSI · MACD · Moving Avg · Bollinger | Fundamental (50%) — P/E · ROE · Margin · Growth · Div · D/E<br>
  SIP Entry Score: RSI Zone · Price vs SMA-50 · 52-Week Position · Recent Drawdown from 20-day high<br>
  Portfolio Analyzer v4.0 — SIP Entry Intelligence Edition · {ist.strftime('%d %b %Y at %I:%M %p IST')}
</div>

</body>
</html>"""

    return html


def generate_email_html(results, monthly_investment, github_url=""):
    return generate_github_pages_html(results, monthly_investment, github_url)


def send_email(html_content, recipient_email, sender_email, sender_password):
    try:
        ist = get_ist_time()
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"📊 Portfolio SIP Report — {ist.strftime('%d %B %Y, %I:%M %p IST')}"
        msg['From']    = sender_email
        msg['To']      = recipient_email
        msg.attach(MIMEText(html_content, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as srv:
            srv.login(sender_email, sender_password)
            srv.send_message(msg)
        print(f"✅ Email sent to {recipient_email}")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False


def main():
    ist = get_ist_time()
    print("=" * 70)
    print("Portfolio Analyzer v4.0 — SIP Entry Intelligence Edition")
    print(f"Time: {ist.strftime('%d %B %Y, %I:%M %p IST')}")
    print("=" * 70)

    results = analyze_portfolio(PORTFOLIO_STOCKS)
    results = calculate_allocation(results, MONTHLY_INVESTMENT)

    github_url = f"https://{GITHUB_USERNAME}.github.io/{GITHUB_REPO}/"
    html       = generate_github_pages_html(results, MONTHLY_INVESTMENT, github_url)

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("✅ index.html saved")

    with open('portfolio_data.json', 'w') as f:
        json.dump({'generated': ist.isoformat(),
                   'monthly_investment': MONTHLY_INVESTMENT,
                   'stocks': results}, f, indent=2)
    print("✅ portfolio_data.json saved")

    # Email
    se = os.environ.get('SENDER_EMAIL')    or (EMAIL_CONFIG['sender_email']    if EMAIL_CONFIG['enabled'] else None)
    sp = os.environ.get('SENDER_PASSWORD') or (EMAIL_CONFIG['sender_password'] if EMAIL_CONFIG['enabled'] else None)
    re = os.environ.get('RECIPIENT_EMAIL') or (EMAIL_CONFIG['recipient_email'] if EMAIL_CONFIG['enabled'] else None)
    if se and sp and re:
        send_email(html, re, se, sp)

    # Terminal summary
    print("\n" + "=" * 70)
    print("SIP SUMMARY — THIS MONTH")
    print("=" * 70)
    sectors = group_by_sector(results)
    for sector, stks in sectors.items():
        st = sum(s.get('allocation_amount', 0) for s in stks)
        print(f"\n{sector} — ₹{st:,.0f}")
        print("-" * 70)
        for s in stks:
            amt = s.get('allocation_amount', 0) or 0
            pct = s.get('allocation_percent', 0) or 0
            print(f"  {s['ticker']:15} | Score:{s['combined_score']:5.1f}"
                  f" | SIP:{s.get('sip_score',0):5.1f}"
                  f" | {s.get('sip_verdict','N/A'):18}"
                  f" | ₹{amt:>9,.0f} ({pct:5.1f}%)")
    print("\n✅ Done!")
    return results


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
