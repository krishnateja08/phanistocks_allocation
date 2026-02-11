"""
Stock Portfolio Analysis Script with Technical & Fundamental Analysis
Author: Portfolio Manager
Date: February 2026
Version: 2.0 - Enhanced UI with IST timezone (Overwrite Mode)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz  # For timezone conversion
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import warnings
import time
import sys
import os
warnings.filterwarnings('ignore')

# Portfolio Configuration
MONTHLY_INVESTMENT = 60000  # INR
PORTFOLIO_STOCKS = [
    "BANKBEES.NS", "HDFCBANK.NS", "NXST-RR.NS", "EMBASSY-RR.NS",
    "BIRET-RR.NS", "MINDSPACE-RR.NS", "ITBEES.NS", "TECHM.NS",
    "TCS.NS", "INFY.NS", "RELIANCE.NS", "TMPV.NS",
    "NIFTYBEES.NS", "M&M.NS", "GOLDBEES.NS", "SILVERBEES.NS"
]

# GITHUB PAGES CONFIGURATION
GITHUB_USERNAME = os.environ.get('GITHUB_REPOSITORY', '').split('/')[0] if '/' in os.environ.get('GITHUB_REPOSITORY', '') else "your-username"
GITHUB_REPO = os.environ.get('GITHUB_REPOSITORY', '').split('/')[1] if '/' in os.environ.get('GITHUB_REPOSITORY', '') else "your-repo"

# EMAIL CONFIGURATION
EMAIL_CONFIG = {
    'enabled': False,
    'recipient_email': "your.email@gmail.com",
    'sender_email': "sender.email@gmail.com",
    'sender_password': "your-app-specific-password"
}


def get_ist_time():
    """Get current time in IST timezone"""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)


class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.data = None
        self.info = None
        
    def fetch_data(self, period="6mo", max_retries=3):
        """Fetch historical stock data with retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"  Fetching data for {self.ticker} (attempt {attempt + 1}/{max_retries})...")
                self.data = self.stock.history(period=period)
                self.info = self.stock.info
                
                if self.data is None or len(self.data) == 0:
                    print(f"  ⚠️ No data returned for {self.ticker}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return False
                
                print(f"  ✅ Successfully fetched {len(self.data)} days of data")
                return True
                
            except Exception as e:
                print(f"  ❌ Error fetching data for {self.ticker}: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"  Failed after {max_retries} attempts")
                    return False
        
        return False
    
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        if self.data is None or len(self.data) < period:
            return 0
        
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        result = rsi.iloc[-1]
        
        return 0 if pd.isna(result) else result
    
    def calculate_macd(self):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if self.data is None or len(self.data) < 26:
            return 0, 0, 0
        
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        macd_val = macd.iloc[-1]
        signal_val = signal.iloc[-1]
        histogram_val = histogram.iloc[-1]
        
        macd_val = 0 if pd.isna(macd_val) else macd_val
        signal_val = 0 if pd.isna(signal_val) else signal_val
        histogram_val = 0 if pd.isna(histogram_val) else histogram_val
        
        return macd_val, signal_val, histogram_val
    
    def calculate_moving_averages(self):
        """Calculate Simple Moving Averages"""
        if self.data is None or len(self.data) == 0:
            return 0, 0, 0, 0
        
        current_price = self.data['Close'].iloc[-1] if len(self.data) > 0 else 0
        sma_20 = self.data['Close'].rolling(window=20).mean().iloc[-1] if len(self.data) >= 20 else 0
        sma_50 = self.data['Close'].rolling(window=50).mean().iloc[-1] if len(self.data) >= 50 else 0
        sma_200 = self.data['Close'].rolling(window=200).mean().iloc[-1] if len(self.data) >= 200 else 0
        
        current_price = 0 if pd.isna(current_price) else current_price
        sma_20 = 0 if pd.isna(sma_20) else sma_20
        sma_50 = 0 if pd.isna(sma_50) else sma_50
        sma_200 = 0 if pd.isna(sma_200) else sma_200
        
        return current_price, sma_20, sma_50, sma_200
    
    def calculate_bollinger_bands(self, period=20):
        """Calculate Bollinger Bands"""
        if self.data is None or len(self.data) < period:
            return 0, 0, 0
        
        sma = self.data['Close'].rolling(window=period).mean()
        std = self.data['Close'].rolling(window=period).std()
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        upper = upper_band.iloc[-1]
        middle = sma.iloc[-1]
        lower = lower_band.iloc[-1]
        
        upper = 0 if pd.isna(upper) else upper
        middle = 0 if pd.isna(middle) else middle
        lower = 0 if pd.isna(lower) else lower
        
        return upper, middle, lower
    
    def get_fundamental_data(self):
        """Extract fundamental data"""
        if self.info is None:
            return {}
        
        fundamentals = {
            'market_cap': self.info.get('marketCap', 0),
            'pe_ratio': self.info.get('trailingPE', 0),
            'forward_pe': self.info.get('forwardPE', 0),
            'peg_ratio': self.info.get('pegRatio', 0),
            'dividend_yield': self.info.get('dividendYield', 0) * 100 if self.info.get('dividendYield') else 0,
            'beta': self.info.get('beta', 0),
            'profit_margin': self.info.get('profitMargins', 0) * 100 if self.info.get('profitMargins') else 0,
            'roe': self.info.get('returnOnEquity', 0) * 100 if self.info.get('returnOnEquity') else 0,
            'debt_to_equity': self.info.get('debtToEquity', 0),
            'revenue_growth': self.info.get('revenueGrowth', 0) * 100 if self.info.get('revenueGrowth') else 0,
            'current_price': self.info.get('currentPrice', 0),
            '52w_high': self.info.get('fiftyTwoWeekHigh', 0),
            '52w_low': self.info.get('fiftyTwoWeekLow', 0),
        }
        
        return fundamentals
    
    def calculate_technical_score(self):
        """Calculate technical analysis score (0-100)"""
        score = 0
        max_score = 100
        
        rsi = self.calculate_rsi()
        if rsi and rsi > 0:
            if 30 <= rsi <= 70:
                score += 20
            elif 70 < rsi <= 80:
                score += 10
            elif 20 <= rsi < 30:
                score += 15
        
        macd, signal, histogram = self.calculate_macd()
        if macd is not None and signal is not None and histogram is not None:
            if histogram > 0:
                score += 20
            elif histogram > -0.5:
                score += 10
        
        current, sma_20, sma_50, sma_200 = self.calculate_moving_averages()
        if current and sma_20 and current > 0 and sma_20 > 0:
            if current > sma_20:
                score += 10
        if current and sma_50 and current > 0 and sma_50 > 0:
            if current > sma_50:
                score += 10
        if current and sma_200 and current > 0 and sma_200 > 0:
            if current > sma_200:
                score += 10
        
        upper, middle, lower = self.calculate_bollinger_bands()
        if current and lower and upper and lower > 0 and upper > lower:
            band_position = (current - lower) / (upper - lower)
            if 0.2 <= band_position <= 0.5:
                score += 15
            elif 0.5 < band_position <= 0.8:
                score += 10
        
        if self.data is not None and len(self.data) >= 30:
            price_change_30d = ((self.data['Close'].iloc[-1] - self.data['Close'].iloc[-30]) / 
                               self.data['Close'].iloc[-30] * 100)
            if not pd.isna(price_change_30d):
                if price_change_30d > 5:
                    score += 15
                elif price_change_30d > 0:
                    score += 10
                elif price_change_30d > -5:
                    score += 5
        
        return round(score, 2)
    
    def calculate_fundamental_score(self):
        """Calculate fundamental analysis score (0-100)"""
        score = 0
        fundamentals = self.get_fundamental_data()
        
        pe = fundamentals.get('pe_ratio', 0)
        if 0 < pe < 15:
            score += 20
        elif 15 <= pe < 25:
            score += 15
        elif 25 <= pe < 35:
            score += 10
        
        div_yield = fundamentals.get('dividend_yield', 0)
        if div_yield > 3:
            score += 15
        elif div_yield > 1.5:
            score += 10
        elif div_yield > 0.5:
            score += 5
        
        roe = fundamentals.get('roe', 0)
        if roe > 20:
            score += 20
        elif roe > 15:
            score += 15
        elif roe > 10:
            score += 10
        
        profit_margin = fundamentals.get('profit_margin', 0)
        if profit_margin > 20:
            score += 15
        elif profit_margin > 10:
            score += 10
        elif profit_margin > 5:
            score += 5
        
        revenue_growth = fundamentals.get('revenue_growth', 0)
        if revenue_growth > 20:
            score += 15
        elif revenue_growth > 10:
            score += 10
        elif revenue_growth > 0:
            score += 5
        
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        if 0 <= debt_to_equity < 0.5:
            score += 15
        elif 0.5 <= debt_to_equity < 1:
            score += 10
        elif 1 <= debt_to_equity < 2:
            score += 5
        
        return round(score, 2)
    
    def get_recommendation(self):
        """Generate buy/hold/sell recommendation"""
        tech_score = self.calculate_technical_score()
        fund_score = self.calculate_fundamental_score()
        combined_score = (tech_score * 0.5 + fund_score * 0.5)
        
        if combined_score >= 70:
            return "STRONG BUY", combined_score, tech_score, fund_score
        elif combined_score >= 55:
            return "BUY", combined_score, tech_score, fund_score
        elif combined_score >= 40:
            return "HOLD", combined_score, tech_score, fund_score
        elif combined_score >= 25:
            return "REDUCE", combined_score, tech_score, fund_score
        else:
            return "SELL", combined_score, tech_score, fund_score


def analyze_portfolio(stocks):
    """Analyze entire portfolio"""
    results = []
    successful = 0
    failed = 0
    
    print(f"\n{'='*70}")
    print(f"Starting analysis of {len(stocks)} stocks...")
    print(f"{'='*70}\n")
    
    for idx, ticker in enumerate(stocks, 1):
        print(f"[{idx}/{len(stocks)}] Analyzing {ticker}...")
        analyzer = StockAnalyzer(ticker)
        
        if analyzer.fetch_data():
            try:
                recommendation, combined_score, tech_score, fund_score = analyzer.get_recommendation()
                
                current, sma_20, sma_50, sma_200 = analyzer.calculate_moving_averages()
                rsi = analyzer.calculate_rsi()
                macd, signal, histogram = analyzer.calculate_macd()
                fundamentals = analyzer.get_fundamental_data()
                
                results.append({
                    'ticker': ticker,
                    'recommendation': recommendation,
                    'combined_score': combined_score,
                    'technical_score': tech_score,
                    'fundamental_score': fund_score,
                    'current_price': current,
                    'rsi': rsi,
                    'macd': macd,
                    'signal': signal,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'sma_200': sma_200,
                    'pe_ratio': fundamentals.get('pe_ratio', 0),
                    'dividend_yield': fundamentals.get('dividend_yield', 0),
                    'market_cap': fundamentals.get('market_cap', 0),
                    'roe': fundamentals.get('roe', 0),
                    'profit_margin': fundamentals.get('profit_margin', 0),
                    'revenue_growth': fundamentals.get('revenue_growth', 0),
                    '52w_high': fundamentals.get('52w_high', 0),
                    '52w_low': fundamentals.get('52w_low', 0),
                })
                successful += 1
                print(f"  ✅ Analysis complete: {recommendation} (Score: {combined_score:.1f})")
            except Exception as e:
                print(f"  ❌ Error analyzing {ticker}: {e}")
                results.append({
                    'ticker': ticker,
                    'recommendation': 'DATA ERROR',
                    'combined_score': 0,
                    'technical_score': 0,
                    'fundamental_score': 0,
                    'current_price': 0,
                })
                failed += 1
        else:
            results.append({
                'ticker': ticker,
                'recommendation': 'DATA ERROR',
                'combined_score': 0,
                'technical_score': 0,
                'fundamental_score': 0,
                'current_price': 0,
            })
            failed += 1
        
        if idx < len(stocks):
            time.sleep(1)
    
    print(f"\n{'='*70}")
    print(f"Analysis Summary:")
    print(f"  ✅ Successful: {successful}/{len(stocks)}")
    print(f"  ❌ Failed: {failed}/{len(stocks)}")
    print(f"{'='*70}\n")
    
    if successful == 0:
        print("ERROR: No stocks were successfully analyzed. Exiting.")
        sys.exit(1)
    
    return results


def calculate_allocation(results, monthly_investment):
    """Calculate investment allocation based on scores"""
    valid_stocks = [r for r in results if r['combined_score'] >= 40]
    
    if not valid_stocks:
        equal_allocation = monthly_investment / len(results)
        for result in results:
            result['allocation_amount'] = equal_allocation
            result['allocation_percent'] = (equal_allocation / monthly_investment) * 100
        return results
    
    total_score = sum(r['combined_score'] for r in valid_stocks)
    
    for result in results:
        if result['combined_score'] >= 40:
            weight = result['combined_score'] / total_score
            result['allocation_amount'] = monthly_investment * weight
            result['allocation_percent'] = weight * 100
        else:
            result['allocation_amount'] = 0
            result['allocation_percent'] = 0
    
    return results


def generate_github_pages_html(results, monthly_investment, github_url=""):
    """Generate enhanced index.html for GitHub Pages with IST timezone"""
    
    ist_time = get_ist_time()
    total_allocation = sum(r['allocation_amount'] for r in results)
    strong_buy = len([r for r in results if r['recommendation'] == 'STRONG BUY'])
    buy = len([r for r in results if r['recommendation'] == 'BUY'])
    hold = len([r for r in results if r['recommendation'] == 'HOLD'])
    reduce = len([r for r in results if r['recommendation'] == 'REDUCE'])
    sell = len([r for r in results if r['recommendation'] == 'SELL'])
    
    # Calculate portfolio health metrics
    avg_score = sum(r['combined_score'] for r in results)/len(results)
    top_performers = [r for r in results if r['combined_score'] >= 70]
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Analysis - {ist_time.strftime('%d %B %Y')}</title>
    <meta name="description" content="Comprehensive Stock Portfolio Analysis with Technical and Fundamental Indicators">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        /* Enhanced Header with IST Time */
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 50px 40px 40px 40px;
            text-align: center;
            position: relative;
        }}
        
        .header h1 {{
            font-size: 3em;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 1.3em;
            opacity: 0.95;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .ist-time-badge {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: rgba(255,255,255,0.2);
            padding: 12px 25px;
            border-radius: 30px;
            margin-top: 20px;
            font-size: 1.1em;
            font-weight: 500;
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255,255,255,0.3);
        }}
        
        .ist-time-badge .clock-icon {{
            font-size: 1.3em;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
        }}
        
        /* Quick Stats Dashboard */
        .quick-stats {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 40px;
            border-bottom: 3px solid #dee2e6;
        }}
        
        .stats-title {{
            text-align: center;
            font-size: 1.8em;
            color: #1e3c72;
            margin-bottom: 30px;
            font-weight: 700;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 30px 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: all 0.3s ease;
            border-left: 5px solid #1e3c72;
        }}
        
        .stat-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.15);
        }}
        
        .stat-card.highlight {{
            border-left-color: #00a86b;
            background: linear-gradient(135deg, #ffffff 0%, #f0fff4 100%);
        }}
        
        .stat-card .icon {{
            font-size: 2.5em;
            margin-bottom: 12px;
            opacity: 0.8;
        }}
        
        .stat-card h3 {{
            color: #6c757d;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 12px;
            font-weight: 600;
        }}
        
        .stat-card .value {{
            font-size: 2.3em;
            font-weight: 700;
            color: #1e3c72;
            line-height: 1.2;
        }}
        
        .stat-card .subvalue {{
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 8px;
        }}
        
        /* Portfolio Health Indicator */
        .health-indicator {{
            text-align: center;
            padding: 25px;
            background: white;
            border-radius: 12px;
            margin-top: 25px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        }}
        
        .health-title {{
            font-size: 1.1em;
            color: #6c757d;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .health-bar {{
            width: 100%;
            height: 35px;
            background: #e9ecef;
            border-radius: 20px;
            overflow: hidden;
            position: relative;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .health-fill {{
            height: 100%;
            background: linear-gradient(90deg, #00a86b 0%, #4caf50 50%, #8bc34a 100%);
            border-radius: 20px;
            transition: width 1.5s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 15px;
            color: white;
            font-weight: 700;
            font-size: 1.1em;
        }}
        
        /* Recommendation Breakdown */
        .recommendation-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            padding: 35px 40px;
            background: white;
            border-bottom: 2px solid #eee;
        }}
        
        .rec-item {{
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            transition: all 0.3s ease;
        }}
        
        .rec-item:hover {{
            transform: scale(1.05);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }}
        
        .rec-item .count {{
            font-size: 3em;
            font-weight: 800;
            margin-bottom: 8px;
            line-height: 1;
        }}
        
        .rec-item .label {{
            font-size: 1em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .rec-item.strong-buy {{
            background: linear-gradient(135deg, #00a86b 0%, #00d084 100%);
            color: white;
        }}
        
        .rec-item.buy {{
            background: linear-gradient(135deg, #4caf50 0%, #8bc34a 100%);
            color: white;
        }}
        
        .rec-item.hold {{
            background: linear-gradient(135deg, #ff9800 0%, #ffb74d 100%);
            color: white;
        }}
        
        .rec-item.reduce {{
            background: linear-gradient(135deg, #ff5722 0%, #ff7043 100%);
            color: white;
        }}
        
        .rec-item.sell {{
            background: linear-gradient(135deg, #f44336 0%, #e57373 100%);
            color: white;
        }}
        
        /* Enhanced Table */
        .table-container {{
            padding: 40px;
            overflow-x: auto;
        }}
        
        .table-title {{
            font-size: 1.8em;
            color: #1e3c72;
            margin-bottom: 25px;
            font-weight: 700;
            text-align: center;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        thead {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
        }}
        
        th {{
            padding: 18px 15px;
            text-align: left;
            font-weight: 600;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        
        td {{
            padding: 18px 15px;
            border-bottom: 1px solid #f1f3f5;
        }}
        
        tbody tr {{
            transition: background 0.2s ease;
        }}
        
        tbody tr:hover {{
            background: #f8f9fa;
        }}
        
        .ticker {{
            font-weight: 700;
            color: #1e3c72;
            font-size: 1.15em;
        }}
        
        .recommendation {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .recommendation.strong-buy {{
            background: #00a86b;
            color: white;
        }}
        
        .recommendation.buy {{
            background: #4caf50;
            color: white;
        }}
        
        .recommendation.hold {{
            background: #ff9800;
            color: white;
        }}
        
        .recommendation.reduce {{
            background: #ff5722;
            color: white;
        }}
        
        .recommendation.sell {{
            background: #f44336;
            color: white;
        }}
        
        .score {{
            font-weight: 700;
            font-size: 1.15em;
        }}
        
        .score.excellent {{ color: #00a86b; }}
        .score.good {{ color: #4caf50; }}
        .score.average {{ color: #ff9800; }}
        .score.poor {{ color: #f44336; }}
        
        .allocation {{
            font-weight: 700;
            color: #1e3c72;
            font-size: 1.05em;
        }}
        
        .technical-indicators {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            font-size: 0.88em;
        }}
        
        .fundamental-data {{
            font-size: 0.88em;
            color: #495057;
        }}
        
        .disclaimer {{
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 25px;
            margin: 30px 40px;
            border-radius: 8px;
        }}
        
        .disclaimer h4 {{
            color: #856404;
            margin-bottom: 12px;
            font-size: 1.2em;
        }}
        
        .disclaimer p {{
            color: #856404;
            line-height: 1.8;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 35px 40px;
            text-align: center;
            color: #6c757d;
            font-size: 0.95em;
            border-top: 3px solid #dee2e6;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 2em;
            }}
            
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            table {{
                font-size: 0.85em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Enhanced Header -->
        <div class="header">
            <h1>📊 Portfolio Analysis Dashboard</h1>
            <div class="subtitle">Technical & Fundamental Analysis Report</div>
            <div class="ist-time-badge">
                <span class="clock-icon">🕐</span>
                <span>Generated: {ist_time.strftime('%d %B %Y')} at {ist_time.strftime('%I:%M %p IST')}</span>
            </div>
        </div>
        
        <!-- Quick Stats Dashboard -->
        <div class="quick-stats">
            <h2 class="stats-title">📈 Portfolio Overview</h2>
            
            <div class="stats-grid">
                <div class="stat-card highlight">
                    <div class="icon">💰</div>
                    <h3>Monthly Investment</h3>
                    <div class="value">₹{monthly_investment:,}</div>
                    <div class="subvalue">Allocated: ₹{total_allocation:,.0f}</div>
                </div>
                
                <div class="stat-card">
                    <div class="icon">📊</div>
                    <h3>Total Stocks</h3>
                    <div class="value">{len(results)}</div>
                    <div class="subvalue">{len(top_performers)} top performers</div>
                </div>
                
                <div class="stat-card">
                    <div class="icon">⭐</div>
                    <h3>Avg Score</h3>
                    <div class="value">{avg_score:.1f}</div>
                    <div class="subvalue">Portfolio Health</div>
                </div>
                
                <div class="stat-card">
                    <div class="icon">🎯</div>
                    <h3>Strong Buys</h3>
                    <div class="value">{strong_buy}</div>
                    <div class="subvalue">Best opportunities</div>
                </div>
            </div>
            
            <!-- Portfolio Health Bar -->
            <div class="health-indicator">
                <div class="health-title">Overall Portfolio Health Score</div>
                <div class="health-bar">
                    <div class="health-fill" style="width: {avg_score}%;">
                        {avg_score:.1f}%
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Recommendation Breakdown -->
        <div class="recommendation-summary">
            <div class="rec-item strong-buy">
                <div class="count">{strong_buy}</div>
                <div class="label">Strong Buy</div>
            </div>
            <div class="rec-item buy">
                <div class="count">{buy}</div>
                <div class="label">Buy</div>
            </div>
            <div class="rec-item hold">
                <div class="count">{hold}</div>
                <div class="label">Hold</div>
            </div>
            <div class="rec-item reduce">
                <div class="count">{reduce}</div>
                <div class="label">Reduce</div>
            </div>
            <div class="rec-item sell">
                <div class="count">{sell}</div>
                <div class="label">Sell</div>
            </div>
        </div>
        
        <!-- Detailed Table -->
        <div class="table-container">
            <h2 class="table-title">📋 Detailed Stock Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Stock</th>
                        <th>Recommendation</th>
                        <th>Combined Score</th>
                        <th>Tech Score</th>
                        <th>Fund Score</th>
                        <th>Allocation %</th>
                        <th>Amount (₹)</th>
                        <th>Current Price</th>
                        <th>Technical Indicators</th>
                        <th>Fundamentals</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    results_sorted = sorted(results, key=lambda x: x['combined_score'], reverse=True)
    
    for r in results_sorted:
        combined_score = r.get('combined_score', 0) or 0
        tech_score = r.get('technical_score', 0) or 0
        fund_score = r.get('fundamental_score', 0) or 0
        alloc_pct = r.get('allocation_percent', 0) or 0
        alloc_amt = r.get('allocation_amount', 0) or 0
        current_price = r.get('current_price', 0) or 0
        rsi = r.get('rsi', 0) or 0
        macd = r.get('macd', 0) or 0
        sma_20 = r.get('sma_20', 0) or 0
        pe_ratio = r.get('pe_ratio', 0) or 0
        roe = r.get('roe', 0) or 0
        div_yield = r.get('dividend_yield', 0) or 0
        
        score_class = 'excellent' if combined_score >= 70 else 'good' if combined_score >= 55 else 'average' if combined_score >= 40 else 'poor'
        rec_class = r['recommendation'].lower().replace(' ', '-')
        
        tech_info = f"""
        <div class="technical-indicators">
            <div>RSI: {rsi:.1f}</div>
            <div>MACD: {macd:.2f}</div>
            <div>SMA20: ₹{sma_20:.2f}</div>
        </div>
        """
        
        fund_info = f"""
        <div class="fundamental-data">
            <div>P/E: {pe_ratio:.2f}</div>
            <div>ROE: {roe:.1f}%</div>
            <div>Div Yield: {div_yield:.2f}%</div>
        </div>
        """
        
        html += f"""
                    <tr>
                        <td class="ticker">{r['ticker']}</td>
                        <td><span class="recommendation {rec_class}">{r['recommendation']}</span></td>
                        <td><span class="score {score_class}">{combined_score:.1f}</span></td>
                        <td>{tech_score:.1f}</td>
                        <td>{fund_score:.1f}</td>
                        <td>{alloc_pct:.1f}%</td>
                        <td class="allocation">₹{alloc_amt:,.0f}</td>
                        <td>₹{current_price:.2f}</td>
                        <td>{tech_info}</td>
                        <td>{fund_info}</td>
                    </tr>
        """
    
    html += f"""
                </tbody>
            </table>
        </div>
        
        <div class="disclaimer">
            <h4>⚠️ Important Disclaimer</h4>
            <p>
                This analysis is for informational purposes only and should not be considered as financial advice. 
                The recommendations are based on technical and fundamental analysis algorithms and do not guarantee 
                future performance. Always conduct your own research and consult with a qualified financial advisor 
                before making investment decisions. Past performance is not indicative of future results.
            </p>
        </div>
        
        <div class="footer">
            <p><strong>Analysis Methodology:</strong></p>
            <p>Technical Analysis (50%): RSI, MACD, Moving Averages, Bollinger Bands, Price Momentum</p>
            <p>Fundamental Analysis (50%): P/E Ratio, ROE, Profit Margin, Revenue Growth, Dividend Yield, Debt/Equity</p>
            <p style="margin-top: 20px; font-weight: 600;">Generated by Portfolio Analyzer v2.0 Enhanced | © 2026</p>
            <p style="margin-top: 10px; font-size: 0.85em; opacity: 0.8;">
                Last Updated: {ist_time.strftime('%d %B %Y')} at {ist_time.strftime('%I:%M %p IST')}
            </p>
        </div>
    </div>
</body>
</html>
    """
    
    return html


def generate_email_html(results, monthly_investment, github_url=""):
    """Generate HTML for email (same enhanced format with IST time)"""
    return generate_github_pages_html(results, monthly_investment, github_url)


def send_email(html_content, recipient_email, sender_email, sender_password):
    """Send HTML report via email"""
    try:
        ist_time = get_ist_time()
        print(f"\n📧 Preparing to send email to {recipient_email}...")
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"📊 Portfolio Analysis Report - {ist_time.strftime('%d %B %Y, %I:%M %p IST')}"
        msg['From'] = sender_email
        msg['To'] = recipient_email
        
        text_content = f"""
Portfolio Analysis Report - {ist_time.strftime('%d %B %Y')} at {ist_time.strftime('%I:%M %p IST')}

This is an HTML email. Please view it in an email client that supports HTML to see the full report.

Generated by Portfolio Analyzer v2.0 Enhanced
        """
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_content, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        print("  Connecting to Gmail SMTP server...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            print("  Logging in...")
            server.login(sender_email, sender_password)
            print("  Sending email...")
            server.send_message(msg)
        
        print(f"✅ Email sent successfully to {recipient_email}")
        print(f"   Check your inbox for the portfolio analysis report!")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print(f"❌ Authentication failed. Please check your email and password.")
        print(f"   For Gmail, you need to use an 'App Password' instead of your regular password.")
        print(f"   Generate one at: https://myaccount.google.com/apppasswords")
        return False
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False


def main():
    """Main execution function"""
    ist_time = get_ist_time()
    
    print("=" * 80)
    print("Portfolio Stock Analyzer v2.0 - Enhanced Edition")
    print("Technical & Fundamental Analysis with IST Timezone")
    print("=" * 80)
    print(f"Current IST Time: {ist_time.strftime('%d %B %Y, %I:%M %p IST')}")
    print()
    
    results = analyze_portfolio(PORTFOLIO_STOCKS)
    results = calculate_allocation(results, MONTHLY_INVESTMENT)
    
    github_url = f"https://{GITHUB_USERNAME}.github.io/{GITHUB_REPO}/"
    print(f"\n📌 GitHub Pages URL: {github_url}")
    
    # FIXED: Use fixed filenames that will overwrite
    github_pages_html = generate_github_pages_html(results, MONTHLY_INVESTMENT, github_url)
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(github_pages_html)
    print(f"✅ GitHub Pages HTML generated: index.html (overwrites existing)")
    
    email_html = generate_email_html(results, MONTHLY_INVESTMENT, github_url)
    
    # FIXED: Use fixed filename instead of timestamped
    report_filename = "portfolio_analysis.html"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(email_html)
    print(f"✅ Email HTML report generated: {report_filename} (overwrites existing)")
    
    json_data = {
        'generated_date_ist': ist_time.isoformat(),
        'generated_date_display': ist_time.strftime('%d %B %Y, %I:%M %p IST'),
        'monthly_investment': MONTHLY_INVESTMENT,
        'github_pages_url': github_url,
        'stocks': results
    }
    
    # FIXED: Use fixed filename instead of timestamped
    json_filename = "portfolio_data.json"
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✅ JSON data generated: {json_filename} (overwrites existing)")
    
    # Email handling (same logic as original)
    sender_email = os.environ.get('SENDER_EMAIL')
    sender_password = os.environ.get('SENDER_PASSWORD')
    recipient_email = os.environ.get('RECIPIENT_EMAIL')
    
    email_enabled = False
    
    if sender_email and sender_password and recipient_email:
        email_enabled = True
        print("\n" + "=" * 80)
        print("EMAIL DELIVERY: ENABLED (via environment variables)")
        print("=" * 80)
    elif EMAIL_CONFIG['enabled']:
        email_enabled = True
        sender_email = EMAIL_CONFIG['sender_email']
        sender_password = EMAIL_CONFIG['sender_password']
        recipient_email = EMAIL_CONFIG['recipient_email']
        print("\n" + "=" * 80)
        print("EMAIL DELIVERY: ENABLED (via EMAIL_CONFIG)")
        print("=" * 80)
    
    if email_enabled:
        send_email(email_html, recipient_email, sender_email, sender_password)
    else:
        print("\n" + "=" * 80)
        print("EMAIL DELIVERY: DISABLED")
        print("=" * 80)
    
    # Display enhanced summary
    print("\n" + "=" * 80)
    print("📊 INVESTMENT RECOMMENDATIONS SUMMARY")
    print("=" * 80)
    print(f"Generated: {ist_time.strftime('%d %B %Y at %I:%M %p IST')}\n")
    
    for r in sorted(results, key=lambda x: x['combined_score'], reverse=True):
        combined = r.get('combined_score', 0) or 0
        alloc_amt = r.get('allocation_amount', 0) or 0
        alloc_pct = r.get('allocation_percent', 0) or 0
        
        print(f"{r['ticker']:15} | {r['recommendation']:12} | Score: {combined:5.1f} | ₹{alloc_amt:>10,.0f} ({alloc_pct:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ Files Generated (will overwrite on next run):")
    print(f"   - index.html")
    print(f"   - portfolio_analysis.html")
    print(f"   - portfolio_data.json")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    try:
        main()
        
        if os.path.exists('index.html'):
            print("\n✅ Portfolio analysis completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️ Analysis ran but index.html not found!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
