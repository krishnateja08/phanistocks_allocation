"""
Stock Portfolio Analysis Script with Technical & Fundamental Analysis
Author: Portfolio Manager
Date: February 2026
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import warnings
import time
import sys
warnings.filterwarnings('ignore')

# Portfolio Configuration
MONTHLY_INVESTMENT = 60000  # INR
PORTFOLIO_STOCKS = [
    "BANKBEES.NS", "HDFCBANK.NS", "NXST-RR.NS", "EMBASSY-RR.NS",
    "BIRET-RR.NS", "MINDSPACE-RR.NS", "ITBEES.NS", "TECHM.NS",
    "TCS.NS", "INFY.NS", "RELIANCE.NS", "TMPV.NS",
    "NIFTYBEES.NS", "M&M.NS", "GOLDBEES.NS", "SILVERBEES.NS"
]

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
                        time.sleep(2 ** attempt)  # Exponential backoff
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
            return None
        
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def calculate_macd(self):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if self.data is None or len(self.data) < 26:
            return None, None, None
        
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return macd.iloc[-1], signal.iloc[-1], histogram.iloc[-1]
    
    def calculate_moving_averages(self):
        """Calculate Simple Moving Averages"""
        if self.data is None:
            return None, None, None
        
        current_price = self.data['Close'].iloc[-1]
        sma_20 = self.data['Close'].rolling(window=20).mean().iloc[-1] if len(self.data) >= 20 else None
        sma_50 = self.data['Close'].rolling(window=50).mean().iloc[-1] if len(self.data) >= 50 else None
        sma_200 = self.data['Close'].rolling(window=200).mean().iloc[-1] if len(self.data) >= 200 else None
        
        return current_price, sma_20, sma_50, sma_200
    
    def calculate_bollinger_bands(self, period=20):
        """Calculate Bollinger Bands"""
        if self.data is None or len(self.data) < period:
            return None, None, None
        
        sma = self.data['Close'].rolling(window=period).mean()
        std = self.data['Close'].rolling(window=period).std()
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]
    
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
        
        # RSI Score (20 points)
        rsi = self.calculate_rsi()
        if rsi:
            if 30 <= rsi <= 70:
                score += 20
            elif 70 < rsi <= 80:
                score += 10
            elif 20 <= rsi < 30:
                score += 15
        
        # MACD Score (20 points)
        macd, signal, histogram = self.calculate_macd()
        if macd and signal:
            if histogram > 0:
                score += 20
            elif histogram > -0.5:
                score += 10
        
        # Moving Average Score (30 points)
        current, sma_20, sma_50, sma_200 = self.calculate_moving_averages()
        if current and sma_20:
            if current > sma_20:
                score += 10
        if current and sma_50:
            if current > sma_50:
                score += 10
        if current and sma_200:
            if current > sma_200:
                score += 10
        
        # Bollinger Bands Score (15 points)
        upper, middle, lower = self.calculate_bollinger_bands()
        if current and lower and upper:
            band_position = (current - lower) / (upper - lower)
            if 0.2 <= band_position <= 0.5:
                score += 15
            elif 0.5 < band_position <= 0.8:
                score += 10
        
        # Price Momentum (15 points)
        if len(self.data) >= 30:
            price_change_30d = ((self.data['Close'].iloc[-1] - self.data['Close'].iloc[-30]) / 
                               self.data['Close'].iloc[-30] * 100)
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
        
        # P/E Ratio Score (20 points)
        pe = fundamentals.get('pe_ratio', 0)
        if 0 < pe < 15:
            score += 20
        elif 15 <= pe < 25:
            score += 15
        elif 25 <= pe < 35:
            score += 10
        
        # Dividend Yield Score (15 points)
        div_yield = fundamentals.get('dividend_yield', 0)
        if div_yield > 3:
            score += 15
        elif div_yield > 1.5:
            score += 10
        elif div_yield > 0.5:
            score += 5
        
        # ROE Score (20 points)
        roe = fundamentals.get('roe', 0)
        if roe > 20:
            score += 20
        elif roe > 15:
            score += 15
        elif roe > 10:
            score += 10
        
        # Profit Margin Score (15 points)
        profit_margin = fundamentals.get('profit_margin', 0)
        if profit_margin > 20:
            score += 15
        elif profit_margin > 10:
            score += 10
        elif profit_margin > 5:
            score += 5
        
        # Revenue Growth Score (15 points)
        revenue_growth = fundamentals.get('revenue_growth', 0)
        if revenue_growth > 20:
            score += 15
        elif revenue_growth > 10:
            score += 10
        elif revenue_growth > 0:
            score += 5
        
        # Debt to Equity Score (15 points)
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
        
        # Small delay to avoid rate limiting
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
    # Filter out data errors and poor performers
    valid_stocks = [r for r in results if r['combined_score'] >= 40]
    
    if not valid_stocks:
        # If no good stocks, distribute equally among all
        equal_allocation = monthly_investment / len(results)
        for result in results:
            result['allocation_amount'] = equal_allocation
            result['allocation_percent'] = (equal_allocation / monthly_investment) * 100
        return results
    
    # Calculate weighted allocation based on combined score
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


def generate_html_report(results, monthly_investment):
    """Generate professional HTML report"""
    
    # Calculate totals
    total_allocation = sum(r['allocation_amount'] for r in results)
    strong_buy = len([r for r in results if r['recommendation'] == 'STRONG BUY'])
    buy = len([r for r in results if r['recommendation'] == 'BUY'])
    hold = len([r for r in results if r['recommendation'] == 'HOLD'])
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Portfolio Analysis Report - {datetime.now().strftime('%B %d, %Y')}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            
            .header .date {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                padding: 40px;
                background: #f8f9fa;
            }}
            
            .summary-card {{
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.3s ease;
            }}
            
            .summary-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            }}
            
            .summary-card h3 {{
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 10px;
            }}
            
            .summary-card .value {{
                font-size: 2em;
                font-weight: bold;
                color: #1e3c72;
            }}
            
            .recommendation-summary {{
                display: flex;
                justify-content: space-around;
                padding: 30px 40px;
                background: white;
                border-bottom: 2px solid #eee;
            }}
            
            .rec-item {{
                text-align: center;
            }}
            
            .rec-item .count {{
                font-size: 2em;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            
            .rec-item.strong-buy .count {{ color: #00a86b; }}
            .rec-item.buy .count {{ color: #4caf50; }}
            .rec-item.hold .count {{ color: #ff9800; }}
            
            .table-container {{
                padding: 40px;
                overflow-x: auto;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            thead {{
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
            }}
            
            th {{
                padding: 15px;
                text-align: left;
                font-weight: 600;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            td {{
                padding: 15px;
                border-bottom: 1px solid #eee;
            }}
            
            tbody tr:hover {{
                background: #f8f9fa;
            }}
            
            .ticker {{
                font-weight: bold;
                color: #1e3c72;
                font-size: 1.1em;
            }}
            
            .recommendation {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 0.85em;
                text-transform: uppercase;
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
                font-weight: bold;
                font-size: 1.1em;
            }}
            
            .score.excellent {{ color: #00a86b; }}
            .score.good {{ color: #4caf50; }}
            .score.average {{ color: #ff9800; }}
            .score.poor {{ color: #f44336; }}
            
            .allocation {{
                font-weight: bold;
                color: #1e3c72;
            }}
            
            .footer {{
                background: #f8f9fa;
                padding: 30px 40px;
                text-align: center;
                color: #666;
                font-size: 0.9em;
                border-top: 2px solid #eee;
            }}
            
            .disclaimer {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 20px;
                margin: 20px 40px;
                border-radius: 5px;
            }}
            
            .disclaimer h4 {{
                color: #856404;
                margin-bottom: 10px;
            }}
            
            .disclaimer p {{
                color: #856404;
                line-height: 1.6;
            }}
            
            .technical-indicators {{
                background: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                font-size: 0.85em;
            }}
            
            .fundamental-data {{
                font-size: 0.85em;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Portfolio Analysis Report</h1>
                <div class="date">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
            </div>
            
            <div class="summary">
                <div class="summary-card">
                    <h3>Monthly Investment</h3>
                    <div class="value">₹{monthly_investment:,.0f}</div>
                </div>
                <div class="summary-card">
                    <h3>Total Stocks</h3>
                    <div class="value">{len(results)}</div>
                </div>
                <div class="summary-card">
                    <h3>Avg Combined Score</h3>
                    <div class="value">{sum(r['combined_score'] for r in results)/len(results):.1f}</div>
                </div>
                <div class="summary-card">
                    <h3>Allocated Amount</h3>
                    <div class="value">₹{total_allocation:,.0f}</div>
                </div>
            </div>
            
            <div class="recommendation-summary">
                <div class="rec-item strong-buy">
                    <div class="count">{strong_buy}</div>
                    <div>Strong Buy</div>
                </div>
                <div class="rec-item buy">
                    <div class="count">{buy}</div>
                    <div>Buy</div>
                </div>
                <div class="rec-item hold">
                    <div class="count">{hold}</div>
                    <div>Hold</div>
                </div>
            </div>
            
            <div class="table-container">
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
    
    # Sort by combined score descending
    results_sorted = sorted(results, key=lambda x: x['combined_score'], reverse=True)
    
    for r in results_sorted:
        score_class = 'excellent' if r['combined_score'] >= 70 else 'good' if r['combined_score'] >= 55 else 'average' if r['combined_score'] >= 40 else 'poor'
        rec_class = r['recommendation'].lower().replace(' ', '-')
        
        # Technical indicators
        tech_info = f"""
        <div class="technical-indicators">
            <div>RSI: {r.get('rsi', 0):.1f}</div>
            <div>MACD: {r.get('macd', 0):.2f}</div>
            <div>SMA20: ₹{r.get('sma_20', 0):.2f}</div>
        </div>
        """
        
        # Fundamental data
        fund_info = f"""
        <div class="fundamental-data">
            <div>P/E: {r.get('pe_ratio', 0):.2f}</div>
            <div>ROE: {r.get('roe', 0):.1f}%</div>
            <div>Div Yield: {r.get('dividend_yield', 0):.2f}%</div>
        </div>
        """
        
        html += f"""
                        <tr>
                            <td class="ticker">{r['ticker']}</td>
                            <td><span class="recommendation {rec_class}">{r['recommendation']}</span></td>
                            <td><span class="score {score_class}">{r['combined_score']:.1f}</span></td>
                            <td>{r['technical_score']:.1f}</td>
                            <td>{r['fundamental_score']:.1f}</td>
                            <td>{r.get('allocation_percent', 0):.1f}%</td>
                            <td class="allocation">₹{r.get('allocation_amount', 0):,.0f}</td>
                            <td>₹{r.get('current_price', 0):.2f}</td>
                            <td>{tech_info}</td>
                            <td>{fund_info}</td>
                        </tr>
        """
    
    html += """
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
                <p style="margin-top: 15px;">Generated by Portfolio Analyzer v1.0 | © 2026</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def send_email(html_content, recipient_email, sender_email, sender_password):
    """Send HTML report via email"""
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Portfolio Analysis Report - {datetime.now().strftime('%B %d, %Y')}"
        msg['From'] = sender_email
        msg['To'] = recipient_email
        
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Using Gmail SMTP
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        print(f"✅ Email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False


def main():
    """Main execution function"""
    print("=" * 60)
    print("Portfolio Stock Analyzer - Technical & Fundamental Analysis")
    print("=" * 60)
    print()
    
    # Analyze portfolio
    results = analyze_portfolio(PORTFOLIO_STOCKS)
    
    # Calculate allocation
    results = calculate_allocation(results, MONTHLY_INVESTMENT)
    
    # Generate HTML report
    html_report = generate_html_report(results, MONTHLY_INVESTMENT)
    
    # Save HTML report
    report_filename = f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.html"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"\n✅ HTML report generated: {report_filename}")
    
    # Save JSON data for GitHub Pages
    json_data = {
        'generated_date': datetime.now().isoformat(),
        'monthly_investment': MONTHLY_INVESTMENT,
        'stocks': results
    }
    
    json_filename = f"portfolio_data_{datetime.now().strftime('%Y%m%d')}.json"
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✅ JSON data generated: {json_filename}")
    
    # Display summary
    print("\n" + "=" * 60)
    print("INVESTMENT RECOMMENDATIONS SUMMARY")
    print("=" * 60)
    
    for r in sorted(results, key=lambda x: x['combined_score'], reverse=True):
        print(f"\n{r['ticker']:15} | {r['recommendation']:12} | Score: {r['combined_score']:5.1f} | Allocation: ₹{r.get('allocation_amount', 0):,.0f} ({r.get('allocation_percent', 0):.1f}%)")
        print(f"                 Tech: {r['technical_score']:.1f} | Fund: {r['fundamental_score']:.1f} | Price: ₹{r.get('current_price', 0):.2f}")
    
    print("\n" + "=" * 60)
    
    # Email configuration (uncomment and configure to send email)
    # RECIPIENT_EMAIL = "your.email@example.com"
    # SENDER_EMAIL = "sender.email@gmail.com"
    # SENDER_PASSWORD = "your-app-specific-password"
    # send_email(html_report, RECIPIENT_EMAIL, SENDER_EMAIL, SENDER_PASSWORD)
    
    return results


if __name__ == "__main__":
    try:
        main()
        
        # Verify files were created
        import os
        from glob import glob
        
        html_files = glob("portfolio_analysis_*.html")
        json_files = glob("portfolio_data_*.json")
        
        if html_files and json_files:
            print("\n✅ Portfolio analysis completed successfully!")
            print(f"   Generated: {html_files[0]}")
            print(f"   Generated: {json_files[0]}")
            sys.exit(0)  # Explicitly exit with success code
        else:
            print("\n⚠️ Analysis ran but files not found!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # Exit with error code
