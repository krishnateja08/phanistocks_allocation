import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

# Portfolio Configuration
PORTFOLIO = {
    "BANKBEES.NS": 586.86,
    "HDFCBANK.NS": 835.26,
    "NXST-RR.NS": 162.52,
    "EMBASSY-RR.NS": 442.78,
    "BIRET-RR.NS": 334.91,
    "MINDSPACE-RR.NS": 487.32,
    "ITBEES.NS": 40.40,
    "TECHM.NS": 1534.55,
    "TCS.NS": 3177.00,
    "INFY.NS": 1524.19,
    "RELIANCE.NS": 1270.57,
    "TMPV.NS": 365.53,
    "NIFTYBEES.NS": 262.72,
    "M&M.NS": 3174.53,
    "GOLDBEES.NS": 92.91,
    "SILVERBEES.NS": 112.65
}

MONTHLY_BUDGET = 60000
CASH_RESERVE_PERCENT = 20

# Stock Categories
CATEGORIES = {
    "Banking & Finance": ["BANKBEES.NS", "HDFCBANK.NS"],
    "REITs": ["NXST-RR.NS", "EMBASSY-RR.NS", "BIRET-RR.NS", "MINDSPACE-RR.NS"],
    "IT Sector": ["ITBEES.NS", "TECHM.NS", "TCS.NS", "INFY.NS"],
    "Index & Others": ["TMPV.NS", "NIFTYBEES.NS", "M&M.NS", "GOLDBEES.NS", "SILVERBEES.NS", "RELIANCE.NS"]
}

def get_stock_data(symbol):
    """Fetch current stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1y")
        
        current_price = info.get('currentPrice', hist['Close'].iloc[-1] if len(hist) > 0 else 0)
        
        # Calculate 52-week high/low
        week_52_high = hist['High'].max() if len(hist) > 0 else current_price
        week_52_low = hist['Low'].min() if len(hist) > 0 else current_price
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'week_52_high': week_52_high,
            'week_52_low': week_52_low,
            'volume': info.get('volume', 0),
            'market_cap': info.get('marketCap', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    if len(prices) < period:
        return 50  # Default neutral value
    
    deltas = prices.diff()
    gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
    loss = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

def calculate_macd(prices):
    """Calculate MACD indicator"""
    if len(prices) < 26:
        return "Neutral"
    
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    if macd.iloc[-1] > signal.iloc[-1]:
        return "Bullish"
    elif macd.iloc[-1] < signal.iloc[-1]:
        return "Bearish"
    else:
        return "Neutral"

def get_technical_indicators(symbol):
    """Calculate technical indicators for a stock"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="3mo")
        
        if len(hist) == 0:
            return {'rsi': 50, 'macd': 'Neutral', 'trend': 'Neutral'}
        
        rsi = calculate_rsi(hist['Close'])
        macd = calculate_macd(hist['Close'])
        
        # Determine trend based on moving averages
        if len(hist) >= 50:
            ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            current = hist['Close'].iloc[-1]
            
            if current > ma_20 > ma_50:
                trend = "Uptrend"
            elif current < ma_20 < ma_50:
                trend = "Downtrend"
            else:
                trend = "Sideways"
        else:
            trend = "Neutral"
        
        return {
            'rsi': round(rsi, 2),
            'macd': macd,
            'trend': trend
        }
    except Exception as e:
        print(f"Error calculating indicators for {symbol}: {e}")
        return {'rsi': 50, 'macd': 'Neutral', 'trend': 'Neutral'}

def generate_recommendation(symbol, avg_price, current_price, rsi, macd, trend):
    """Generate buy/hold/avoid recommendation based on technical analysis"""
    pnl_percent = ((current_price - avg_price) / avg_price) * 100
    
    # Scoring system
    score = 0
    
    # RSI scoring
    if rsi < 30:
        score += 2  # Oversold - strong buy signal
    elif rsi < 40:
        score += 1  # Approaching oversold
    elif rsi > 70:
        score -= 2  # Overbought - avoid
    elif rsi > 60:
        score -= 1  # Approaching overbought
    
    # MACD scoring
    if macd == "Bullish":
        score += 1
    elif macd == "Bearish":
        score -= 1
    
    # Trend scoring
    if trend == "Uptrend":
        score += 1
    elif trend == "Downtrend":
        score -= 1
    
    # Price action scoring
    if current_price < avg_price * 0.95:  # 5% below avg
        score += 1
    elif current_price > avg_price * 1.10:  # 10% above avg
        score -= 1
    
    # Generate recommendation
    if score >= 2:
        return "BUY", "buy"
    elif score <= -2:
        return "AVOID NOW", "avoid"
    else:
        return "HOLD", "hold"

def calculate_allocations(portfolio_data):
    """Calculate investment allocations based on recommendations"""
    total_investable = MONTHLY_BUDGET * (1 - CASH_RESERVE_PERCENT / 100)
    
    # Count BUY recommendations per category
    category_buy_counts = {cat: 0 for cat in CATEGORIES.keys()}
    
    for stock in portfolio_data:
        if stock['recommendation_class'] == 'buy':
            for cat, symbols in CATEGORIES.items():
                if stock['symbol'] in symbols:
                    category_buy_counts[cat] += 1
                    break
    
    # Allocate budget proportionally
    total_buy_stocks = sum(category_buy_counts.values())
    
    if total_buy_stocks == 0:
        return {}
    
    allocations = {}
    per_stock_amount = total_investable / total_buy_stocks
    
    for stock in portfolio_data:
        if stock['recommendation_class'] == 'buy':
            allocations[stock['symbol']] = round(per_stock_amount, 2)
        else:
            allocations[stock['symbol']] = 0
    
    return allocations

def generate_html_report(portfolio_data, allocations, report_time):
    """Generate HTML report"""
    
    total_investable = MONTHLY_BUDGET * (1 - CASH_RESERVE_PERCENT / 100)
    cash_reserve = MONTHLY_BUDGET - total_investable
    
    # Calculate category allocations
    category_allocations = {cat: 0 for cat in CATEGORIES.keys()}
    for stock in portfolio_data:
        for cat, symbols in CATEGORIES.items():
            if stock['symbol'] in symbols:
                category_allocations[cat] += allocations.get(stock['symbol'], 0)
                break
    
    # Calculate total recommended investment
    total_recommended = sum(allocations.values())
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Investment Analysis - {report_time}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        .report-time {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        
        .report-time h2 {{
            font-size: 24px;
            margin-bottom: 8px;
        }}
        
        .report-time p {{
            font-size: 16px;
            opacity: 0.9;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .header h1 {{
            font-size: 36px;
            margin-bottom: 10px;
        }}
        
        .header-info {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            margin-top: 20px;
        }}
        
        .info-box {{
            background: rgba(255,255,255,0.2);
            padding: 15px 25px;
            border-radius: 10px;
            margin: 5px;
        }}
        
        .info-box .label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .info-box .value {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}
        
        .category-section {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        
        .category-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }}
        
        .category-title {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .category-allocation {{
            font-size: 18px;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stocks-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}
        
        .stock-card {{
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
        }}
        
        .stock-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            border-color: #667eea;
        }}
        
        .stock-card.buy {{
            border-left: 5px solid #27ae60;
        }}
        
        .stock-card.hold {{
            border-left: 5px solid #f39c12;
        }}
        
        .stock-card.avoid {{
            border-left: 5px solid #e74c3c;
        }}
        
        .stock-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .stock-name {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .signal-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
        }}
        
        .signal-badge.buy {{
            background: #27ae60;
            color: white;
        }}
        
        .signal-badge.hold {{
            background: #f39c12;
            color: white;
        }}
        
        .signal-badge.avoid {{
            background: #e74c3c;
            color: white;
        }}
        
        .stock-details {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 15px;
        }}
        
        .detail-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        
        .detail-label {{
            color: #666;
            font-size: 13px;
        }}
        
        .detail-value {{
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .price-positive {{
            color: #27ae60;
        }}
        
        .price-negative {{
            color: #e74c3c;
        }}
        
        .technical-signals {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 12px;
        }}
        
        .signals-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 10px;
        }}
        
        .signal-item {{
            text-align: center;
            padding: 8px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }}
        
        .signal-label {{
            font-size: 11px;
            color: #666;
            margin-bottom: 4px;
        }}
        
        .signal-value {{
            font-size: 14px;
            font-weight: bold;
        }}
        
        .allocation-bar {{
            background: #e0e0e0;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .allocation-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.5s ease;
        }}
        
        .recommendation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px;
            border-radius: 6px;
            margin-top: 12px;
        }}
        
        .recommendation-text {{
            font-size: 14px;
            color: #856404;
            line-height: 1.5;
        }}
        
        .summary-box {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .summary-item {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 10px;
        }}
        
        .summary-item .label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .summary-item .value {{
            font-size: 28px;
            font-weight: bold;
            margin-top: 5px;
        }}
        
        @media (max-width: 768px) {{
            .stocks-grid {{
                grid-template-columns: 1fr;
            }}
            
            .signals-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Report Time -->
        <div class="report-time">
            <h2>📅 Portfolio Analysis Report</h2>
            <p>Generated on: {report_time}</p>
        </div>
        
        <!-- Header -->
        <div class="header">
            <h1>📊 Portfolio Investment Analysis</h1>
            <p style="opacity: 0.9; margin-top: 10px;">Based on Technical & Fundamental Analysis</p>
            <div class="header-info">
                <div class="info-box">
                    <div class="label">Monthly Budget</div>
                    <div class="value">₹{MONTHLY_BUDGET:,}</div>
                </div>
                <div class="info-box">
                    <div class="label">Recommended Investment</div>
                    <div class="value">₹{total_recommended:,.2f}</div>
                </div>
                <div class="info-box">
                    <div class="label">Keep as Cash</div>
                    <div class="value">₹{cash_reserve:,.2f} ({CASH_RESERVE_PERCENT}%)</div>
                </div>
            </div>
        </div>
        
        <!-- Investment Summary -->
        <div class="summary-box">
            <h2 style="margin-bottom: 15px;">📈 Investment Allocation Summary</h2>
            <div class="summary-grid">
"""
    
    for category, amount in category_allocations.items():
        if amount > 0:
            percent = (amount / total_recommended * 100) if total_recommended > 0 else 0
            html += f"""
                <div class="summary-item">
                    <div class="label">{category}</div>
                    <div class="value">₹{amount:,.0f}</div>
                    <div style="margin-top: 5px; font-size: 14px;">{percent:.1f}% allocation</div>
                </div>
"""
    
    html += """
            </div>
        </div>
"""
    
    # Generate stock cards by category
    for category, symbols in CATEGORIES.items():
        category_stocks = [s for s in portfolio_data if s['symbol'] in symbols]
        
        if not category_stocks:
            continue
        
        cat_allocation = category_allocations.get(category, 0)
        
        html += f"""
        <!-- {category} -->
        <div class="category-section">
            <div class="category-header">
                <div class="category-title">{category}</div>
                <div class="category-allocation">Allocate: ₹{cat_allocation:,.0f}</div>
            </div>
            <div class="stocks-grid">
"""
        
        for stock in category_stocks:
            allocation_amount = allocations.get(stock['symbol'], 0)
            allocation_percent = (allocation_amount / cat_allocation * 100) if cat_allocation > 0 else 0
            
            rsi_color = "#27ae60" if stock['rsi'] < 40 else ("#e74c3c" if stock['rsi'] > 60 else "#f39c12")
            macd_color = "#27ae60" if stock['macd'] == "Bullish" else ("#e74c3c" if stock['macd'] == "Bearish" else "#f39c12")
            trend_color = "#27ae60" if stock['trend'] == "Uptrend" else ("#e74c3c" if stock['trend'] == "Downtrend" else "#f39c12")
            
            html += f"""
                <div class="stock-card {stock['recommendation_class']}">
                    <div class="stock-header">
                        <div class="stock-name">{stock['display_name']}</div>
                        <div class="signal-badge {stock['recommendation_class']}">{stock['recommendation']}</div>
                    </div>
                    <div class="stock-details">
                        <div class="detail-item">
                            <span class="detail-label">Your Avg Price</span>
                            <span class="detail-value">₹{stock['avg_price']:,.2f}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Current Price</span>
                            <span class="detail-value {'price-positive' if stock['pnl_percent'] >= 0 else 'price-negative'}">₹{stock['current_price']:,.2f}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Your P&L</span>
                            <span class="detail-value {'price-positive' if stock['pnl_percent'] >= 0 else 'price-negative'}">{stock['pnl_percent']:+.2f}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">52W Range</span>
                            <span class="detail-value">₹{stock['week_52_low']:,.0f}-₹{stock['week_52_high']:,.0f}</span>
                        </div>
                    </div>
                    <div class="technical-signals">
                        <div style="font-weight: bold; margin-bottom: 8px; color: #2c3e50;">Technical Indicators</div>
                        <div class="signals-grid">
                            <div class="signal-item">
                                <div class="signal-label">RSI (14)</div>
                                <div class="signal-value" style="color: {rsi_color};">{stock['rsi']:.0f}</div>
                            </div>
                            <div class="signal-item">
                                <div class="signal-label">MACD</div>
                                <div class="signal-value" style="color: {macd_color};">{stock['macd']}</div>
                            </div>
                            <div class="signal-item">
                                <div class="signal-label">Trend</div>
                                <div class="signal-value" style="color: {trend_color};">{stock['trend']}</div>
                            </div>
                        </div>
                    </div>
                    <div class="allocation-bar">
                        <div class="allocation-fill" style="width: {allocation_percent}%;">₹{allocation_amount:,.0f} ({allocation_percent:.0f}%)</div>
                    </div>
                    <div class="recommendation">
                        <div class="recommendation-text">
                            {stock['analysis']}
                        </div>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    html += f"""
        <!-- Risk Warning -->
        <div style="background: #fff3cd; border: 2px solid #ffc107; border-radius: 12px; padding: 20px; margin-top: 25px;">
            <h3 style="color: #856404; margin-bottom: 10px;">⚠️ Important Risk Disclaimers</h3>
            <ul style="color: #856404; line-height: 1.8; margin-left: 20px;">
                <li><strong>Market Volatility:</strong> Markets can be volatile - expect price swings</li>
                <li><strong>Diversification:</strong> Don't invest all at once - spread over 2-4 weeks</li>
                <li><strong>Stop Loss:</strong> Maintain 8-10% stop loss from purchase price</li>
                <li><strong>Review Frequency:</strong> Reassess weekly based on market movements</li>
                <li><strong>Cash Reserve:</strong> Keep ₹{cash_reserve:,.0f} ({CASH_RESERVE_PERCENT}%) in cash for opportunities</li>
                <li><strong>Not Financial Advice:</strong> This is technical analysis only. Consult your financial advisor</li>
            </ul>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; margin-top: 30px; padding: 20px; color: #666; font-style: italic;">
            <p><strong>Remember:</strong> "The stock market is a device for transferring money from the impatient to the patient." - Warren Buffett</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html

def generate_analysis_text(symbol, avg_price, current_price, rsi, macd, trend, recommendation):
    """Generate analysis text for each stock"""
    pnl_percent = ((current_price - avg_price) / avg_price) * 100
    
    analysis = f"<strong>📊 Technical Analysis:</strong> "
    
    # RSI analysis
    if rsi < 30:
        analysis += f"RSI at {rsi:.0f} shows oversold condition - strong reversal potential. "
    elif rsi < 40:
        analysis += f"RSI at {rsi:.0f} approaching oversold levels. "
    elif rsi > 70:
        analysis += f"RSI at {rsi:.0f} indicates overbought - caution advised. "
    elif rsi > 60:
        analysis += f"RSI at {rsi:.0f} showing momentum strength. "
    else:
        analysis += f"RSI at {rsi:.0f} in neutral zone. "
    
    # MACD analysis
    analysis += f"MACD is {macd.lower()}. "
    
    # Trend analysis
    analysis += f"Current trend: {trend}. "
    
    # P&L context
    if pnl_percent > 0:
        analysis += f"Currently in profit at +{pnl_percent:.2f}%. "
    else:
        analysis += f"Currently at {pnl_percent:.2f}%. "
    
    analysis += "<br><br><strong>💡 Recommendation:</strong> "
    
    # Recommendation text
    if recommendation == "BUY":
        if rsi < 35:
            analysis += "Strong buy opportunity due to oversold conditions. Consider adding to position."
        else:
            analysis += "Technical indicators support accumulation. Good entry point for long-term investors."
    elif recommendation == "HOLD":
        analysis += "Maintain current position. Wait for clearer technical signals before adding. Monitor closely."
    else:  # AVOID
        analysis += "Avoid fresh investment now. Technical indicators suggest weakness. Wait for better entry point."
    
    return analysis

def send_email(html_content, recipient_email, sender_email, sender_password):
    """Send email with HTML report"""
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Portfolio Investment Report - {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d %B %Y, %I:%M %p IST')}"
        
        # Attach HTML
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        print("✅ Email sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False

def main():
    """Main function to generate report"""
    print("🚀 Starting Portfolio Analysis...")
    
    # Get current time in IST
    ist = pytz.timezone('Asia/Kolkata')
    report_time = datetime.now(ist).strftime('%d %B %Y, %I:%M %p IST')
    
    print(f"📅 Report Time: {report_time}")
    
    # Fetch data for all stocks
    portfolio_data = []
    
    for symbol, avg_price in PORTFOLIO.items():
        print(f"📊 Fetching data for {symbol}...")
        
        stock_data = get_stock_data(symbol)
        if not stock_data:
            continue
        
        indicators = get_technical_indicators(symbol)
        
        current_price = stock_data['current_price']
        rsi = indicators['rsi']
        macd = indicators['macd']
        trend = indicators['trend']
        
        recommendation, rec_class = generate_recommendation(
            symbol, avg_price, current_price, rsi, macd, trend
        )
        
        analysis = generate_analysis_text(
            symbol, avg_price, current_price, rsi, macd, trend, recommendation
        )
        
        pnl_percent = ((current_price - avg_price) / avg_price) * 100
        
        # Clean display name
        display_name = symbol.replace('.NS', '').replace('-RR', ' REIT')
        
        portfolio_data.append({
            'symbol': symbol,
            'display_name': display_name,
            'avg_price': avg_price,
            'current_price': current_price,
            'pnl_percent': pnl_percent,
            'week_52_high': stock_data['week_52_high'],
            'week_52_low': stock_data['week_52_low'],
            'rsi': rsi,
            'macd': macd,
            'trend': trend,
            'recommendation': recommendation,
            'recommendation_class': rec_class,
            'analysis': analysis
        })
    
    # Calculate allocations
    allocations = calculate_allocations(portfolio_data)
    
    # Generate HTML report
    html_report = generate_html_report(portfolio_data, allocations, report_time)
    
    # Save to file
    output_file = 'index.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"✅ Report generated: {output_file}")
    
    # Send email (configure these in GitHub Secrets)
    sender_email = os.environ.get('SENDER_EMAIL')
    sender_password = os.environ.get('SENDER_PASSWORD')
    recipient_email = os.environ.get('RECIPIENT_EMAIL')
    
    if sender_email and sender_password and recipient_email:
        send_email(html_report, recipient_email, sender_email, sender_password)
    else:
        print("⚠️ Email credentials not configured. Skipping email.")
    
    print("✅ Portfolio Analysis Complete!")

if __name__ == "__main__":
    main()
