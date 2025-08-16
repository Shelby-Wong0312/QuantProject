# 🚀 Ultimate Simple Auto Trader

極簡量化交易系統 - 一個檔案監控4000+股票並自動交易

A minimalist quantitative trading system that monitors 4000+ stocks, generates signals, and executes trades automatically in just 300 lines of code.

## ✨ Why Simple?

**Traditional Quant Systems:**
- 40+ files, 2700+ lines of code
- Complex multi-tier architecture
- Over-engineered abstractions
- Difficult to understand and maintain

**This System:**
- **1 file, 300 lines of code**
- Same functionality, 90% less complexity
- Easy to understand, modify, and deploy
- **Complexity ≠ Profitability**

## 🚀 Features

- **Monitor 4000+ Stocks** - Real-time monitoring via Yahoo Finance
- **Auto Signal Generation** - RSI, Volume Spike, Price Breakout detection
- **Auto Trading** - Seamless integration with Capital.com API
- **Risk Management** - Built-in 2% stop loss and 5% take profit
- **Multi-threading** - Parallel processing for speed
- **Zero Dependencies** - Only 5 essential packages needed

## 📊 System Architecture

```
Yahoo Finance (Free Real-time Data)
    ↓
Signal Scanner (RSI + Volume + Breakout)
    ↓
Trade Executor (Capital.com API)
    ↓
Risk Manager (Stop Loss + Position Sizing)
```

## 📁 Files

```
QuantProject/
├── ULTIMATE_SIMPLE_TRADER.py  # Main system (300 lines)
├── RUN_SIMPLE_TRADER.bat      # Windows launcher
├── test_ultimate.py           # Test script
├── .env                       # API credentials
└── data/
    └── all_symbols.txt        # 4000+ stock symbols
```

## 🛠️ Quick Start

### 1. Clone & Install (30 seconds)
```bash
git clone https://github.com/yourusername/QuantProject.git
cd QuantProject
pip install yfinance pandas numpy requests python-dotenv
```

### 2. Configure API (.env file)
```env
CAPITAL_API_KEY=your_api_key
CAPITAL_IDENTIFIER=your_email
CAPITAL_API_PASSWORD=your_password
```

### 3. Run!
```bash
python ULTIMATE_SIMPLE_TRADER.py
```

That's it! System will start monitoring 4000+ stocks immediately.

## 📈 Trading Strategy

### Buy Signals
- **RSI < 30** (Oversold condition)
- **Volume > 2x average** (Unusual activity)
- **Price > MA20 * 1.02** (Breakout)

### Sell Signals
- **RSI > 70** (Overbought condition)
- **Stop Loss** (-2% from entry)
- **Take Profit** (+5% from entry)

### Risk Management
- Maximum 20 concurrent positions
- 1% capital per trade
- Automatic stop loss/take profit

## 📊 How It Works

```python
# 1. Scan 4000+ stocks every 30 seconds
for symbol in stocks:
    signal = get_signals(symbol)
    
# 2. Calculate indicators
RSI = calculate_rsi(prices, period=14)
Volume_Ratio = current_volume / average_volume

# 3. Generate signals
if RSI < 30 and Volume_Ratio > 2:
    execute_trade('BUY', symbol)
    
# 4. Manage positions
if price < stop_loss or price > take_profit:
    close_position(symbol)
```

## 🎯 Performance

- **Scan Speed**: 4000+ stocks in ~10 seconds
- **Signal Rate**: ~5-10 signals per scan
- **Execution**: < 1 second per trade
- **CPU Usage**: < 20% on modern hardware
- **Memory**: < 500MB RAM

## 🧪 Testing

```bash
# Test signal generation
python test_ultimate.py

# Expected output:
[TEST] System initialized successfully!
[TEST] Loaded 4000 stocks
[TEST] Testing signal generation...
  AAPL: No signal
  ABBV: SELL (RSI=81.7)
  ...
[TEST] All tests passed!
```

## 🔧 Customization

Easy to modify parameters in `ULTIMATE_SIMPLE_TRADER.py`:

```python
# Trading parameters
self.min_rsi = 30         # RSI oversold level
self.max_rsi = 70         # RSI overbought level
self.volume_spike = 2.0   # Volume multiplier
self.position_size = 0.01 # 1% per trade
self.stop_loss = 0.02     # 2% stop loss
self.take_profit = 0.05   # 5% take profit
```

## ⚠️ Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Always test with demo accounts first. Past performance does not guarantee future results.

## 📄 License

MIT License - Use at your own risk

## 🤝 Contributing

Pull requests welcome! Keep it simple.

---

**Remember: The best trading system is the one you understand.**

Built with ❤️ for traders who prefer results over complexity