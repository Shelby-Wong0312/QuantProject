#!/usr/bin/env python3
"""
PPO LIVE TRADING SYSTEM
使用訓練好的PPO模型進行實盤交易
"""

import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import requests
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ActorCritic(nn.Module):
    """PPO Actor-Critic Network"""

    def __init__(self, obs_dim: int = 220, action_dim: int = 4, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1)
        )

        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(self, obs: torch.Tensor) -> int:
        """Get action from observation"""
        with torch.no_grad():
            action_logits, _ = self.forward(obs)
            probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action


class PPOLiveTrader:
    """PPO實盤交易系統"""

    def __init__(self):
        # Capital.com API
        self.api_key = os.getenv("CAPITAL_API_KEY")
        self.identifier = os.getenv("CAPITAL_IDENTIFIER")
        self.password = os.getenv("CAPITAL_API_PASSWORD")
        # Allow switching Demo/Live via env; default to Demo
        self.base_url = os.getenv(
            "CAPITAL_BASE_URL", "https://demo-api-capital.backend-capital.com"
        )
        self.session_token = None
        self.cst = None

        # Safety / test flags
        self.dry_run = (
            os.getenv("DRY_RUN", "0") == "1"
        )  # When true, skip login and never place orders
        self.max_loops = int(os.getenv("MAX_LOOPS", "0"))  # When >0, stop after N scan loops
        self.scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))  # Interval between scans

        # Load PPO model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(obs_dim=220, action_dim=4)  # Match trained model dimensions
        self.load_model()

        # Trading parameters
        self.symbols_file = os.getenv("SYMBOLS_FILE", "").strip()
        # Default to 0 (no cap) so thousands can be loaded if not configured
        self.max_symbols = int(os.getenv("MAX_SYMBOLS", "0"))  # 0 or negative = no cap
        # Default to 200 to avoid overloading when scanning thousands
        self.batch_size = int(os.getenv("BATCH_SIZE", "200"))  # 0 = process all each loop
        self.batch_index = 0
        # Symbol mappings (Capital <-> Yahoo, EPIC codes)
        self.symbol_map_yahoo: Dict[str, str] = {}
        self.symbol_map_epic: Dict[str, str] = {}
        self.load_symbol_mappings()
        self.symbols = self.load_symbols()
        self.positions = {}
        self.max_positions = 10
        self.position_size = 0.02  # 2% per position

        # Feature parameters
        self.lookback = 50  # Number of candles for features
        self.feature_dim = 220  # Match model input dimension

        print(f"[PPO] Model loaded on {self.device}")
        print(f"[PPO] Monitoring {len(self.symbols)} stocks")

    def load_model(self):
        """Load trained PPO model"""
        # Try to load the best available model
        model_paths = [
            "reports/ml_models/ppo_trader_final.pt",  # Final trained model
            "ppo_trader_iter_150.pt",  # 150 iterations
            "ppo_trader_iter_100.pt",  # 100 iterations
            "ppo_trader_iter_50.pt",  # 50 iterations
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path and os.path.exists(model_path):
            # Load with weights_only=False to handle custom classes
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Extract model state from checkpoint
            if isinstance(checkpoint, dict):
                # Try different keys
                if "model" in checkpoint:
                    # Sometimes the whole model is saved
                    if hasattr(checkpoint["model"], "state_dict"):
                        self.model.load_state_dict(checkpoint["model"].state_dict())
                    else:
                        self.model = checkpoint["model"]
                elif "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                elif "actor_critic" in checkpoint:
                    self.model.load_state_dict(checkpoint["actor_critic"])
                else:
                    # Try to load as state dict directly
                    try:
                        self.model.load_state_dict(checkpoint)
                    except Exception:
                        print(f"[INFO] Checkpoint keys: {checkpoint.keys()}")
                        print("[WARNING] Could not load model weights, using random initialization")
            else:
                # Direct model state
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            print(f"[PPO] Loaded model from {model_path}")
        else:
            print(f"[WARNING] No trained model found at {model_path}")
            print("[WARNING] Using random initialized model")

    def load_symbol_mappings(self):
        """Load Capital<->Yahoo mapping to ensure we price/order correctly."""
        # Prefer full mapping file
        try:
            if os.path.exists("capital_yahoo_full_mapping.json"):
                with open("capital_yahoo_full_mapping.json", "r", encoding="utf-8") as f:
                    json.load(f)
                mapped = data.get("mapped") if isinstance(data, dict) else None
                if isinstance(mapped, list):
                    for item in mapped:
                        cap = str(item.get("capital_ticker", "")).upper()
                        epic = str(item.get("capital_epic", "")).upper()
                        yh = str(item.get("yahoo_symbol", "")).upper()
                        if cap:
                            if yh:
                                self.symbol_map_yahoo[cap] = yh
                            if epic:
                                self.symbol_map_epic[cap] = epic
        except Exception as e:
            logger.error(f"Failed to load full mapping: {e}")

        # Fallback: simple mapping (capital->yahoo)
        try:
            if os.path.exists("capital_yahoo_simple_map.json"):
                with open("capital_yahoo_simple_map.json", "r", encoding="utf-8") as f:
                    json.load(f)
                if isinstance(data, dict):
                    for cap, yh in data.items():
                        cap_u = str(cap).upper()
                        yh_u = str(yh).upper()
                        self.symbol_map_yahoo.setdefault(cap_u, yh_u)
        except Exception as e:
            logger.error(f"Failed to load simple mapping: {e}")

    def get_yahoo_symbol(self, capital_ticker: str) -> str:
        s = str(capital_ticker).upper()
        return self.symbol_map_yahoo.get(s, s)

    def get_epic(self, capital_ticker: str) -> str:
        s = str(capital_ticker).upper()
        if s in self.symbol_map_epic:
            return self.symbol_map_epic[s]
        return f"US.{s}.CASH"

    def _read_symbols_file(self, path: str) -> List[str]:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                raw: List[str] = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "," in line:
                        raw.extend([p.strip() for p in line.split(",") if p.strip()])
                    else:
                        raw.append(line)
                # normalize and dedupe
                symbols: List[str] = []
                seen = set()
                for s in raw:
                    s = s.upper()
                    if s and s not in seen:
                        seen.add(s)
                        symbols.append(s)
                return symbols
        except Exception as e:
            logger.error(f"Failed reading symbols from {path}: {e}")
            return []

    def load_symbols(self) -> List[str]:
        """Load stock symbols with flexible sources and limits"""
        candidates: List[str] = []
        sources_tried: List[str] = []

        if self.symbols_file:
            sources_tried.append(self.symbols_file)
            if os.path.exists(self.symbols_file):
                candidates = self._read_symbols_file(self.symbols_file)

        if not candidates:
            # common repo files as fallbacks (prefer validated list)
            fallback_files = [
                "validated_capital_symbols_final.txt",
                "data/tier_s.txt",
                "capital_symbols_all.txt",
                "validated_yahoo_symbols_final.txt",
                "capital_tickers.txt",
                "quick_yahoo_symbols.txt",
                "yahoo_symbols_all.txt",
            ]
            for fp in fallback_files:
                sources_tried.append(fp)
                if os.path.exists(fp):
                    candidates = self._read_symbols_file(fp)
                    if candidates:
                        break

        if not candidates:
            # Default stocks
            default_syms = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "AMD",
                "NFLX",
                "JPM",
                "BAC",
                "V",
                "MA",
                "WMT",
                "DIS",
                "PYPL",
            ]
            print("[PPO] No symbols file found or empty; using default symbols (16)")
            return default_syms

        total = len(candidates)
        if self.max_symbols and self.max_symbols > 0 and total > self.max_symbols:
            used = candidates[: self.max_symbols]
        else:
            used = candidates

        # Log which source selected
        src = next((s for s in sources_tried if os.path.exists(s)), "N/A")
        print(
            f"[PPO] Loaded {len(used)}/{total} symbols from {src}. Set MAX_SYMBOLS to adjust; use SYMBOLS_FILE to choose file."
        )
        return used

    def login_capital(self):
        """Login to Capital.com"""
        if self.dry_run:
            print("[DRY-RUN] Skip Capital.com login (no orders will be sent)")
            return False
        headers = {"X-CAP-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"identifier": self.identifier, "password": self.password}

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/session", headers=headers, json=payload, timeout=10
            )
            if response.status_code == 200:
                self.cst = response.headers.get("CST")
                self.session_token = response.headers.get("X-SECURITY-TOKEN")
                print("[OK] Capital.com connected")
                return True
            else:
                logger.error(
                    f"Capital.com login failed: HTTP {response.status_code} {response.text[:200] if response.text else ''}"
                )
        except Exception as e:
            print(f"[ERROR] Capital.com connection failed: {e}")

        return False

    def extract_features(self, symbol: str) -> np.ndarray:
        """Extract features for PPO model"""
        try:
            # Get historical data
            yahoo_symbol = self.get_yahoo_symbol(symbol)
            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(period="3mo", interval="1d")

            if len(hist) < self.lookback:
                return None

            # Get last N candles
            hist = hist.tail(self.lookback)

            # Calculate technical indicators
            features = []

            # 1. Price features (normalized)
            close_prices = hist["Close"].values
            returns = np.diff(close_prices) / close_prices[:-1]
            returns = np.append(returns, 0)  # Add 0 for last candle

            # 2. Volume (normalized)
            volumes = hist["Volume"].values
            vol_mean = volumes.mean()
            vol_std = volumes.std() + 1e-8
            norm_volumes = (volumes - vol_mean) / vol_std

            # 3. RSI
            delta = np.diff(close_prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.convolve(gain, np.ones(14) / 14, mode="valid")
            avg_loss = np.convolve(loss, np.ones(14) / 14, mode="valid")
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))

            # Pad RSI to match length
            rsi_padded = np.zeros(self.lookback)
            rsi_padded[-len(rsi) :] = rsi

            # 4. Moving averages
            ma20 = np.convolve(close_prices, np.ones(20) / 20, mode="same")
            ma_ratio = close_prices / (ma20 + 1e-8)

            # 5. Bollinger Bands
            bb_std = np.array(
                [close_prices[: i + 1].std() if i >= 19 else 0 for i in range(len(close_prices))]
            )
            bb_upper = ma20 + 2 * bb_std
            bb_lower = ma20 - 2 * bb_std
            bb_position = (close_prices - bb_lower) / (bb_upper - bb_lower + 1e-8)

            # Combine all features
            feature_matrix = np.column_stack(
                [
                    returns[-self.lookback :],
                    norm_volumes[-self.lookback :],
                    rsi_padded[-self.lookback :],
                    ma_ratio[-self.lookback :],
                    bb_position[-self.lookback :],
                ]
            )

            # Flatten and ensure we have exactly feature_dim features
            features = feature_matrix.flatten()
            if len(features) > self.feature_dim:
                features = features[: self.feature_dim]
            elif len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))

            return features

        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
            return None

    def get_ppo_signal(self, symbol: str) -> int:
        """Get trading signal from PPO model
        Returns: 0=Hold, 1=Buy, 2=Sell, 3=Strong Buy/Sell
        """
        features = self.extract_features(symbol)
        if features is None:
            return 0  # Hold if no features

        # Convert to tensor
        torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Get action from PPO model
        action = self.model.get_action(obs)

        # Map 4 actions to trading decisions
        # 0 = Hold, 1 = Buy, 2 = Sell, 3 = Strong signal (double position)
        return action

    def execute_trade(self, symbol: str, action: int, current_price: float):
        """Execute trade based on PPO signal"""
        # 0=Hold, 1=Buy, 2=Sell, 3=Strong signal

        if action == 0:  # Hold
            return

        elif action == 1 or action == 3:  # Buy (3 = strong buy with larger position)
            if symbol not in self.positions and len(self.positions) < self.max_positions:
                shares = 200 if action == 3 else 100  # Double size for strong signal
                self.positions[symbol] = {
                    "entry_price": current_price,
                    "shares": shares,
                    "entry_time": datetime.now(),
                    "stop_loss": current_price * 0.98,  # 2% stop loss
                    "take_profit": current_price * 1.05,  # 5% take profit
                }
                signal_type = "STRONG BUY" if action == 3 else "BUY"
                print(f"[PPO {signal_type}] {symbol} @ ${current_price:.2f} x {shares} shares")

                # Execute on Capital.com if connected
                if self.cst:
                    self.place_order(symbol, "BUY", shares)

        elif action == 2:  # Sell
            if symbol in self.positions:
                position = self.positions[symbol]
                pnl = (current_price - position["entry_price"]) * position["shares"]
                pnl_pct = (current_price / position["entry_price"] - 1) * 100
                print(
                    f"[PPO SELL] {symbol} @ ${current_price:.2f} | PnL: ${pnl:.2f} ({pnl_pct:.2f}%)"
                )
                del self.positions[symbol]

                # Execute on Capital.com if connected
                if self.cst:
                    self.place_order(symbol, "SELL", position["shares"])

    def place_capital_order(self, symbol: str, direction: str, size: int):
        """Place order on Capital.com"""
        if self.dry_run:
            print(f"[DRY-RUN] 模擬下單：{direction} {size} {symbol}")
            return
        if not self.cst:
            return

        headers = {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.session_token,
            "Content-Type": "application/json",
        }

        payload = {"epic": f"US.{symbol}.CASH", "direction": direction, "size": size}

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/positions", headers=headers, json=payload, timeout=10
            )
            if response.status_code == 200:
                print(f"[CAPITAL.COM] Order executed: {direction} {size} {symbol}")
        except Exception as e:
            logger.error(f"Capital.com order failed: {e}")

    def place_order(self, symbol: str, direction: str, size: int):
        """Place order using mapping and auto re-login on auth failure."""
        if self.dry_run:
            print(f"[DRY-RUN] Simulate order: {direction} {size} {symbol}")
            return

        if not self.cst:
            # Try to login if not connected
            if not self.login_capital():
                logger.error("Cannot place order: not connected to Capital.com")
                return

        headers = {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.session_token,
            "Content-Type": "application/json",
        }

        epic = self.get_epic(symbol)
        payload = {"epic": epic, "direction": direction, "size": size}

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/positions",
                headers=headers,
                json=payload,
                timeout=10,
            )
            if response.status_code == 200:
                print(f"[CAPITAL.COM] Order executed: {direction} {size} {symbol} (EPIC={epic})")
                return
            if response.status_code in (401, 403):
                logger.warning("Auth failed (401/403). Re-login and retry once.")
                if self.login_capital():
                    headers["CST"] = self.cst
                    headers["X-SECURITY-TOKEN"] = self.session_token
                    response = requests.post(
                        f"{self.base_url}/api/v1/positions",
                        headers=headers,
                        json=payload,
                        timeout=10,
                    )
                    if response.status_code == 200:
                        print(
                            f"[CAPITAL.COM] Order executed after relogin: {direction} {size} {symbol} (EPIC={epic})"
                        )
                        return
            logger.error(
                f"Order failed: HTTP {response.status_code} {response.text[:200] if getattr(response,'text',None) else ''}"
            )
        except Exception as e:
            logger.error(f"Capital.com order exception: {e}")

    def check_positions(self):
        """Check stop loss and take profit"""
        for symbol in list(self.positions.keys()):
            try:
                yahoo_symbol = self.get_yahoo_symbol(symbol)
                current_price = yf.Ticker(yahoo_symbol).info.get("regularMarketPrice", 0)
                position = self.positions[symbol]

                # Check stop loss
                if current_price <= position["stop_loss"]:
                    pnl = (current_price - position["entry_price"]) * position["shares"]
                    print(f"[STOP LOSS] {symbol} @ ${current_price:.2f} | Loss: ${pnl:.2f}")
                    del self.positions[symbol]
                    if self.cst:
                        self.place_order(symbol, "SELL", position["shares"])

                # Check take profit
                elif current_price >= position["take_profit"]:
                    pnl = (current_price - position["entry_price"]) * position["shares"]
                    print(f"[TAKE PROFIT] {symbol} @ ${current_price:.2f} | Profit: ${pnl:.2f}")
                    del self.positions[symbol]
                    if self.cst:
                        self.place_order(symbol, "SELL", position["shares"])

            except Exception as e:
                logger.error(f"Error checking position {symbol}: {e}")

    def run(self):
        """Main trading loop"""
        print("\n" + "=" * 60)
        print(" PPO LIVE TRADING SYSTEM")
        print(" AI-Powered Trading with Trained PPO Model")
        print("=" * 60)
        print(
            f"[CONFIG] base_url={self.base_url} dry_run={self.dry_run} interval={self.scan_interval}s"
        )

        # Connect to Capital.com
        self.login_capital()

        # Main loop
        loops = 0
        while True:
            try:
                total_syms = len(self.symbols)
                if total_syms == 0:
                    print("[WARNING] No symbols to scan. Check SYMBOLS_FILE or data/tier_s.txt")
                    time.sleep(self.scan_interval)
                    continue

                # batching support
                if self.batch_size and self.batch_size > 0 and self.batch_size < total_syms:
                    start = (self.batch_index * self.batch_size) % total_syms
                    end = start + self.batch_size
                    if end <= total_syms:
                        batch = self.symbols[start:end]
                    else:
                        batch = self.symbols[start:] + self.symbols[: end - total_syms]
                    self.batch_index += 1
                    scan_set = batch
                    print(
                        f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(scan_set)} of {total_syms} stocks (batch_size={self.batch_size})..."
                    )
                else:
                    scan_set = self.symbols
                    print(
                        f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning {total_syms} stocks with PPO..."
                    )

                # Process each symbol in scan set
                for symbol in scan_set:
                    try:
                        # Get PPO signal
                        action = self.get_ppo_signal(symbol)

                        # Get current price
                        current_price = yf.Ticker(symbol).info.get("regularMarketPrice", 0)

                        if current_price > 0:
                            # Execute trade based on PPO decision
                            self.execute_trade(symbol, action, current_price)

                            # Log PPO decision
                            if action == 1:
                                logger.info(f"PPO Signal: BUY {symbol}")
                            elif action == 2:
                                logger.info(f"PPO Signal: SELL {symbol}")

                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        continue

                # Check existing positions
                self.check_positions()

                # Display current positions
                if self.positions:
                    print(f"\n[POSITIONS] {len(self.positions)} active:")
                    for sym, pos in self.positions.items():
                        print(f"  {sym}: Entry=${pos['entry_price']:.2f}, Shares={pos['shares']}")

                # Wait before next scan
                time.sleep(self.scan_interval)

                # Stop in test mode after N loops
                loops += 1
                if self.max_loops and loops >= self.max_loops:
                    print(f"[INFO] MAX_LOOPS={self.max_loops} reached, exiting test run.")
                    break

            except KeyboardInterrupt:
                print("\n[EXIT] PPO Trading System stopped")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(5)


if __name__ == "__main__":
    trader = PPOLiveTrader()
    trader.run()
