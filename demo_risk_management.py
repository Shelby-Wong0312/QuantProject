"""
Risk Management System Demo
Demonstrates VaR, Position Sizing, Stop Loss, and ROI Monitoring
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from risk_management import RiskMetrics, PositionSizing, StopLoss, ROIMonitor, ROITarget


def demo_risk_metrics():
    """Demonstrate risk metrics calculations"""
    print("\n=== RISK METRICS DEMO ===")

    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year

    # Calculate metrics
    var_95 = RiskMetrics.calculate_var(returns, 0.95)
    cvar_95 = RiskMetrics.calculate_cvar(returns, 0.95)

    # Generate price series for drawdown
    prices = pd.Series(100 * np.cumprod(1 + returns))
    max_dd, start_idx, end_idx = RiskMetrics.calculate_max_drawdown(prices)

    sharpe = RiskMetrics.calculate_sharpe_ratio(returns)
    sortino = RiskMetrics.calculate_sortino_ratio(returns)

    print(f"VaR (95%): {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"CVaR (95%): {cvar_95:.4f} ({cvar_95*100:.2f}%)")
    print(f"Max Drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Sortino Ratio: {sortino:.4f}")


def demo_position_sizing():
    """Demonstrate position sizing strategies"""
    print("\n=== POSITION SIZING DEMO ===")

    # Sample trading statistics
    win_prob = 0.6
    avg_win = 0.05
    avg_loss = 0.03

    # Kelly criterion
    kelly_fraction = PositionSizing.kelly_criterion(win_prob, avg_win, avg_loss)
    print(f"Kelly Criterion: {kelly_fraction:.4f} ({kelly_fraction*100:.2f}%)")

    # Fixed proportion
    capital = 100000
    fixed_size = PositionSizing.fixed_proportion(capital, 0.02)
    print(f"Fixed Proportion (2%): ${fixed_size:.2f}")

    # Volatility adjusted
    volatility = 0.25
    vol_adjusted = PositionSizing.volatility_adjusted(capital, volatility, 0.15)
    print(f"Volatility Adjusted: ${vol_adjusted:.2f}")

    # Risk parity
    volatilities = [0.15, 0.20, 0.25, 0.30]
    risk_parity = PositionSizing.risk_parity_weights(volatilities)
    print(f"Risk Parity Weights: {[f'{w:.3f}' for w in risk_parity]}")

    # ATR position sizing
    price = 50.0
    atr = 2.0
    risk_amount = 1000
    atr_size = PositionSizing.atr_position_sizing(capital, price, atr, risk_amount)
    print(f"ATR Position Size: {atr_size} shares")


def demo_stop_loss():
    """Demonstrate stop loss management"""
    print("\n=== STOP LOSS DEMO ===")

    stop_manager = StopLoss()

    # Set different types of stops
    symbol = "SPY"
    entry_price = 400.0

    # Fixed stop
    stop_manager.set_fixed_stop(symbol, entry_price, 390.0, "long")
    print(f"Fixed Stop Set: Entry ${entry_price}, Stop ${390.0}")

    # Percentage stop
    stop_manager.set_percentage_stop(symbol + "_PCT", entry_price, 0.05, "long")
    print(f"Percentage Stop Set: Entry ${entry_price}, Stop 5%")

    # Trailing stop
    stop_manager.set_trailing_stop(symbol + "_TRAIL", entry_price, 10.0, "long")
    print(f"Trailing Stop Set: Entry ${entry_price}, Trail $10")

    # ATR stop
    atr = 5.0
    stop_manager.set_atr_stop(symbol + "_ATR", entry_price, atr, 2.0, "long")
    print(f"ATR Stop Set: Entry ${entry_price}, ATR {atr}, Multiplier 2.0")

    # Test stop triggers
    current_price = 385.0
    triggered = stop_manager.check_stop_triggered(symbol, current_price)
    print(f"Fixed Stop Triggered at ${current_price}: {triggered}")

    # Update trailing stop
    high_price = 420.0
    trailing_triggered = stop_manager.update_trailing_stop(
        symbol + "_TRAIL", current_price, high_price
    )
    print(f"Trailing Stop Updated with High ${high_price}, Triggered: {trailing_triggered}")


def demo_roi_monitor():
    """Demonstrate ROI monitoring"""
    print("\n=== ROI MONITORING DEMO ===")

    # Create ROI monitor with custom targets
    targets = ROITarget(
        annual_target=10.0,  # 1000%
        monthly_target=0.15,  # 15%
        degradation_threshold=5.0,  # 500%
        consecutive_months_limit=2,
    )

    roi_monitor = ROIMonitor(targets)

    # Simulate performance tracking
    base_date = datetime.now() - timedelta(days=90)

    for i in range(12):  # 12 periods
        # Generate sample returns
        if i < 8:  # Good performance first 8 periods
            monthly_return = np.random.normal(0.18, 0.05)  # 18% average
        else:  # Poor performance last 4 periods
            monthly_return = np.random.normal(0.03, 0.02)  # 3% average

        period_date = base_date + timedelta(days=i * 7)

        # Track performance
        roi_monitor.track_performance(
            returns=monthly_return, date=period_date, costs=0.01  # 1% costs
        )

    # Get performance summary
    summary = roi_monitor.get_performance_summary(90)
    print("Performance Summary (90 days):")
    print(f"  Average ROI: {summary['average_roi']:.2f}%")
    print(f"  Current ROI: {summary['current_roi']:.2f}%")
    print(f"  Meets Annual Target: {summary['meets_annual_target']*100:.1f}% of time")
    print(f"  Above Degradation: {summary['above_degradation']*100:.1f}% of time")
    print(f"  Active Alerts: {summary['active_alerts']}")

    # Show alerts
    alerts = roi_monitor.get_alerts()
    if alerts:
        print("\nActive Alerts:")
        for i, alert in enumerate(alerts):
            print(f"  {i+1}. {alert['type']}: {alert['message']}")

    # Zero cost validation
    total_returns = 12.5  # 1250%
    validation = roi_monitor.zero_cost_validation(total_returns, 0.0)
    print("\nZero Cost Validation:")
    print(f"  Is Zero Cost: {validation['is_zero_cost']}")
    print(f"  Meets 1000% Target: {validation['meets_1000_percent']}")
    print(f"  Sustainable: {validation['sustainable']}")


def main():
    """Run all risk management demos"""
    print("QUANTITATIVE TRADING RISK MANAGEMENT SYSTEM")
    print("=" * 50)

    demo_risk_metrics()
    demo_position_sizing()
    demo_stop_loss()
    demo_roi_monitor()

    print(f"\n{'='*50}")
    print("RISK MANAGEMENT SYSTEM READY FOR STAGE 6!")
    print("[OK] VaR & CVaR calculations")
    print("[OK] Kelly criterion & position sizing")
    print("[OK] Multi-type stop loss management")
    print("[OK] ROI monitoring with 1000% target")
    print("[OK] Zero-cost strategy validation")


if __name__ == "__main__":
    main()
