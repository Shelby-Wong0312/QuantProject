"""
Circuit Breaker Mechanism
熔斷機制
Cloud Quant - Task Q-603
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class BreakerLevel(Enum):
    """Circuit breaker levels"""
    NORMAL = 0
    LEVEL_1 = 1  # -5% -> 5 min pause
    LEVEL_2 = 2  # -10% -> 15 min pause
    LEVEL_3 = 3  # -15% -> 60 min pause
    EMERGENCY = 4  # -20% -> Full stop


@dataclass
class BreakerEvent:
    """Circuit breaker event record"""
    timestamp: datetime
    level: BreakerLevel
    trigger_value: float
    portfolio_value: float
    pause_duration: int  # minutes
    reason: str
    auto_resume: bool


class CircuitBreaker:
    """
    Circuit breaker system for risk management
    Automatically pauses trading during extreme market conditions
    """
    
    def __init__(self,
                 initial_value: float = 100000,
                 check_interval: int = 1):  # seconds
        """
        Initialize circuit breaker
        
        Args:
            initial_value: Initial portfolio value
            check_interval: Check interval in seconds
        """
        self.initial_value = initial_value
        self.check_interval = check_interval
        
        # Breaker configuration
        self.levels = {
            BreakerLevel.LEVEL_1: {
                'threshold': -0.05,  # -5%
                'pause_minutes': 5,
                'auto_resume': True
            },
            BreakerLevel.LEVEL_2: {
                'threshold': -0.10,  # -10%
                'pause_minutes': 15,
                'auto_resume': True
            },
            BreakerLevel.LEVEL_3: {
                'threshold': -0.15,  # -15%
                'pause_minutes': 60,
                'auto_resume': False
            },
            BreakerLevel.EMERGENCY: {
                'threshold': -0.20,  # -20%
                'pause_minutes': -1,  # Indefinite
                'auto_resume': False
            }
        }
        
        # State management
        self.current_level = BreakerLevel.NORMAL
        self.is_paused = False
        self.pause_start_time: Optional[datetime] = None
        self.pause_end_time: Optional[datetime] = None
        self.trading_enabled = True
        
        # History
        self.breaker_history: List[BreakerEvent] = []
        self.daily_high = initial_value
        self.daily_low = initial_value
        
        # Callbacks
        self.on_breaker_triggered: Optional[Callable] = None
        self.on_breaker_resumed: Optional[Callable] = None
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.current_portfolio_value = initial_value
        
        logger.info("Circuit Breaker initialized")
    
    def check_trigger(self, portfolio_value: float) -> Optional[BreakerLevel]:
        """
        Check if circuit breaker should be triggered
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Triggered breaker level or None
        """
        self.current_portfolio_value = portfolio_value
        
        # Update daily high/low
        self.daily_high = max(self.daily_high, portfolio_value)
        self.daily_low = min(self.daily_low, portfolio_value)
        
        # Calculate drawdown from initial value
        drawdown = (portfolio_value - self.initial_value) / self.initial_value
        
        # Check each level
        triggered_level = None
        for level, config in sorted(self.levels.items(), 
                                   key=lambda x: x[1]['threshold']):
            if drawdown <= config['threshold']:
                triggered_level = level
        
        # Trigger if necessary
        if triggered_level and triggered_level != self.current_level:
            if triggered_level.value > self.current_level.value:
                self._trigger_breaker(triggered_level, drawdown, portfolio_value)
                return triggered_level
        
        return None
    
    def _trigger_breaker(self, 
                        level: BreakerLevel,
                        trigger_value: float,
                        portfolio_value: float):
        """
        Trigger circuit breaker
        
        Args:
            level: Breaker level to trigger
            trigger_value: Drawdown value that triggered breaker
            portfolio_value: Current portfolio value
        """
        config = self.levels[level]
        pause_duration = config['pause_minutes']
        
        # Create event
        event = BreakerEvent(
            timestamp=datetime.now(),
            level=level,
            trigger_value=trigger_value,
            portfolio_value=portfolio_value,
            pause_duration=pause_duration,
            reason=f"Portfolio drawdown reached {trigger_value:.2%}",
            auto_resume=config['auto_resume']
        )
        
        # Update state
        self.current_level = level
        self.is_paused = True
        self.trading_enabled = False
        self.pause_start_time = datetime.now()
        
        if pause_duration > 0:
            self.pause_end_time = self.pause_start_time + timedelta(minutes=pause_duration)
        else:
            self.pause_end_time = None  # Indefinite pause
        
        # Record event
        self.breaker_history.append(event)
        
        # Log critical event
        logger.critical(f"CIRCUIT BREAKER TRIGGERED - Level {level.name}")
        logger.critical(f"Portfolio Value: ${portfolio_value:,.2f}")
        logger.critical(f"Drawdown: {trigger_value:.2%}")
        logger.critical(f"Trading paused for {pause_duration} minutes")
        
        # Execute callback
        if self.on_breaker_triggered:
            self.on_breaker_triggered(event)
        
        # Send notifications
        self._send_emergency_notification(event)
        
        # Schedule auto-resume if configured
        if config['auto_resume'] and pause_duration > 0:
            asyncio.create_task(self._schedule_resume(pause_duration))
    
    async def _schedule_resume(self, minutes: int):
        """
        Schedule automatic resume after pause
        
        Args:
            minutes: Minutes to wait before resume
        """
        await asyncio.sleep(minutes * 60)
        
        if self.is_paused:
            self.resume_trading("Automatic resume after timeout")
    
    def resume_trading(self, reason: str = "Manual resume"):
        """
        Resume trading after circuit breaker
        
        Args:
            reason: Reason for resuming
        """
        if not self.is_paused:
            logger.warning("Trading is not paused, cannot resume")
            return
        
        # Update state
        self.is_paused = False
        self.trading_enabled = True
        self.current_level = BreakerLevel.NORMAL
        self.pause_start_time = None
        self.pause_end_time = None
        
        logger.info(f"Trading resumed: {reason}")
        
        # Execute callback
        if self.on_breaker_resumed:
            self.on_breaker_resumed(reason)
    
    def emergency_stop(self, reason: str = "Emergency stop activated"):
        """
        Activate emergency stop
        
        Args:
            reason: Reason for emergency stop
        """
        self._trigger_breaker(
            BreakerLevel.EMERGENCY,
            -0.99,  # Extreme value
            self.current_portfolio_value
        )
        
        logger.critical(f"EMERGENCY STOP: {reason}")
    
    async def start_monitoring(self, 
                              portfolio_callback: Callable[[], float]):
        """
        Start continuous monitoring
        
        Args:
            portfolio_callback: Function to get current portfolio value
        """
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(portfolio_callback)
        )
        logger.info("Circuit breaker monitoring started")
    
    async def _monitoring_loop(self, 
                              portfolio_callback: Callable[[], float]):
        """
        Continuous monitoring loop
        
        Args:
            portfolio_callback: Function to get portfolio value
        """
        while True:
            try:
                # Get current portfolio value
                portfolio_value = portfolio_callback()
                
                # Check for breaker trigger
                self.check_trigger(portfolio_value)
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
            logger.info("Circuit breaker monitoring stopped")
    
    def _send_emergency_notification(self, event: BreakerEvent):
        """
        Send emergency notifications
        
        Args:
            event: Breaker event
        """
        # In production, this would send emails, SMS, etc.
        notification = {
            'type': 'CIRCUIT_BREAKER_ALERT',
            'timestamp': event.timestamp.isoformat(),
            'level': event.level.name,
            'portfolio_value': event.portfolio_value,
            'drawdown': f"{event.trigger_value:.2%}",
            'pause_duration': f"{event.pause_duration} minutes",
            'message': f"Circuit breaker {event.level.name} triggered"
        }
        
        # Save to file for now
        alerts_file = Path("reports/circuit_breaker_alerts.json")
        alerts_file.parent.mkdir(exist_ok=True)
        
        alerts = []
        if alerts_file.exists():
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
        
        alerts.append(notification)
        
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        logger.info(f"Emergency notification saved: {alerts_file}")
    
    def get_status(self) -> Dict:
        """
        Get circuit breaker status
        
        Returns:
            Status dictionary
        """
        status = {
            'current_level': self.current_level.name,
            'is_paused': self.is_paused,
            'trading_enabled': self.trading_enabled,
            'current_portfolio_value': self.current_portfolio_value,
            'drawdown': (self.current_portfolio_value - self.initial_value) / self.initial_value,
            'daily_high': self.daily_high,
            'daily_low': self.daily_low,
            'pause_start': self.pause_start_time.isoformat() if self.pause_start_time else None,
            'pause_end': self.pause_end_time.isoformat() if self.pause_end_time else None,
            'breaker_count': len(self.breaker_history)
        }
        
        # Add time remaining if paused
        if self.is_paused and self.pause_end_time:
            remaining = (self.pause_end_time - datetime.now()).total_seconds()
            status['pause_remaining_seconds'] = max(0, remaining)
        
        return status
    
    def get_history(self) -> List[Dict]:
        """
        Get breaker event history
        
        Returns:
            List of breaker events
        """
        return [
            {
                'timestamp': event.timestamp.isoformat(),
                'level': event.level.name,
                'trigger_value': f"{event.trigger_value:.2%}",
                'portfolio_value': event.portfolio_value,
                'pause_duration': event.pause_duration,
                'reason': event.reason,
                'auto_resume': event.auto_resume
            }
            for event in self.breaker_history
        ]
    
    def reset_daily_limits(self):
        """Reset daily high/low limits"""
        self.daily_high = self.current_portfolio_value
        self.daily_low = self.current_portfolio_value
        logger.info("Daily limits reset")
    
    def update_thresholds(self, level: BreakerLevel, threshold: float):
        """
        Update breaker threshold
        
        Args:
            level: Breaker level to update
            threshold: New threshold value
        """
        if level in self.levels:
            self.levels[level]['threshold'] = threshold
            logger.info(f"Updated {level.name} threshold to {threshold:.2%}")


if __name__ == "__main__":
    # Test circuit breaker
    breaker = CircuitBreaker(initial_value=100000)
    
    print("Circuit Breaker Test")
    print("=" * 50)
    
    # Test different portfolio values
    test_values = [
        100000,  # Initial
        97000,   # -3% (no trigger)
        94000,   # -6% (Level 1)
        88000,   # -12% (Level 2)
        83000,   # -17% (Level 3)
        78000,   # -22% (Emergency)
    ]
    
    for value in test_values:
        triggered = breaker.check_trigger(value)
        status = breaker.get_status()
        
        print(f"\nPortfolio: ${value:,}")
        print(f"Drawdown: {status['drawdown']:.2%}")
        print(f"Level: {status['current_level']}")
        print(f"Paused: {status['is_paused']}")
        
        if triggered:
            print(f"*** BREAKER TRIGGERED: {triggered.name} ***")
    
    # Test manual resume
    if breaker.is_paused:
        print("\nTesting manual resume...")
        breaker.resume_trading("Test resume")
        print(f"Trading enabled: {breaker.trading_enabled}")
    
    # Show history
    history = breaker.get_history()
    print(f"\nBreaker History: {len(history)} events")
    for event in history:
        print(f"  {event['timestamp']}: {event['level']} at {event['trigger_value']}")