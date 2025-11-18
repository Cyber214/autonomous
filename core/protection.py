"""
Protection System for PulseTraderX
Loss limits, volatility guard, consecutive loss shutdown, schedule enforcement
"""

import datetime
import pandas as pd

class ProtectionSystem:
    def __init__(self, max_daily_loss=50.0, max_consecutive_losses=5, trading_hours=("07:00", "18:00")):
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses
        self.trading_hours = trading_hours

        self.daily_start = datetime.date.today()
        self.daily_loss = 0.0
        self.consecutive_losses = 0

        self.last_shutdown = None
        self.cooldown_minutes = 5

    # ------------------------------------------------------------
    def reset_daily_if_needed(self):
        today = datetime.date.today()
        if today != self.daily_start:
            self.daily_start = today
            self.daily_loss = 0.0
            self.consecutive_losses = 0

    # ------------------------------------------------------------
    def update_after_trade(self, profit_loss):
        self.reset_daily_if_needed()
        self.daily_loss += (-profit_loss if profit_loss < 0 else 0)

        if profit_loss < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    # ------------------------------------------------------------
    def loss_limit_triggered(self):
        return self.daily_loss >= self.max_daily_loss

    # ------------------------------------------------------------
    def consecutive_loss_triggered(self):
        return self.consecutive_losses >= self.max_consecutive_losses

    # ------------------------------------------------------------
    def within_trading_hours(self):
        start, end = self.trading_hours
        now = datetime.datetime.now().strftime("%H:%M")
        return start <= now <= end

    # ------------------------------------------------------------
    def schedule_blocked(self):
        return not self.within_trading_hours()

    # ------------------------------------------------------------
    def in_cooldown(self):
        if not self.last_shutdown:
            return False
        delta = datetime.datetime.now() - self.last_shutdown
        return delta.total_seconds() < self.cooldown_minutes * 60

    # ------------------------------------------------------------
    def should_shutdown(self):
        if self.loss_limit_triggered():
            self.last_shutdown = datetime.datetime.now()
            return True
        if self.consecutive_loss_triggered():
            self.last_shutdown = datetime.datetime.now()
            return True
        if self.schedule_blocked():
            return True
        return False

    # ------------------------------------------------------------
    def can_resume(self):
        if self.in_cooldown():
            return False
        if self.schedule_blocked():
            return False
        return True

    # ------------------------------------------------------------
    def update_schedule(self, start, end):
        self.trading_hours = (start, end)

    # ------------------------------------------------------------
    def summary(self):
        return {
            "daily_loss": self.daily_loss,
            "consecutive_losses": self.consecutive_losses,
            "loss_limit": self.max_daily_loss,
            "hours": self.trading_hours,
            "shutdown": self.last_shutdown,
        }
