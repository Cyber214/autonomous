"""
Test different market regimes
"""
import logging
logger = logging.getLogger(__name__)
import numpy as np
from datetime import datetime, timedelta

class MarketRegimeTester:
    """Test strategy in different market conditions"""
    
    def generate_sideways_data(self, start_price: float = 70000.0, periods: int = 100):
        """Generate sideways/choppy market data."""
        prices = [start_price]
        for i in range(periods):
            # Small random movements (±0.5%)
            change_pct = np.random.uniform(-0.005, 0.005)
            new_price = prices[-1] * (1 + change_pct)
            prices.append(new_price)
        
        logger.info(f"📊 Generated {periods} periods of SIDEWAYS data")
        logger.info(f"  Start: ${start_price:.2f}")
        logger.info(f"  End: ${prices[-1]:.2f}")
        logger.info(f"  Range: {min(prices):.2f} - {max(prices):.2f}")
        
        return prices
    
    def generate_uptrend_data(self, start_price: float = 70000.0, periods: int = 100):
        """Generate bullish uptrend data."""
        prices = [start_price]
        trend_strength = 0.001  # 0.1% per period trend
        
        for i in range(periods):
            # Upward bias with noise
            change_pct = np.random.uniform(-0.005, 0.008) + trend_strength
            new_price = prices[-1] * (1 + change_pct)
            prices.append(new_price)
        
        logger.info(f"📊 Generated {periods} periods of UPTREND data")
        logger.info(f"  Start: ${start_price:.2f}")
        logger.info(f"  End: ${prices[-1]:.2f}")
        logger.info(f"  Total Return: {(prices[-1]/start_price-1)*100:.2f}%")
        
        return prices
    
    def generate_volatile_data(self, start_price: float = 70000.0, periods: int = 100):
        """Generate high volatility/whippy data."""
        prices = [start_price]
        
        for i in range(periods):
            # Large random movements (±2%)
            change_pct = np.random.uniform(-0.02, 0.02)
            new_price = prices[-1] * (1 + change_pct)
            prices.append(new_price)
        
        logger.info(f"📊 Generated {periods} periods of VOLATILE data")
        logger.info(f"  Start: ${start_price:.2f}")
        logger.info(f"  End: ${prices[-1]:.2f}")
        logger.info(f"  Max Daily Move: {max(abs(np.diff(prices))/prices[:-1])*100:.2f}%")
        
        return prices
    
    def generate_downtrend_data(self, start_price: float = 70000.0, periods: int = 100):
        """Generate bearish downtrend data (your original test)."""
        prices = [start_price]
        trend_strength = -0.002  # -0.2% per period trend
        
        for i in range(periods):
            # Downward bias with noise
            change_pct = np.random.uniform(-0.008, 0.005) + trend_strength
            new_price = prices[-1] * (1 + change_pct)
            prices.append(new_price)
        
        logger.info(f"📊 Generated {periods} periods of DOWNTREND data")
        logger.info(f"  Start: ${start_price:.2f}")
        logger.info(f"  End: ${prices[-1]:.2f}")
        logger.info(f"  Total Return: {(prices[-1]/start_price-1)*100:.2f}%")
        
        return prices

# Test all regimes
if __name__ == "__main__":
    tester = MarketRegimeTester()
    
    print("\n" + "="*50)
    print("TESTING ALL MARKET REGIMES")
    print("="*50)
    
    # Test each regime
    regimes = {
        "SIDEWAYS": tester.generate_sideways_data(periods=30),
        "UPTREND": tester.generate_uptrend_data(periods=30),
        "VOLATILE": tester.generate_volatile_data(periods=30),
        "DOWNTREND": tester.generate_downtrend_data(periods=30)
    }
    
    # You would feed these prices into your trading system
    # For now, just save them
    import pandas as pd
    for regime, prices in regimes.items():
        df = pd.DataFrame({"price": prices})
        df.to_csv(f"{regime.lower()}_test_data.csv", index=False)
        print(f"✅ Saved {regime} data: {len(prices)} periods")