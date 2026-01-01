#!/usr/bin/env python3
import sys
sys.path.append('.')

from core.risk_manager import RiskManager

# Test 1: See what calculate_position actually does
print("=" * 60)
print("DEBUG: What does calculate_position() actually return?")
print("=" * 60)

rm = RiskManager(1000, margin_pct=0.20, risk_of_margin_pct=0.50)
risk_info = rm.get_risk_info()

print(f"\nRisk Manager Setup:")
print(f"  Capital: ${risk_info['capital']}")
print(f"  Risk amount: ${risk_info['risk_amount']}")
print(f"  Target amount: ${risk_info['risk_amount'] * 3}")

# Test different scenarios
test_cases = [
    (50000, 49000, "1% stop"),
    (50000, 49500, "0.5% stop"),
    (50000, 48500, "3% stop"),
]

for entry, stop_loss, desc in test_cases:
    print(f"\n{desc}:")
    print(f"  Entry: ${entry}, Stop: ${stop_loss}")
    
    position_size, risk_per_unit, max_loss = rm.calculate_position(
        entry=entry,
        stop_loss=stop_loss,
        direction="long",
        confidence=0.7
    )
    
    print(f"  Position size: {position_size}")
    print(f"  Risk per unit: ${risk_per_unit}")
    print(f"  Max loss: ${max_loss}")
    
    if position_size > 0:
        # Calculate what risk this position actually has
        price_risk = abs(entry - stop_loss)
        actual_risk = position_size * price_risk
        print(f"  Actual risk: ${actual_risk:.2f}")
        print(f"  Target risk: ${risk_info['risk_amount']}")
        print(f"  Difference: ${actual_risk - risk_info['risk_amount']:.2f}")
    
    print(f"  Is zero? {position_size == 0}")

# Test 2: Check if it's using min/max position sizes
print(f"\n" + "=" * 60)
print("DEBUG: Checking min/max position sizes")
print("=" * 60)

print(f"\nMin position size: {rm.min_position_size()}")
print(f"Max position size: {rm.max_position_size()}")

# Test 3: Manually calculate what position size SHOULD be
print(f"\n" + "=" * 60)
print("DEBUG: Manual calculation of correct position size")
print("=" * 60)

for entry, stop_loss, desc in test_cases:
    price_risk = abs(entry - stop_loss)
    
    # What position size gives us $100 risk?
    required_position_size = risk_info['risk_amount'] / price_risk
    
    print(f"\n{desc}:")
    print(f"  Entry: ${entry}, Stop: ${stop_loss}")
    print(f"  Price risk: ${price_risk}")
    print(f"  Required position size for ${risk_info['risk_amount']} risk: {required_position_size:.6f}")
    
    # What's the position value?
    position_value = required_position_size * entry
    print(f"  Position value: ${position_value:.2f}")
    
    # What margin is needed at 25x leverage?
    margin_needed = position_value / 25
    print(f"  Margin needed (25x): ${margin_needed:.2f}")
    print(f"  Available margin: ${risk_info['margin_amount']}")
    print(f"  Margin utilization: {margin_needed/risk_info['margin_amount']*100:.1f}%")