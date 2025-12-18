"""
Monte Carlo Simulation for Strategy Validation
"""
import random
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt

class MonteCarloTester:
    """Run Monte Carlo simulations on strategy performance"""
    
    def __init__(self, initial_capital: float = 500.0):
        self.initial_capital = initial_capital
        
    def simulate_trades(self, win_rate: float = 0.3, r_r_ratio: float = 3.0, 
                       num_trades: int = 100, risk_per_trade: float = 0.02,
                       leverage: float = 10.0) -> Dict:
        """Simulate a series of trades with given parameters."""
        
        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        
        for i in range(num_trades):
            # Determine if trade wins or loses
            is_win = random.random() < win_rate
            
            # Calculate risk amount
            risk_amount = capital * risk_per_trade * leverage
            
            if is_win:
                # Win: gain risk_amount * R:R ratio
                profit = risk_amount * r_r_ratio
                trades.append({"outcome": "WIN", "profit": profit})
            else:
                # Lose: lose risk_amount
                profit = -risk_amount
                trades.append({"outcome": "LOSS", "profit": profit})
            
            # Update capital
            capital += profit
            equity_curve.append(max(capital, 0))  # Can't go below 0
            
            # Stop if bankrupt
            if capital <= 0:
                break
        
        return {
            "final_capital": capital,
            "total_return": (capital / self.initial_capital - 1) * 100,
            "equity_curve": equity_curve,
            "trades": trades,
            "win_rate_actual": len([t for t in trades if t["outcome"] == "WIN"]) / len(trades) if trades else 0,
            "max_drawdown": self.calculate_max_drawdown(equity_curve)
        }
    
    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from peak."""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def run_multiple_simulations(self, num_simulations: int = 1000, **kwargs):
        """Run many simulations and collect statistics."""
        
        results = []
        for i in range(num_simulations):
            result = self.simulate_trades(**kwargs)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{num_simulations} simulations")
        
        # Calculate statistics
        final_capitals = [r["final_capital"] for r in results]
        total_returns = [r["total_return"] for r in results]
        max_drawdowns = [r["max_drawdown"] for r in results]
        
        stats = {
            "mean_final_capital": np.mean(final_capitals),
            "median_final_capital": np.median(final_capitals),
            "std_final_capital": np.std(final_capitals),
            "mean_return": np.mean(total_returns),
            "median_return": np.median(total_returns),
            "winning_sims": len([c for c in final_capitals if c > self.initial_capital]),
            "losing_sims": len([c for c in final_capitals if c < self.initial_capital]),
            "bankrupt_sims": len([c for c in final_capitals if c <= 0]),
            "avg_max_drawdown": np.mean(max_drawdowns),
            "max_max_drawdown": np.max(max_drawdowns),
            "min_final_capital": np.min(final_capitals),
            "max_final_capital": np.max(final_capitals),
        }
        
        return stats, results
    
    def plot_results(self, results: List[Dict], stats: Dict):
        """Plot Monte Carlo simulation results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Equity curves (sample of 100)
        sample_size = min(100, len(results))
        for i in range(sample_size):
            axes[0, 0].plot(results[i]["equity_curve"], alpha=0.1, color='blue')
        axes[0, 0].set_title(f'Sample of {sample_size} Equity Curves')
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histogram of final capitals
        final_capitals = [r["final_capital"] for r in results]
        axes[0, 1].hist(final_capitals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(self.initial_capital, color='red', linestyle='--', label='Initial Capital')
        axes[0, 1].axvline(stats["mean_final_capital"], color='green', linestyle='--', label='Mean')
        axes[0, 1].set_title('Distribution of Final Capital')
        axes[0, 1].set_xlabel('Final Capital ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Histogram of returns
        returns = [(r["final_capital"]/self.initial_capital - 1)*100 for r in results]
        axes[1, 0].hist(returns, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(0, color='red', linestyle='--', label='Breakeven')
        axes[1, 0].axvline(stats["mean_return"], color='green', linestyle='--', label='Mean')
        axes[1, 0].set_title('Distribution of Total Returns')
        axes[1, 0].set_xlabel('Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Statistics table
        stats_text = f"""
        Statistics ({len(results)} simulations):
        
        Initial Capital: ${self.initial_capital:,.2f}
        Mean Final Capital: ${stats['mean_final_capital']:,.2f}
        Median Final Capital: ${stats['median_final_capital']:,.2f}
        
        Mean Return: {stats['mean_return']:.2f}%
        Median Return: {stats['median_return']:,.2f}%
        
        Winning Simulations: {stats['winning_sims']} ({stats['winning_sims']/len(results)*100:.1f}%)
        Losing Simulations: {stats['losing_sims']} ({stats['losing_sims']/len(results)*100:.1f}%)
        Bankrupt Simulations: {stats['bankrupt_sims']} ({stats['bankrupt_sims']/len(results)*100:.1f}%)
        
        Avg Max Drawdown: {stats['avg_max_drawdown']:.2f}%
        Max Drawdown: {stats['max_max_drawdown']:.2f}%
        
        Min Final Capital: ${stats['min_final_capital']:,.2f}
        Max Final Capital: ${stats['max_final_capital']:,.2f}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=9,
                       verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('monte_carlo_results.png', dpi=150, bbox_inches='tight')
        plt.show()

# Run simulation
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION - 3:1 R:R with 10x Leverage")
    print("="*60)
    
    tester = MonteCarloTester(initial_capital=500.0)
    
    # Run with realistic parameters (30% win rate needed for 3:1 R:R)
    stats, results = tester.run_multiple_simulations(
        num_simulations=1000,
        win_rate=0.3,      # 30% win rate
        r_r_ratio=3.0,     # 3:1 risk-reward
        num_trades=100,    # 100 trades per simulation
        risk_per_trade=0.02,  # 2% risk per trade
        leverage=10.0      # 10x leverage
    )
    
    print("\n📊 FINAL STATISTICS:")
    print("-" * 40)
    for key, value in stats.items():
        if 'capital' in key or 'Capital' in key:
            print(f"{key}: ${value:,.2f}")
        elif 'return' in key or 'drawdown' in key:
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value}")
    
    # Generate plot
    tester.plot_results(results, stats)
    
    # Probability of 7 consecutive wins
    prob_7_wins = 0.3 ** 7  # win_rate^7
    print(f"\n🎯 Probability of 7 consecutive wins (30% win rate):")
    print(f"   {prob_7_wins:.8f} or 1 in {1/prob_7_wins:,.0f}")