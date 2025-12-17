# core/trade_executor.py
# ============================================================
# ==================== TRADE EXECUTION =======================
# ============================================================
import time
from datetime import datetime

from utils.logger import get_logger

logger = get_logger()


async def execute_trade(deriv, decision, config, protection, controller,
                        telegram, current_price, tick,
                        trade_amount=None, duration=None):
    """PROPER trade execution that follows strategy decisions"""
    try:
        trade_amount = trade_amount or config["strategy"]["trade_amount"]

        df = controller.strategy_engine._df()
        smart_duration = controller.analyze_market_volatility(df)
        duration = duration or smart_duration

        logger.info(
            f"Executing {decision} trade at price {current_price} "
            f"(Amount: ${trade_amount}, Duration: {duration}s)"
        )

        deriv_contract_type = "CALL" if decision.upper() == "BUY" else "PUT"

        trade_result = await deriv.buy(
            amount=trade_amount,
            symbol=config["deriv"]["symbol"],
            contract_type=deriv_contract_type,
            duration=duration
        )

        if isinstance(trade_result, dict):
            profit_loss = trade_amount * 0.95  # Standard binary payout

            if trade_result.get('ok'):
                if controller.real_balance is not None:
                    controller.real_balance -= trade_amount
                    balance_str = f"${controller.real_balance:.2f} (after trade)"
                    logger.info(
                        f"💰 Balance updated: -${trade_amount:.2f} "
                        f"(New balance: ${controller.real_balance:.2f})"
                    )
                else:
                    balance_str = "Unknown"
                    logger.error("⚠️ Cannot deduct balance: real_balance is None")

                minutes = duration // 60

                strategy_results = controller.strategy_engine.decide()[1]
                buy_count = list(strategy_results.values()).count("BUY")
                sell_count = list(strategy_results.values()).count("SELL")
                hold_count = list(strategy_results.values()).count("HOLD")

                combined_message = (
                    f"🎯 STRATEGY EXECUTED\n"
                    f"• Decision: {decision}\n"
                    f"• Votes: BUY {buy_count} | SELL {sell_count} | HOLD {hold_count}\n"
                    f"• Entry: ${current_price:.4f}\n"
                    f"• Amount: ${trade_amount}\n"
                    f"• Duration: {minutes} minutes\n"
                    f"• Potential Payout: ${profit_loss:+.2f}\n"
                    f"• Balance: {balance_str}\n"
                    f"⏰ Expires: {minutes} minutes\n"
                    f"📊 Strategy Breakdown:\n" +
                    "\n".join([f"  • {k}: {v}" for k, v in strategy_results.items()])
                )

                await telegram.send(combined_message)

                trade_data = {
                    'entry_price': current_price,
                    'amount': trade_amount,
                    'contract_type': decision,
                    'duration': duration,
                    'profit_loss': profit_loss,
                    'expiry_time': time.time() + duration,
                    'current_price': current_price,
                    'trade_result': trade_result
                }

                controller.add_pending_trade(trade_data)
                controller.update_trade_time()

            else:
                error_msg = trade_result.get('error', 'Unknown error')
                await telegram.send(f"❌ Trade failed: {error_msg}")
        else:
            await telegram.send(f"❌ Trade error: Invalid response")

        protection.update_after_trade(0)  # Will be updated when trade expires

        controller.log_trade(
            timestamp=tick.get("epoch", int(datetime.now().timestamp())),
            decision=decision,
            price=current_price,
            result=trade_result if isinstance(trade_result, dict) else {"error": "Invalid response"},
            profit_loss=0,
            entry_price=current_price,
            exit_price=None,
            outcome=None
        )

        return trade_result

    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        await telegram.send(f"❌ Trade execution error: {str(e)}")
        return None
