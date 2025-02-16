# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:55:41 2025

@author: RW
"""
import numpy as np

# buy stock profit/loss

def buy_stock(x, stock_price):
         
    # Calculate the trading fee
    trading_fee = np.maximum(stock_price * 10000 * 0.0001854, 5)
    
    # Calculate the profit 
    profits = (x - stock_price) * 10000 - trading_fee
    
    return profits


# buy call Option profit/loss

def buy_call(x, call_price, strike_price, fee_per_contract=3.6):
    # Profit = max(0, x - strike_price) - call_price
    return (np.maximum(0, x - strike_price) - call_price) * 10000 - fee_per_contract   

#Sell call Option profit/loss
def sell_call(x, call_price, strike_price, fee_per_contract=3.6):
    """
    Calculate the profit/loss for selling a call option.

    Parameters:
    - x: array-like, stock prices at expiration.
    - call_price: float, premium received for selling the call.
    - strike_price: float, strike price of the call option.
    - fee_per_contract: float, total trading fee per contract (default: 3.6 CNY),including:
        Option trading fee期权交易费: 2.0 CNY per option contract.
        Clearing fee结算费: 0.3 CNY per option contract.
        Handling fee 经手费: 1.3 CNY per option contract.

    Returns:
    - array-like, profit/loss values.
    """
    return (np.minimum(0, strike_price - x) + call_price) * 10000 - fee_per_contract

# Buy Put Option profit/loss
def buy_put(x, put_price, strike_price, fee_per_contract=3.6):
    # Profit = max(0, strike price - x) - put price
    return (np.maximum(0, strike_price - x) - put_price) * 10000 - fee_per_contract

# Sell Put Option profit/loss
def sell_put(x, put_price, strike_price, fee_per_contract=3.6):
    # Profit = min(0, x - strike price) + put price
    return (np.minimum(0, x - strike_price) + put_price) * 10000 - fee_per_contract



# Covered Write Strategy profit/loss
def covered_write(x, stock_price, call_price, strike_price):
    if x > strike_price:
        # Optioned assigned, selling stocks to call buyer
        profit = (strike_price - stock_price)*10000 - np.maximum(strike_price * 10000 * 0.0001854, 5)
    else:
        # Profit from stock: (Stock price - Buy price)
        stock_profit = buy_stock(x, stock_price)
        
        # Profit from selling the call: sell_call(x, call_price, strike_price)
        call_profit = sell_call(x, call_price, strike_price)
        
        profit = stock_profit + call_profit

    # Total profit from covered write strategy
    return profit




