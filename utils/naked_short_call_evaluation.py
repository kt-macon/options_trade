# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:13:47 2025

@author: RW
"""

import numpy as np
import pandas as pd
from scipy.stats import lognorm
from scipy.integrate import simps
from utils.option_strategies import sell_call


'''
def calculate_expected_profit_rate(df_calls, mu, sigma, num_points=1000):
    """
    Calculate the expected profit and profit rate for each call option.

    Parameters:
    - df_calls (pd.DataFrame): DataFrame with 'call_price' and 'strike_price' columns.
    - mu (float): Logarithmic mean of the lognormal distribution.
    - sigma (float): Standard deviation of the lognormal distribution.
    - num_points (int): Number of points for numerical integration.

    Returns:
    - pd.DataFrame: Updated DataFrame with 'expected_profit' and 'expected_profit_rate (%)' columns.
    """
    from scipy.stats import lognorm
    from scipy.integrate import simps

    # Initialize result columns
    df_calls['expected_profit'] = 0.0
    df_calls['expected_profit_rate (%)'] = 0.0

    # Define stock price range for integration
    stock_price_range = np.linspace(0, 5 * np.exp(mu), num_points)
    pdf = lognorm.pdf(stock_price_range, sigma, scale=np.exp(mu))

    for index, row in df_calls.iterrows():
        call_price = row['call_price']
        strike_price = row['strike_price']

        # Calculate profit for selling a call at each stock price
        profits = sell_call(stock_price_range, call_price, strike_price)

        # Numerically integrate to calculate expected profit
        expected_profit = simps(profits * pdf, stock_price_range)

        # Calculate cost (assume 100% margin required)
        cost = (strike_price - call_price) * 10000  # Adjust for scaling (e.g., per contract)
        
        # Calculate expected profit rate
        expected_profit_rate = (expected_profit / cost) * 100 if cost > 0 else 0

        # Update the DataFrame
        df_calls.at[index, 'expected_profit'] = expected_profit
        df_calls.at[index, 'expected_profit_rate (%)'] = expected_profit_rate

    return df_calls

def sell_calls_evaluation(stock_price, df, sigma):
    """
    Plot the profit/loss and lognormal distribution for selling a call option.
    
    Parameters:
    - stock_price: float, current stock price.
    - call_price: float, premium received for selling the call.
    - strike_price: float, strike price of the call option.
    - sigma: float, volatility (standard deviation) of the stock.

    Returns:
    - dict with key financial metrics.
    """
    # Parameters
    #call_price = df_calls['call_price']
    #strike_price = df_calls['strike_price']
    mu = np.log(stock_price)  # Mean of the log of the stock price (lognormal)

    df_calls = df.copy()
    # Break-even point
    df_calls['break_even_x'] = df_calls['strike_price'] + df_calls['call_price']  # Break-even is strike price + call price
    df_calls['break_even_probability (%)'] = lognorm.cdf(df_calls['break_even_x'], sigma, scale=np.exp(mu)) * 100
    
    # Max Profit point
    df_calls['max_profit_x'] = df_calls['strike_price']
    df_calls['max_profit'] =  df_calls['call_price'] * 10000
    df_calls['max_profit_probability (%)'] = lognorm.cdf(df_calls['max_profit_x'], sigma, scale=np.exp(mu)) *100
    
    # Calculate the expected profit using 蒙特卡洛模拟方法         
    df_calls = calculate_expected_profit_rate(df_calls, mu, sigma, num_samples=1000000)
    

    
    return df_calls

'''
import numpy as np
import pandas as pd
from scipy.stats import lognorm
from scipy.integrate import simps
from utils.option_strategies import sell_call


def calculate_expected_profit_rate(df_calls, mu, sigma, num_points=1000):
    """
    Calculate the expected profit and profit rate for each call option.

    Parameters:
    - df_calls (pd.DataFrame): DataFrame with 'call_price' and 'strike_price' columns.
    - mu (float): Logarithmic mean of the lognormal distribution.
    - sigma (float): Standard deviation of the lognormal distribution.
    - num_points (int): Number of points for numerical integration.

    Returns:
    - pd.DataFrame: Updated DataFrame with 'expected_profit' and 'expected_profit_rate (%)' columns.
    """
    # Define stock price range for integration
    stock_price_range = np.linspace(0, 5 * np.exp(mu), num_points)
    pdf = lognorm.pdf(stock_price_range, sigma, scale=np.exp(mu))

    # Initialize result columns
    df_calls['expected_profit'] = 0.0
    df_calls['expected_profit_rate (%)'] = 0.0

    for index, row in df_calls.iterrows():
        call_price = row['call_price']
        strike_price = row['strike_price']

        # Calculate profit for selling a call at each stock price
        profits = sell_call(stock_price_range, call_price, strike_price)
        

        # Numerically integrate to calculate expected profit
        expected_profit = simps(profits * pdf, stock_price_range)

        # Calculate cost (assume 100% margin required)
        cost = max((strike_price - call_price) * 10000, 0)  # Prevent negative cost

        # Calculate expected profit rate
        expected_profit_rate = (expected_profit / cost) * 100 if cost > 0 else 0

        # Update the DataFrame
        df_calls.at[index, 'expected_profit'] = expected_profit
        df_calls.at[index, 'expected_profit_rate (%)'] = expected_profit_rate

    return df_calls


def naked_short_call_evaluation(stock_price, df, sigma):
    """
    Evaluate key metrics for selling call options.

    Parameters:
    - stock_price: float, current stock price.
    - df: DataFrame containing 'call_price' and 'strike_price' columns.
    - sigma: float, volatility (standard deviation) of the stock.

    Returns:
    - pd.DataFrame: Updated DataFrame with key metrics for evaluation.
    """
    # Lognormal mean
    mu = np.log(stock_price)

    df_calls = df.copy()

    # Break-even point calculations
    df_calls['break_even_x'] = df_calls['strike_price'] + df_calls['call_price']
    df_calls['break_even_probability (%)'] = lognorm.cdf(
        df_calls['break_even_x'], sigma, scale=np.exp(mu)) * 100

    # Max Profit point calculations
    df_calls['max_profit_x'] = df_calls['strike_price']
    df_calls['max_profit'] = df_calls['call_price'] * 10000
    df_calls['max_profit_probability (%)'] = lognorm.cdf(
        df_calls['max_profit_x'], sigma, scale=np.exp(mu)) * 100

    # Calculate the expected profit and profit rate using numerical integration
    df_calls = calculate_expected_profit_rate(df_calls, mu, sigma)

    return df_calls

